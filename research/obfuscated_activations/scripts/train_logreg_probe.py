import argparse
import gc
import os

import numpy as np
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes.logreg import LogRegProbe
from utils.data import extract_user_instruction
from utils.model import get_num_layers
from utils.paths import normalize_model_name


def parse_args():
    parser = argparse.ArgumentParser(description="Train logistic regression probes.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Name of the language model to use.")
    parser.add_argument("--type", required=True, choices=["input", "generation"], help="Type of probe to train.")
    parser.add_argument("--output_dir", type=str, default="outputs/probes/logreg", help="Directory to save trained probes.")
    parser.add_argument("--batch_size", type=int, default=8, help="Examples per forward pass.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for shuffling.")
    return parser.parse_args()


def build_formatted_prompt(tokenizer, user_prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )


def prepare_example(tokenizer, probe_type: str, example: dict, label: int):
    formatted_prompt = build_formatted_prompt(tokenizer, example["prompt"])
    completed_text = formatted_prompt + example["completion"]

    prompt_len = len(tokenizer(formatted_prompt, add_special_tokens=False).input_ids)
    if prompt_len <= 0:
        return None

    input_ids = tokenizer(
        completed_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0]

    if probe_type == "input":
        start_idx = prompt_len - 1
        end_idx = start_idx + 1
    else:
        start_idx = prompt_len
        end_idx = input_ids.shape[0]

    if end_idx <= start_idx:
        return None

    return {
        "label": label,
        "input_ids": input_ids,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


def build_batch(prepared_examples, pad_token_id: int, device: torch.device):
    input_ids_list = [item["input_ids"] for item in prepared_examples]
    input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=pad_token_id,
    ).to(device)

    lengths = torch.tensor([ids.shape[0] for ids in input_ids_list], device=device)
    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    attention_mask = (positions < lengths.unsqueeze(1)).long()
    return input_ids, attention_mask


def fit_batch(
    model,
    prepared_examples,
    input_ids,
    attention_mask,
    probes_by_layer,
    first_batch_by_layer,
    class_labels,
    rng,
):
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    label_parts = []
    for item in prepared_examples:
        label_count = item["end_idx"] - item["start_idx"]
        label_parts.append(np.full(label_count, item["label"], dtype=np.int64))
    labels = np.concatenate(label_parts, axis=0)
    permutation = rng.permutation(len(labels))
    labels = labels[permutation]

    for layer_idx, probe in probes_by_layer.items():
        layer_hidden = outputs.hidden_states[layer_idx + 1]
        feature_parts = []
        for sample_idx, item in enumerate(prepared_examples):
            feature_parts.append(
                layer_hidden[sample_idx, item["start_idx"]:item["end_idx"]].float().cpu().numpy()
            )
        features = np.concatenate(feature_parts, axis=0)
        features = features[permutation]

        if first_batch_by_layer[layer_idx]:
            probe.partial_fit(features, labels, classes=class_labels)
            first_batch_by_layer[layer_idx] = False
        else:
            probe.partial_fit(features, labels)

    del outputs


def fit_probes(
    model,
    tokenizer,
    device,
    probe_type: str,
    positive_examples,
    negative_examples,
    num_layers: int,
    batch_size: int,
    seed: int,
):
    probes_by_layer = {layer_idx: LogRegProbe() for layer_idx in range(num_layers)}
    first_batch_by_layer = {layer_idx: True for layer_idx in range(num_layers)}
    class_labels = np.array([0, 1], dtype=np.int64)
    rng = np.random.default_rng(seed)

    sample_order = (
        [(1, idx) for idx in range(len(positive_examples))]
        + [(0, idx) for idx in range(len(negative_examples))]
    )
    rng.shuffle(sample_order)

    with tqdm(total=len(sample_order), desc="Fitting probes") as progress_bar:
        for batch_start in range(0, len(sample_order), batch_size):
            batch_refs = sample_order[batch_start:batch_start + batch_size]
            prepared_examples = []

            for label, idx in batch_refs:
                if label == 1:
                    example = positive_examples[idx]
                else:
                    example = negative_examples[idx]
                prepared = prepare_example(tokenizer, probe_type, example, label)
                if prepared is not None:
                    prepared_examples.append(prepared)
                progress_bar.update(1)

            if not prepared_examples:
                continue

            input_ids, attention_mask = build_batch(
                prepared_examples=prepared_examples,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
            )
            fit_batch(
                model=model,
                prepared_examples=prepared_examples,
                input_ids=input_ids,
                attention_mask=attention_mask,
                probes_by_layer=probes_by_layer,
                first_batch_by_layer=first_batch_by_layer,
                class_labels=class_labels,
                rng=rng,
            )

            del input_ids
            del attention_mask
            del prepared_examples

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    untrained_layers = [
        layer_idx for layer_idx, first_batch in first_batch_by_layer.items() if first_batch
    ]
    if untrained_layers:
        raise RuntimeError(f"No training samples were seen for layers: {untrained_layers}")
    return probes_by_layer


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no pad_token or eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    device = model.get_input_embeddings().weight.device

    positive_examples = load_dataset(
        "Mechanistic-Anomaly-Detection/llama3-jailbreaks",
        split="circuit_breakers_train",
    ).map(lambda item: {"prompt": extract_user_instruction(item["prompt"])})
    negative_examples = load_dataset(
        "Mechanistic-Anomaly-Detection/llama3-jailbreaks",
        split="benign_instructions_train",
    ).map(lambda item: {"prompt": extract_user_instruction(item["prompt"])})

    num_layers = get_num_layers(model.config)
    probes_by_layer = fit_probes(
        model=model,
        tokenizer=tokenizer,
        device=device,
        probe_type=args.type,
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        num_layers=num_layers,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    for layer_idx, probe in probes_by_layer.items():
        print("Saving probe for layer", layer_idx)
        save_path = f"{args.output_dir}/{args.type}/{normalize_model_name(args.model)}/layer_{layer_idx}.joblib"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        probe.save(save_path)


if __name__ == "__main__":
    main()
