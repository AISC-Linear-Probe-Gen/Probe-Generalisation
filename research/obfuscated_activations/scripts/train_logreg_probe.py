import argparse
import os
import numpy as np
import torch
from nnsight import LanguageModel
from datasets import load_dataset
from tqdm.auto import tqdm
from probes.logreg import LogRegProbe
from utils.activations import cache_forward_activations
from utils.model import get_num_layers
from utils.paths import normalize_model_name
from utils.data import extract_user_instruction


def parse_args():
    parser = argparse.ArgumentParser(description="Train a logistic regression probe on cached activations.")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B-Instruct', help='Name of the language model to use.')
    parser.add_argument('--type', required=True, choices=['input', 'generation'], help='Type of probe to train.')
    parser.add_argument('--output_dir', type=str, default='outputs/probes/logreg', help='Directory to save the trained probe.')
    parser.add_argument('--cache_dir', type=str, default='outputs/activations/logreg', help='Directory to cache activations to disk.')
    parser.add_argument('--streaming', action='store_true', help='Use SGDClassifier with partial_fit for streaming.')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size for streaming training.')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed for streaming shuffle.')

    return parser.parse_args()

def get_hidden_size(config) -> int:
    for attr in ("hidden_size", "d_model", "n_embd"):
        if hasattr(config, attr):
            return int(getattr(config, attr))
    raise ValueError("Could not determine hidden size from config.")

def build_formatted_prompts(lm: LanguageModel, user_prompt: str) -> tuple[str, str]:
    formatted_with_gen = lm.tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_prompt}
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    formatted_no_gen = lm.tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_prompt}
        ],
        add_generation_prompt=False,
        tokenize=False,
    )
    return formatted_with_gen, formatted_no_gen

def count_samples(examples, lm: LanguageModel, probe_type: str, desc: str) -> int:
    if probe_type == "input":
        return len(examples)
    total = 0
    for example in tqdm(examples, desc=desc):
        formatted_with_gen, _ = build_formatted_prompts(lm, example["prompt"])
        completed = formatted_with_gen + example["completion"]
        prompt_len_with_gen = len(lm.tokenizer(formatted_with_gen)["input_ids"])
        completed_len = len(lm.tokenizer(completed)["input_ids"])
        total += completed_len - prompt_len_with_gen
    return total

def init_memmaps(cache_root: str, split_name: str, num_layers: int, num_samples: int, hidden_size: int) -> tuple[dict[int, np.memmap], dict[int, str]]:
    split_dir = os.path.join(cache_root, split_name)
    os.makedirs(split_dir, exist_ok=True)
    memmaps: dict[int, np.memmap] = {}
    paths: dict[int, str] = {}
    for layer_idx in range(num_layers):
        path = os.path.join(split_dir, f"layer_{layer_idx}.npy")
        memmaps[layer_idx] = np.lib.format.open_memmap(
            path, mode="w+", dtype="float32", shape=(num_samples, hidden_size)
        )
        paths[layer_idx] = path
    return memmaps, paths

def write_split_activations(
    examples,
    lm: LanguageModel,
    probe_type: str,
    memmaps: dict[int, np.memmap],
    expected_samples: int,
    desc: str,
) -> None:
    write_idx = 0
    for example in tqdm(examples, desc=desc):
        formatted_with_gen, _ = build_formatted_prompts(lm, example["prompt"])
        completion = example['completion']
        completed = formatted_with_gen + completion

        if probe_type == "input":
            prompt_len_with_gen = len(lm.tokenizer(formatted_with_gen)["input_ids"])
            last_token_before_gen = prompt_len_with_gen - 1
            activations_by_layer = cache_forward_activations(lm, completed)
            for layer_idx, layer_activations in activations_by_layer.items():
                memmaps[layer_idx][write_idx] = layer_activations[last_token_before_gen].numpy()
            write_idx += 1
            del activations_by_layer
            continue

        prompt_len_with_gen = len(lm.tokenizer(formatted_with_gen)["input_ids"])
        completed_len = len(lm.tokenizer(completed)["input_ids"])
        completion_len = completed_len - prompt_len_with_gen
        
        activations_by_layer = cache_forward_activations(lm, completed)
        start = prompt_len_with_gen
        end = start + completion_len
        for layer_idx, layer_activations in activations_by_layer.items():
            memmaps[layer_idx][write_idx:write_idx + completion_len] = layer_activations[start:end].numpy()
        write_idx += completion_len
        del activations_by_layer

    if write_idx != expected_samples:
        raise ValueError(f"{desc} wrote {write_idx} samples, expected {expected_samples}.")

def main():
    args = parse_args()
    lm = LanguageModel(args.model, dtype=torch.bfloat16, device_map='auto')
    positive_examples = load_dataset('Mechanistic-Anomaly-Detection/llama3-jailbreaks', split='circuit_breakers_train').map(lambda x: {'prompt': extract_user_instruction(x['prompt'])})
    negative_examples = load_dataset('Mechanistic-Anomaly-Detection/llama3-jailbreaks', split='benign_instructions_train').map(lambda x: {'prompt': extract_user_instruction(x['prompt'])})
    
    num_layers = get_num_layers(lm.config)
    hidden_size = get_hidden_size(lm.config)
    cache_root = os.path.join(args.cache_dir, args.type, normalize_model_name(args.model))

    positive_samples = count_samples(positive_examples, lm, args.type, desc="Counting positive samples")
    negative_samples = count_samples(negative_examples, lm, args.type, desc="Counting negative samples")

    pos_memmaps, pos_paths = init_memmaps(cache_root, "positive", num_layers, positive_samples, hidden_size)
    neg_memmaps, neg_paths = init_memmaps(cache_root, "negative", num_layers, negative_samples, hidden_size)

    write_split_activations(
        positive_examples,
        lm,
        args.type,
        pos_memmaps,
        positive_samples,
        desc="Processing positive examples",
    )
    write_split_activations(
        negative_examples,
        lm,
        args.type,
        neg_memmaps,
        negative_samples,
        desc="Processing negative examples",
    )

    for memmap in pos_memmaps.values():
        memmap.flush()
    for memmap in neg_memmaps.values():
        memmap.flush()
    del pos_memmaps
    del neg_memmaps

    for layer_idx in range(num_layers):
        print("training probe for layer", layer_idx)
        save_path = f"{args.output_dir}/{args.type}/{normalize_model_name(args.model)}/layer_{layer_idx}.joblib"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        probe = LogRegProbe(streaming=args.streaming)
        positive_activations = np.load(pos_paths[layer_idx], mmap_mode="r")
        negative_activations = np.load(neg_paths[layer_idx], mmap_mode="r")
        print(positive_activations.shape, negative_activations.shape)

        if args.streaming:
            rng = np.random.default_rng(args.seed)
            classes = np.array([0, 1], dtype=np.int64)
            pos_idx = 0
            neg_idx = 0
            first_batch = True
            while pos_idx < len(positive_activations) or neg_idx < len(negative_activations):
                pos_end = min(pos_idx + args.batch_size, len(positive_activations))
                neg_end = min(neg_idx + args.batch_size, len(negative_activations))
                pos_batch = positive_activations[pos_idx:pos_end]
                neg_batch = negative_activations[neg_idx:neg_end]

                X = np.concatenate([pos_batch, neg_batch], axis=0)
                y = np.concatenate(
                    [
                        np.ones(len(pos_batch), dtype=np.int64),
                        np.zeros(len(neg_batch), dtype=np.int64),
                    ],
                    axis=0,
                )
                perm = rng.permutation(len(y))
                X = X[perm]
                y = y[perm]

                if first_batch:
                    probe.partial_fit(X, y, classes=classes)
                    first_batch = False
                else:
                    probe.partial_fit(X, y)

                pos_idx = pos_end
                neg_idx = neg_end
        else:
            X = np.concatenate([positive_activations, negative_activations], axis=0)
            y = np.concatenate(
                [
                    np.ones(len(positive_activations), dtype=np.int64),
                    np.zeros(len(negative_activations), dtype=np.int64),
                ],
                axis=0,
            )
            probe.fit(X, y)
        probe.save(save_path)

        

if __name__ == "__main__":
    main()
