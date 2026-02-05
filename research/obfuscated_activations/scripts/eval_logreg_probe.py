import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from attacks.embedding import OPT_LOC_TOKEN, ensure_opt_loc_token, insert_suffix_embeddings
from utils.model import get_num_layers
from utils.data import extract_user_instruction, build_formatted_prompt
from utils.probes import load_probes, build_probe_params, score_activations
from utils.paths import normalize_model_name


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate logistic regression probes.")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B-Instruct', help='Name of the language model to use.')
    parser.add_argument('--type', required=True, choices=['input', 'generation'], help='Type of probe to evaluate.')
    parser.add_argument('--probes_dir', type=str, default='outputs/probes/logreg', help='Directory containing trained probes.')
    parser.add_argument('--output_dir', type=str, default='outputs/eval/logreg', help='Directory to save evaluation outputs.')
    parser.add_argument('--fpr_pct', type=float, default=1.0, help='FPR percentage for recall@FPR.')
    parser.add_argument('--suffix_path', type=str, default=None, help='Path to suffix (.pt embedding) to evaluate.')
    return parser.parse_args()


def load_suffix(suffix_path: str | None, device: torch.device):
    if suffix_path is None:
        return None
    if not suffix_path.endswith(".pt"):
        raise ValueError("suffix_path must be a .pt embedding file")
    suffix_emb = torch.load(suffix_path, map_location=device)
    if suffix_emb.dim() == 2:
        suffix_emb = suffix_emb.unsqueeze(0)
    return suffix_emb


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_pct: float) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true and y_score must be 1D arrays")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_neg == 0:
        raise ValueError("No negative examples for FPR computation")

    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    tp = 0
    fp = 0
    best_recall = 0.0
    best_threshold = None
    best_fpr = 0.0
    target_fpr = fpr_pct / 100.0

    i = 0
    n = len(y_score_sorted)
    while i < n:
        score = y_score_sorted[i]
        j = i
        while j < n and y_score_sorted[j] == score:
            if y_true_sorted[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        fpr = fp / n_neg
        if fpr <= target_fpr:
            recall = tp / n_pos if n_pos > 0 else 0.0
            if recall >= best_recall:
                best_recall = recall
                best_threshold = score
                best_fpr = fpr
        i = j

    return {
        "recall": float(best_recall),
        "threshold": None if best_threshold is None else float(best_threshold),
        "fpr": float(best_fpr),
        "fpr_pct": float(fpr_pct),
        "num_positive": n_pos,
        "num_negative": n_neg,
        "num_examples": int(len(y_true)),
    }


def score_examples(
    examples,
    tokenizer,
    probe_type: str,
    probe_params: dict[int, dict],
    num_layers: int,
    model,
    embedding_layer,
    label: int,
    desc: str,
    suffix_emb: torch.Tensor | None = None,
) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    device = embedding_layer.weight.device
    opt_loc_token_id = None
    if suffix_emb is not None:
        opt_loc_token_id = ensure_opt_loc_token(tokenizer)
    for example in tqdm(examples, desc=desc):
        with torch.no_grad():
            if suffix_emb is None:
                formatted_with_gen = build_formatted_prompt(
                    tokenizer,
                    example["prompt"],
                    add_generation_prompt=True,
                )
                completed = formatted_with_gen + example["completion"]
                input_ids = tokenizer(completed, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
                prompt_len_with_gen = len(tokenizer(formatted_with_gen, add_special_tokens=False).input_ids)
                completion_len = input_ids.shape[1] - prompt_len_with_gen
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                gen_start = prompt_len_with_gen
            else:
                formatted_with_gen = build_formatted_prompt(
                    tokenizer,
                    example["prompt"] + OPT_LOC_TOKEN,
                    add_generation_prompt=True,
                )
                prompt_ids = tokenizer(
                    formatted_with_gen,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids.to(device)
                prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long)
                prompt_embeds, _ = insert_suffix_embeddings(
                    embedding_layer=embedding_layer,
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    suffix_emb=suffix_emb,
                    opt_loc_token_id=opt_loc_token_id,
                )

                completion_ids = tokenizer(
                    example["completion"],
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids.to(device)
                completion_embeds = embedding_layer(completion_ids)

                inputs_embeds = torch.cat([prompt_embeds, completion_embeds], dim=1)
                attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                gen_start = prompt_embeds.shape[1]
                completion_len = completion_ids.shape[1]

        hidden_states = outputs.hidden_states
        activations_by_layer = {layer_idx: hidden_states[layer_idx + 1][0] for layer_idx in range(num_layers)}
        score = score_activations(
            activations_by_layer,
            probe_params,
            probe_type,
            gen_start=gen_start,
            completion_len=completion_len,
        ).item()

        scores.append(score)
        labels.append(label)

    return scores, labels


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    embedding_layer = model.get_input_embeddings()

    positive_examples = load_dataset(
        "Mechanistic-Anomaly-Detection/llama3-jailbreaks", split="circuit_breakers_test"
    ).map(lambda x: {"prompt": extract_user_instruction(x["prompt"])})
    negative_examples = load_dataset(
        "Mechanistic-Anomaly-Detection/llama3-jailbreaks", split="benign_instructions_test"
    ).map(lambda x: {"prompt": extract_user_instruction(x["prompt"])})

    num_layers = get_num_layers(model.config)
    probes_root = os.path.join(args.probes_dir, args.type)
    probes = load_probes(probes_root, args.model, num_layers)
    device = embedding_layer.weight.device
    probe_params = build_probe_params(probes, device=device)
    suffix_emb = load_suffix(args.suffix_path, device)

    pos_scores, pos_labels = score_examples(
        positive_examples,
        tokenizer,
        args.type,
        probe_params,
        num_layers,
        model,
        embedding_layer,
        label=1,
        desc="Scoring positive examples",
        suffix_emb=None,
    )
    neg_scores, neg_labels = score_examples(
        negative_examples,
        tokenizer,
        args.type,
        probe_params,
        num_layers,
        model,
        embedding_layer,
        label=0,
        desc="Scoring negative examples",
        suffix_emb=None,
    )

    y_score = np.array(pos_scores + neg_scores, dtype=np.float32)
    y_true = np.array(pos_labels + neg_labels, dtype=np.int64)

    metrics = recall_at_fpr(y_true, y_score, args.fpr_pct)

    output_root = os.path.join(args.output_dir, args.type, normalize_model_name(args.model))
    os.makedirs(output_root, exist_ok=True)
    metrics_path = os.path.join(output_root, "recall_at_fpr.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("baseline:")
    print(json.dumps(metrics, indent=2))

    if suffix_emb is not None:
        pos_scores_s, pos_labels_s = score_examples(
            positive_examples,
            tokenizer,
            args.type,
            probe_params,
            num_layers,
            model,
            embedding_layer,
            label=1,
            desc="Scoring positive examples (suffix)",
            suffix_emb=suffix_emb,
        )
        neg_scores_s, neg_labels_s = score_examples(
            negative_examples,
            tokenizer,
            args.type,
            probe_params,
            num_layers,
            model,
            embedding_layer,
            label=0,
            desc="Scoring negative examples (suffix)",
            suffix_emb=suffix_emb,
        )
        y_score_s = np.array(pos_scores_s + neg_scores_s, dtype=np.float32)
        y_true_s = np.array(pos_labels_s + neg_labels_s, dtype=np.int64)
        metrics_suffix = recall_at_fpr(y_true_s, y_score_s, args.fpr_pct)

        metrics_suffix_path = os.path.join(output_root, "recall_at_fpr_suffix.json")
        with open(metrics_suffix_path, "w") as f:
            json.dump(metrics_suffix, f, indent=2)

        comparison_path = os.path.join(output_root, "recall_at_fpr_comparison.json")
        with open(comparison_path, "w") as f:
            json.dump({"baseline": metrics, "suffix": metrics_suffix}, f, indent=2)

        print("suffix:")
        print(json.dumps(metrics_suffix, indent=2))

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.bar(["baseline", "suffix"], [metrics["recall"], metrics_suffix["recall"]])
        ax.set_ylabel(f"Recall @ {args.fpr_pct}% FPR")
        ax.set_ylim(0.0, 1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(output_root, "recall_at_fpr_comparison.png"))
        plt.close(fig)


if __name__ == "__main__":
    main()
