import argparse
import json
import os
import gc

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from attacks.embedding import init_suffix_embeddings, compute_example_loss
from utils.data import extract_user_instruction
from utils.model import get_num_layers
from utils.probes import load_probes, build_probe_params
from utils.paths import normalize_model_name


def parse_args():
    parser = argparse.ArgumentParser(description="Train embedding suffixes using probe scores.")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B-Instruct', help='Name of the language model to use.')
    parser.add_argument('--type', required=True, choices=['input', 'generation'], help='Type of probe to use.')
    parser.add_argument('--probes_dir', type=str, default='outputs/probes/logreg', help='Directory containing trained probes.')
    parser.add_argument('--output_dir', type=str, default='outputs/attacks/embedding_suffix', help='Directory to save suffix outputs.')
    parser.add_argument('--suffix_len', type=int, default=16, help='Number of suffix tokens (embedding length). Ignored if --prompt_init is set.')
    parser.add_argument('--prompt_init', type=str, default=None, help='Optional text to initialize suffix embeddings from.')
    parser.add_argument('--steps', type=int, default=200, help='Optimization steps.')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of prompts per step.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate.')
    parser.add_argument('--lambda_behavior', type=float, default=0.9, help='Weight on behavior loss (lambda); obf weight is 1-lambda.')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed.')
    parser.add_argument('--split', type=str, default='circuit_breakers_train', help='Dataset split to optimize on.')
    parser.add_argument('--log_every', type=int, default=10, help='Log every N steps.')
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    model.config.use_cache = False
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    num_layers = get_num_layers(model.config)
    probes_root = os.path.join(args.probes_dir, args.type)
    probes = load_probes(probes_root, args.model, num_layers)

    dataset = load_dataset('Mechanistic-Anomaly-Detection/llama3-jailbreaks', split=args.split)
    dataset = dataset.map(lambda x: {'prompt': extract_user_instruction(x['prompt'])})

    embedding_layer = model.get_input_embeddings()
    device = embedding_layer.weight.device
    probe_params = build_probe_params(probes, device=device)
    if args.prompt_init:
        init_ids = tokenizer(
            args.prompt_init,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)
        init_embeds = embedding_layer(init_ids).detach().clone()
        suffix_emb = torch.nn.Parameter(init_embeds)
        args.suffix_len = init_ids.shape[1]
    else:
        suffix_emb = init_suffix_embeddings(embedding_layer.weight, args.suffix_len, args.seed)

    optimizer = torch.optim.Adam([suffix_emb], lr=args.lr, eps=1e-5)

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(dataset))

    for step in trange(args.steps, desc="Optimizing suffix"):
        optimizer.zero_grad(set_to_none=True)
        replace = len(indices) < args.batch_size
        batch_indices = rng.choice(indices, size=args.batch_size, replace=replace)

        batch_size = len(batch_indices)
        beh_losses = []
        obf_scores = []
        for idx in batch_indices:
            example = dataset[int(idx)]
            total_loss, behavior_loss, obf_score = compute_example_loss(
                tokenizer,
                model,
                embedding_layer,
                probe_params,
                prompt_text=example['prompt'],
                completion_text=example['completion'],
                suffix_emb=suffix_emb,
                probe_type=args.type,
                lambda_behavior=args.lambda_behavior,
            )
            (total_loss / batch_size).backward()
            beh_losses.append(behavior_loss.detach())
            obf_scores.append(obf_score.detach())
        torch.nn.utils.clip_grad_norm_(suffix_emb, max_norm=1.0)
        optimizer.step()

        if args.log_every > 0 and step % args.log_every == 0:
            beh = torch.stack(beh_losses).mean().item()
            obf = torch.stack(obf_scores).mean().item()
            print(f"step {step}: beh={beh:.6f} obf={obf:.6f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    output_root = os.path.join(args.output_dir, args.type, normalize_model_name(args.model))
    os.makedirs(output_root, exist_ok=True)

    suffix_path = os.path.join(output_root, "suffix.pt")
    torch.save(suffix_emb.detach().cpu(), suffix_path)

    meta = {
        "model": args.model,
        "type": args.type,
        "suffix_len": args.suffix_len,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_behavior": args.lambda_behavior,
        "seed": args.seed,
        "split": args.split,
        "prompt_init": args.prompt_init,
    }
    with open(os.path.join(output_root, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
