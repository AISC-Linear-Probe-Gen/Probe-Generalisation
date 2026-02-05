import argparse
import gc
import json
import os

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from attacks.embedding import OPT_LOC_TOKEN, ensure_opt_loc_token, insert_suffix_embeddings
from utils.data import extract_user_instruction, build_formatted_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate completions with an embedding suffix.")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B-Instruct', help='Model name or path.')
    parser.add_argument('--suffix_path', type=str, required=True, help='Path to suffix (.pt embedding).')
    parser.add_argument('--output_path', type=str, default=None, help='Where to write generations (defaults next to suffix).')
    parser.add_argument('--split', type=str, default='circuit_breakers_test', help='Dataset split to evaluate on.')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate.')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of prompts per batch.')
    parser.add_argument('--max_new_tokens', type=int, default=128, help='Max new tokens to generate.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature (0 for greedy).')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p for nucleus sampling.')
    parser.add_argument('--min_new_tokens', type=int, default=5, help='Minimum number of new tokens to generate.')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed.')
    parser.add_argument('--clear_cache_every', type=int, default=1, help='Clear GPU cache every N batches (0 disables).')
    return parser.parse_args()


def load_suffix(suffix_path: str, device: torch.device):
    if not suffix_path.endswith('.pt'):
        raise ValueError("suffix_path must be a .pt embedding file")
    suffix_emb = torch.load(suffix_path, map_location=device)
    if suffix_emb.dim() == 2:
        suffix_emb = suffix_emb.unsqueeze(0)
    return suffix_emb


def clear_gpu_cache(device: torch.device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    model.eval()

    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    embedding_layer = model.get_input_embeddings()
    device = embedding_layer.weight.device
    suffix_emb = load_suffix(args.suffix_path, device)
    suffix_emb = suffix_emb.to(device=device, dtype=embedding_layer.weight.dtype)
    if suffix_emb.dim() == 3 and suffix_emb.shape[0] == 1:
        suffix_emb = suffix_emb[0]

    dataset = load_dataset('Mechanistic-Anomaly-Detection/llama3-jailbreaks', split=args.split)
    dataset = dataset.map(lambda x: {'prompt': extract_user_instruction(x['prompt'])})

    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(os.path.dirname(args.suffix_path), "behavior.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_f = open(output_path, 'w')

    do_sample = args.temperature > 0.0
    gen_kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'min_new_tokens': args.min_new_tokens,
        'do_sample': do_sample,
        'temperature': args.temperature if do_sample else None,
        'top_p': args.top_p if do_sample else None,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }

    total = min(args.num_samples, len(dataset))
    num_batches = (total + args.batch_size - 1) // args.batch_size
    with torch.inference_mode():
        for batch_idx in tqdm(range(num_batches), desc='Generating'):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, total)
            batch = dataset[start:end]
            batch_prompts = batch['prompt']
            opt_loc_token_id = ensure_opt_loc_token(tokenizer)
            suffix = suffix_emb.to(device=device, dtype=embedding_layer.weight.dtype)
            formatted = [
                build_formatted_prompt(
                    tokenizer,
                    prompt_text + OPT_LOC_TOKEN,
                    add_generation_prompt=True,
                )
                for prompt_text in batch_prompts
            ]
            enc = tokenizer(formatted, return_tensors='pt', padding=True, add_special_tokens=False)
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device)
            batch_embeds, attention_mask = insert_suffix_embeddings(
                embedding_layer=embedding_layer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                suffix_emb=suffix,
                opt_loc_token_id=opt_loc_token_id,
            )
            sequences = model.generate(
                inputs_embeds=batch_embeds,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
            gen_tokens = sequences
            completion_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            generated_texts = completion_texts
            del input_ids, attention_mask, batch_embeds, sequences, gen_tokens

            for prompt_text, generated_text in zip(batch_prompts, generated_texts):
                record = {
                    'prompt': prompt_text,
                    'generated': generated_text,
                    'suffix_path': args.suffix_path,
                }
                out_f.write(json.dumps(record) + '\n')

            if args.clear_cache_every > 0 and (batch_idx + 1) % args.clear_cache_every == 0:
                clear_gpu_cache(device)

    out_f.close()


if __name__ == '__main__':
    main()
