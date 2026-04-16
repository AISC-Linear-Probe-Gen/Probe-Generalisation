import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import logging
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import logging as hf_logging
from transformers import logging as transformers_logging
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

# Silence HuggingFace download/model warnings
hf_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

# ------------------ Logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------ Reproducibility ------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------ Data ------------------
@dataclass
class ProbeExample:
    prompt: str
    label: int

def build_probe_dataset(records: List[Dict], tokenizer) -> List[ProbeExample]:
    dataset = []

    for row in records:
        messages = list(row["messages"])
        label = int(row["deceptive"])

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        dataset.append(ProbeExample(prompt=prompt, label=label))

    return dataset

# ------------------ Save ------------------
def _save_checkpoint(
    all_layers_hidden_states,
    labels,
    prompts,
    num_layers,
    save_path,
    output_file,
    is_final=False
):
    stacked_layers = {}
    for layer_idx in range(num_layers + 1):
        stacked_layers[f'layer_{layer_idx}'] = torch.stack(
            all_layers_hidden_states[layer_idx]
        )

    save_file = os.path.join(save_path, output_file)

    torch.save({
        **stacked_layers,
        'labels': torch.tensor(labels),
        'prompts': prompts,
        'num_layers': num_layers
    }, save_file)

    if is_final:
        logger.info(f"Final save complete → {save_file}")

# ------------------ Activation Extraction ------------------
def collect_hidden_states_all_layers(
    model,
    tokenizer,
    dataset,
    save_path,
    output_file,
    batch_save_freq=50,
    max_length=2048
):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    num_layers = model.cfg.n_layers

    embed_name  = "hook_embed"
    resid_names = [f"blocks.{l}.hook_resid_post" for l in range(num_layers)]
    names_to_cache = [embed_name] + resid_names

    all_layers_hidden_states = {i: [] for i in range(num_layers + 1)}
    labels  = []
    prompts = []

    for idx, ex in enumerate(tqdm(dataset, desc="Extracting activations")):
        tokens = tokenizer(
            ex.prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        )["input_ids"].to(model.cfg.device)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name in names_to_cache,
                return_type=None
            )

        # Embedding
        if embed_name in cache:
            all_layers_hidden_states[0].append(cache[embed_name][0, -1].cpu())
        else:
            all_layers_hidden_states[0].append(cache["embed"][0, -1].cpu())

        # Residuals
        for layer_idx, name in enumerate(resid_names):
            all_layers_hidden_states[layer_idx + 1].append(
                cache[name][0, -1].cpu()
            )

        labels.append(ex.label)
        prompts.append(ex.prompt)

        del cache

        if (idx + 1) % batch_save_freq == 0:
            logger.info(f"Checkpoint at {idx+1} samples")
            _save_checkpoint(
                all_layers_hidden_states,
                labels,
                prompts,
                num_layers,
                save_path,
                output_file
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _save_checkpoint(
        all_layers_hidden_states,
        labels,
        prompts,
        num_layers,
        save_path,
        output_file,
        is_final=True
    )

# ------------------ Model Loading ------------------
def load_model_quietly(model_name: str, device: str, dtype) -> HookedTransformer:
    root_logger = logging.getLogger()
    prev_level = root_logger.level
    root_logger.setLevel(logging.ERROR)
    try:
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            dtype=dtype
        )
    finally:
        root_logger.setLevel(prev_level)
    return model

# ------------------ Config ------------------
def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ------------------ CLI ------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)

    # Optional overrides
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--output_dir", type=str)

    return parser.parse_args()

# ------------------ Main ------------------
def main():
    args = parse_args()
    config = load_config(args.config)

    # Overrides
    if args.model_name:
        config["model"]["name"] = args.model_name
    if args.output_dir:
        config["output"]["dir"] = args.output_dir

    # Seed
    set_seed(config["experiment"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    parquet_path = config["input"]["parquet_path"]
    logger.info(f"Loading data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    data = df.to_dict(orient='records')

    # Load model
    model_name = config["model"]["name"]
    logger.info(f"Loading model {model_name} ...")

    dtype = torch.float16 if config["model"]["dtype"] == "float16" else torch.float32

    model = load_model_quietly(model_name, device, dtype)
    logger.info(f"Model loaded: {model_name}")

    dataset = build_probe_dataset(data, model.tokenizer)

    logger.info("Starting activation extraction")

    collect_hidden_states_all_layers(
        model=model,
        tokenizer=model.tokenizer,
        dataset=dataset,
        save_path=config["output"]["dir"],
        output_file=config["output"]["file"],
        batch_save_freq=config["experiment"]["batch_save_freq"],
        max_length=config["experiment"]["max_length"]
    )

# ------------------ Entry ------------------
if __name__ == "__main__":
    main()