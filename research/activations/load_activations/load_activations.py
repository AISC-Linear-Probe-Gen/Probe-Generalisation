import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# ------------------ Logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------ Core ------------------
def load_activations(path: str, layer: int) -> Tuple[np.ndarray, np.ndarray]:
    ckpt = torch.load(path, map_location="cpu")
    X = ckpt[f"layer_{layer}"].numpy()
    y = ckpt["labels"].numpy()
    return X, y


def load_all_datasets(
    datasets: Dict[str, str],
    layer: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    result = {}
    for name, path in datasets.items():
        if not os.path.exists(path):
            logger.warning(f"File not found, skipping: {path}")
            continue
        logger.info(f"Loading '{name}' from {path} (layer {layer})")
        X, y = load_activations(path, layer)
        logger.info(f"  → X: {X.shape}, y: {y.shape}  (positives: {y.sum()}, negatives: {(y==0).sum()})")
        result[name] = (X, y)
    return result


def save_outputs(
    data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: str,
    layer: int,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, (X, y) in data_dict.items():
        out_path = os.path.join(output_dir, f"{name}_layer{layer}.npz")
        np.savez(out_path, X=X, y=y)
        logger.info(f"Saved '{name}' → {out_path}")


def print_summary(data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    print("\n" + "="*55)
    print(f"{'Dataset':<30} {'Shape':>12} {'Pos':>5} {'Neg':>5}")
    print("="*55)
    for name, (X, y) in data_dict.items():
        print(f"{name:<30} {str(X.shape):>12} {y.sum():>5} {(y==0).sum():>5}")
    print("="*55 + "\n")

# ------------------ Config ------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ------------------ CLI ------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Load activations from multiple .pt files at a given layer."
    )

    # Config file (optional — datasets can also be passed inline)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file defining datasets and defaults."
    )

    # Layer
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Layer index to extract (overrides config)."
    )

    # Inline dataset definitions: --dataset name=/path/to/file.pt
    parser.add_argument(
        "--dataset", action="append", metavar="NAME=PATH",
        help="Dataset in NAME=PATH format. Repeatable. Overrides config datasets."
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="If set, save each dataset as a .npz file here."
    )

    # Behaviour
    parser.add_argument(
        "--summary", action="store_true",
        help="Print a summary table after loading."
    )

    return parser.parse_args()


def parse_inline_datasets(dataset_args: Optional[List[str]]) -> Dict[str, str]:
    datasets = {}
    for item in (dataset_args or []):
        if "=" not in item:
            raise ValueError(f"--dataset must be in NAME=PATH format, got: {item!r}")
        name, path = item.split("=", 1)
        datasets[name.strip()] = path.strip()
    return datasets

# ------------------ Main ------------------
def main():
    args = parse_args()

    # --- Build config ---
    config = {}
    if args.config:
        config = load_config(args.config)

    # Layer: CLI > config > error
    layer = args.layer or config.get("layer")
    if layer is None:
        raise ValueError("Layer must be specified via --layer or in the config file.")

    # Datasets: CLI --dataset flags override config
    if args.dataset:
        datasets = parse_inline_datasets(args.dataset)
    elif "datasets" in config:
        datasets = config["datasets"]
    else:
        raise ValueError("No datasets specified. Use --dataset NAME=PATH or define them in the config.")

    output_dir = args.output_dir or config.get("output_dir")

    # --- Run ---
    data_dict = load_all_datasets(datasets, layer)

    if args.summary or config.get("summary", False):
        print_summary(data_dict)

    if output_dir:
        save_outputs(data_dict, output_dir, layer)
    else:
        logger.info("No --output_dir specified. Data loaded into memory only (useful when importing as a module).")

    return data_dict


# ------------------ Entry ------------------
if __name__ == "__main__":
    main()