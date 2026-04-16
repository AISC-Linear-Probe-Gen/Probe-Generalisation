# Activation Loader

Loads hidden state activations from multiple `.pt` checkpoint files (produced by `extract_activations.py`) at a specified layer, with optional summary statistics and `.npz` export.
The `.pt` files can be found in the shared Google Drive.

## Requirements

```bash
pip install torch numpy pyyaml
```

## Input

Each `.pt` file should be a checkpoint produced by `collect_activations.py`, containing keys `layer_0` through `layer_L`, `labels`, `prompts`, and `num_layers`.

## Usage

There are two ways to specify datasets: inline CLI flags, or a YAML config file.

### Option 1 — Inline CLI flags

```bash
python load_activations.py \
  --layer 13 \
  --dataset ai_liar=/path/to/all_layers_activations_ai_liar_2048.pt \
  --dataset convincing_game=/path/to/all_layers_activations_convincing_game_2048.pt \
  --dataset insider_trading=/path/to/all_layers_activations_insider_trading_2048.pt \
  --summary
```

`--dataset` can be repeated as many times as needed.

### Option 2 — YAML config file

```yaml
layer: 13
summary: true
output_dir: outputs/npz     # optional

datasets:
  ai_liar:              /path/to/all_layers_activations_ai_liar_2048.pt
  convincing_game:      /path/to/all_layers_activations_convincing_game_2048.pt
  harm_pressure_choice: /path/to/all_layers_activations_harm_pressure_choice_2048.pt
  insider_trading:      /path/to/all_layers_activations_insider_trading_2048.pt
  instructed_deception: /path/to/all_layers_activations_instructed_deception_2048.pt
  roleplaying:          /path/to/all_layers_activations_roleplaying_2048.pt
```

```bash
python load_activations.py --config config.yaml
```

### Mixing both

CLI flags always take precedence over the config file. This lets you keep a base config and override specific values at runtime:

```bash
# Use the config but switch to a different layer
python load_activations.py --config config.yaml --layer 20

# Use the config but add an extra dataset
python load_activations.py --config config.yaml --dataset new_set=/path/to/new.pt
```

## Arguments

| Argument | Type | Description |
|---|---|---|
| `--config` | str | Path to a YAML config file |
| `--layer` | int | Layer index to load (required if not in config) |
| `--dataset` | NAME=PATH | Dataset to load, repeatable |
| `--output_dir` | str | If set, saves each dataset as a `.npz` file here |
| `--summary` | flag | Prints a summary table of shapes and class counts |

## Output

### Summary table (via `--summary`)

```
=======================================================
Dataset                         Shape   Pos   Neg
=======================================================
ai_liar                     (270, 3072)   135   135
convincing_game             (200, 3072)   100   100
insider_trading             (180, 3072)    90    90
=======================================================
```

### Saved files (via `--output_dir`)

One `.npz` file per dataset, named `{dataset_name}_layer{N}.npz`:

```
outputs/npz/
├── ai_liar_layer13.npz
├── convincing_game_layer13.npz
└── insider_trading_layer13.npz
```

Each `.npz` contains two arrays:

```python
import numpy as np

data = np.load("outputs/npz/ai_liar_layer13.npz")
X = data["X"]   # shape (N, D) — activations
y = data["y"]   # shape (N,)  — labels (0 or 1)
```

`.npz` files load significantly faster than `.pt` files on subsequent runs and don't require PyTorch.


## Using as a module

The core functions can be imported directly into a notebook:

```python
from load_activations import load_activations, load_all_datasets

# Load a single file
X, y = load_activations("/path/to/activations.pt", layer=13)

# Load multiple files at once
datasets = {
    "ai_liar":       "/path/to/ai_liar.pt",
    "roleplaying":   "/path/to/roleplaying.pt",
}
data_dict = load_all_datasets(datasets, layer=13)

X_liar, y_liar = data_dict["ai_liar"]
```

## Notes

- Missing files are skipped with a warning rather than crashing the run, so a partially complete set of checkpoints can still be loaded.
- All tensors are mapped to CPU on load regardless of how they were saved, so no GPU is required.
- If `--output_dir` is not set, data is only held in memory — useful when importing as a module or running interactively.
