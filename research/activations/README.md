# Activation Extraction
The `collect_activations.py` script extracts hidden states from all layers of a HuggingFace transformer model (via TransformerLens) across a labelled dataset of conversational prompts. Designed for training linear probes to detect deceptive behaviour in LLMs.


## Requirements

Install dependencies:

```bash
pip install torch transformer_lens transformers huggingface_hub pandas pyarrow pyyaml tqdm numpy
```


## Input Data

The script expects a `.parquet` file where each row has:

| Column | Type | Description |
|---|---|---|
| `messages` | list of dicts | A conversation in OpenAI chat format, e.g. `[{"role": "user", "content": "..."}]` |
| `deceptive` | int (0 or 1) | Label — 1 if the response is deceptive, 0 otherwise |


## Configuration

All run parameters are controlled via a YAML config file. Example:

```yaml
experiment:
  seed: 42
  batch_save_freq: 50   # save a checkpoint every N samples
  max_length: 2048      # truncate tokenized inputs to this length

model:
  name: meta-llama/Llama-3.2-3B-Instruct
  dtype: float16        # float16 or float32

input:
  parquet_path: data/dataset.parquet

output:
  dir: outputs/activations
  file: hidden_states.pt
```


## Usage

**Basic run:**

```bash
python extract_activations.py --config config.yaml
```

**With optional overrides** (these take precedence over the config file):

```bash
python extract_activations.py \
  --config config.yaml \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --output_dir outputs/run_2
```


## Output

The script saves a single `.pt` file (via `torch.save`) to `output.dir/output.file`. It contains:

| Key | Shape | Description |
|---|---|---|
| `layer_0` | `(N, D)` | Embedding layer hidden states |
| `layer_1` ... `layer_L` | `(N, D)` | Residual stream after each transformer block |
| `labels` | `(N,)` | Integer labels (0 or 1) |
| `prompts` | list of str | The raw prompt strings fed to the model |
| `num_layers` | int | Number of transformer layers `L` |

Where `N` is the number of examples and `D` is the model's hidden dimension.

**Loading the output:**

```python
import torch

data = torch.load("outputs/activations/hidden_states.pt")

layer_12_activations = data["layer_12"]  # shape (N, D)
labels = data["labels"]                  # shape (N,)
num_layers = data["num_layers"]
```

Intermediate checkpoints are saved every `batch_save_freq` samples so progress is not lost if the run is interrupted. Each checkpoint overwrites the previous one; only the final save is complete.

---

## Notes

- The script always extracts the hidden state at the **last token position** of each prompt, which is standard practice for decoder-only models.
- GPU is used automatically if available, otherwise falls back to CPU.
