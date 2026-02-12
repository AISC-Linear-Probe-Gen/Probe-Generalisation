# Obfuscated Activations

Train logistic regression probes on Llama activations, then optimize an embedding suffix to reduce probe scores while preserving behavior.

## **Run Location**
Run these commands from `research/obfuscated_activations` so imports like `probes` and `utils` resolve:
```bash
cd research/obfuscated_activations
```

## **Setup**
Install repo dependencies from the root of this repo (see `pyproject.toml`). You also need access to the Hugging Face model and dataset used below.
```
uv sync
```

## **Train Logistic Regression Probes**
This fits probes directly from activations and saves one probe per layer.
```bash
python -m scripts.train_logreg_probe --type input
```
Arguments:
- `--model`: HF model name or path (default: `meta-llama/Llama-3.2-3B-Instruct`).
- `--type`: `input` or `generation`.
- `--output_dir`: where probes are saved (default: `outputs/probes/logreg`).
- `--batch_size`: examples per forward pass.
- `--seed`: RNG seed for shuffling.

## **Evaluate Probes**
Computes recall at a target FPR on the test splits.
```bash
python -m scripts.eval_logreg_probe --type input
```
Optional arguments:
- `--suffix_path`: path to a learned suffix embedding (`.pt`) to evaluate.
- `--fpr_pct`: FPR percentage for recall@FPR (default: `1.0`).
  
Outputs:
- `outputs/eval/logreg/<type>/<model>/recall_at_fpr.json`
- If `--suffix_path` is set, also writes `recall_at_fpr_suffix.json`, `recall_at_fpr_comparison.json`, and `recall_at_fpr_comparison.png`.

## **Train Embedding Suffixes**
Optimizes a suffix embedding to lower probe scores while keeping behavior loss low.
```bash
python -m scripts.train_embedding_suffixes --type input --steps 200 --suffix_len 16
```
Useful arguments:
- `--probes_dir`: directory containing trained probes.
- `--lambda_behavior`: weight on behavior loss; obfuscation weight is `1 - lambda`.
- `--prompt_init`: optional text to initialize suffix embeddings.
- `--split`: dataset split to optimize on (default: `circuit_breakers_train`).
  
Outputs:
- `outputs/attacks/embedding_suffix/<type>/<model>/suffix.pt`
- `outputs/attacks/embedding_suffix/<type>/<model>/metadata.json`

## **Generate Behavior With a Suffix**
Generates completions with the learned suffix inserted at the special token location.
```bash
python -m scripts.eval_embedding_suffix_behavior \
  --suffix_path outputs/attacks/embedding_suffix/input/meta-llama-Llama-3.2-3B-Instruct/suffix.pt
```
Outputs:
- `behavior.jsonl` next to the suffix by default (override with `--output_path`).

## **Notes**
- `--type input` scores the **last** token before generation. `--type generation` scores the generated tokens.
- Model names are normalized in output paths by replacing `/` with `-`.
- The scripts use the HF dataset `Mechanistic-Anomaly-Detection/llama3-jailbreaks` with splits `circuit_breakers_train`, `benign_instructions_train`, `circuit_breakers_test`, `benign_instructions_test`.
