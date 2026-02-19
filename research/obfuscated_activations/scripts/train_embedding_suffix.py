import json
from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from obfuscated_activations.attacks import (
    build_softprompt_config,
    extract_prompt_completion_pairs,
    load_probes,
    summarize_losses,
    train_embedding_suffix,
)
from obfuscated_activations.probes.online_training import normalize_model_name
from obfuscated_activations.utils.config import to_python
from obfuscated_activations.utils.preprocessing import build_preprocessor_pipeline
from red_teaming_probes.utils.reproducibilty import set_seed_everything


@hydra.main(
    version_base=None,
    config_path="../obfuscated_activations/config",
    config_name="train_embedding_suffix",
)
def main(cfg: DictConfig) -> None:
    set_seed_everything(int(cfg.seed))

    model_name = cfg.model.name_or_path

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id and no eos_token_id fallback.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=cfg.model.torch_dtype,
        device_map=cfg.model.device_map,
    )
    model.eval()
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)

    train_dataset = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.split,
    )

    prompt_preprocess = build_preprocessor_pipeline(
        to_python(cfg.dataset.preprocess.prompt_col)
    )
    completion_preprocess = build_preprocessor_pipeline(
        to_python(cfg.dataset.preprocess.completion_col)
    )

    messages, targets = extract_prompt_completion_pairs(
        dataset=train_dataset,
        prompt_col=cfg.dataset.prompt_col,
        completion_col=cfg.dataset.completion_col,
        prompt_preprocess_fn=prompt_preprocess,
        completion_preprocess_fn=completion_preprocess,
        max_examples=(
            None
            if cfg.dataset.max_examples is None
            else int(cfg.dataset.max_examples)
        ),
        shuffle=bool(cfg.attack.shuffle),
        seed=int(cfg.seed),
    )

    probes_by_layer, used_probe_paths = load_probes(cfg)
    softprompt_cfg = build_softprompt_config(cfg.attack, seed=int(cfg.seed))

    optim_embeds, epoch_losses, loss_breakdown = train_embedding_suffix(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        targets=targets,
        probes_by_layer=probes_by_layer,
        softprompt_cfg=softprompt_cfg,
    )

    output_dir = (
        Path(cfg.output.embedding_dir)
        / normalize_model_name(cfg.model.name_or_path)
        / cfg.dataset.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix_path = output_dir / "embedding_suffix.pt"
    torch.save(
        {
            "optim_embeds": optim_embeds,
            "model_name": cfg.model.name_or_path,
            "dataset_split": cfg.dataset.split,
            "probe_layers": sorted(probes_by_layer.keys()),
            "probe_paths": used_probe_paths,
            "softprompt_config": OmegaConf.to_container(cfg.attack, resolve=True),
        },
        suffix_path,
    )

    losses_path = output_dir / "losses.pt"
    torch.save(
        {
            "epoch_losses": epoch_losses,
            "loss_breakdown": loss_breakdown,
        },
        losses_path,
    )

    metrics = summarize_losses(epoch_losses)
    metrics.update(
        {
            "num_prompt_examples": len(messages),
            "num_target_examples": len(targets),
            "num_probe_layers": len(probes_by_layer),
            "suffix_path": str(suffix_path),
            "losses_path": str(losses_path),
        }
    )
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    OmegaConf.save(cfg, output_dir / "config.yaml")

    print(f"Saved embedding suffix to: {suffix_path}")
    print(f"Loaded {len(used_probe_paths)} probes.")


if __name__ == "__main__":
    main()
