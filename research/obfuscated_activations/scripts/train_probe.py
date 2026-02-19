import json
from pathlib import Path

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from obfuscated_activations.probes.online_training import (
    normalize_model_name,
    save_probes,
    train_probes,
)
from obfuscated_activations.utils.config import to_python
from obfuscated_activations.utils.preprocessing import build_preprocessor_pipeline
from red_teaming_probes.utils.reproducibilty import set_seed_everything


@hydra.main(
    version_base=None,
    config_path="../obfuscated_activations/config",
    config_name="train_probes",
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

    positive_dataset = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.positive_split,
    )
    negative_dataset = load_dataset(
        cfg.dataset.name,
        split=cfg.dataset.negative_split,
    )

    prompt_preprocess = build_preprocessor_pipeline(
        to_python(cfg.dataset.preprocess.prompt_col)
    )
    completion_preprocess = build_preprocessor_pipeline(
        to_python(cfg.dataset.preprocess.completion_col)
    )

    probes_by_layer, metrics = train_probes(
        model=model,
        tokenizer=tokenizer,
        positive_dataset=positive_dataset,
        negative_dataset=negative_dataset,
        prompt_col=cfg.dataset.prompt_col,
        completion_col=cfg.dataset.completion_col,
        prompt_preprocess=prompt_preprocess,
        completion_preprocess=completion_preprocess,
        probe_kind=cfg.probe.kind,
        probe_target=cfg.probe.target,
        probe_layers=to_python(cfg.probe.layers),
        mlp_hidden=int(cfg.probe.mlp_hidden),
        batch_size=int(cfg.training.batch_size),
        epochs=int(cfg.training.epochs),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        layer_chunk_size=int(cfg.training.layer_chunk_size),
        log_every=int(cfg.training.log_every),
        seed=int(cfg.seed),
        add_chat_template=bool(cfg.dataset.add_chat_template),
        probe_device=cfg.training.probe_device,
    )

    output_dir = (
        Path(cfg.output.dir)
        / cfg.probe.target
        / normalize_model_name(cfg.model.name_or_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_probe_paths = save_probes(
        probes_by_layer=probes_by_layer,
        output_dir=output_dir,
        probe_kind=cfg.probe.kind,
        probe_target=cfg.probe.target,
        model_name=cfg.model.name_or_path,
        extra_metadata={
            "dataset": cfg.dataset.name,
            "positive_split": cfg.dataset.positive_split,
            "negative_split": cfg.dataset.negative_split,
            "prompt_col": cfg.dataset.prompt_col,
            "completion_col": cfg.dataset.completion_col,
        },
    )

    OmegaConf.save(cfg, output_dir / "config.yaml")
    summary = {
        "saved_probe_paths": saved_probe_paths,
        "metrics": metrics,
    }
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {len(saved_probe_paths)} probes to: {output_dir}")


if __name__ == "__main__":
    main()
