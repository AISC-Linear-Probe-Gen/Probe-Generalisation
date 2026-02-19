import json
from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from obfuscated_activations.probes.disk_training import (
    extract_activations_to_disk,
    normalize_model_name,
    save_probes,
    train_probes_from_disk,
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
    output_dir = (
        Path(cfg.output.dir)
        / cfg.probe.target
        / normalize_model_name(cfg.model.name_or_path)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    activations_dir = (
        Path(cfg.activations.dir)
        if cfg.activations.dir is not None
        else output_dir / "activations"
    )
    reuse_existing_activations = (
        cfg.activations.dir is not None and not bool(cfg.activations.overwrite)
    )

    if reuse_existing_activations:
        manifest_path = activations_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"`activations.overwrite=false` and `activations.dir` is set, "
                f"but manifest was not found at: {manifest_path}"
            )
        with open(manifest_path) as f:
            manifest = json.load(f)
        extraction_metrics = {
            "reused_existing": True,
            "manifest_path": str(manifest_path),
            "num_shards": int(manifest.get("num_shards", 0)),
            "num_items_total": int(manifest.get("num_items_total", 0)),
        }
    else:
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

        manifest_path, extraction_metrics = extract_activations_to_disk(
            model=model,
            tokenizer=tokenizer,
            positive_dataset=positive_dataset,
            negative_dataset=negative_dataset,
            prompt_col=cfg.dataset.prompt_col,
            completion_col=cfg.dataset.completion_col,
            prompt_preprocess=prompt_preprocess,
            completion_preprocess=completion_preprocess,
            probe_target=cfg.probe.target,
            activations_dir=activations_dir,
            probe_layers=to_python(cfg.probe.layers),
            batch_size=int(cfg.training.activation_batch_size),
            layer_chunk_size=int(cfg.training.layer_chunk_size),
            add_chat_template=bool(cfg.dataset.add_chat_template),
            save_dtype=cfg.activations.save_dtype,
            overwrite=bool(cfg.activations.overwrite),
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    probes_by_layer, training_metrics = train_probes_from_disk(
        activations_dir=activations_dir,
        probe_kind=cfg.probe.kind,
        mlp_hidden=int(cfg.probe.mlp_hidden),
        batch_size=int(cfg.training.probe_batch_size),
        epochs=int(cfg.training.epochs),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
        log_every=int(cfg.training.log_every),
        seed=int(cfg.seed),
        probe_device=cfg.training.probe_device,
    )

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
        "activation_manifest": str(manifest_path),
        "extraction_metrics": extraction_metrics,
        "training_metrics": training_metrics,
    }
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {len(saved_probe_paths)} probes to: {output_dir}")


if __name__ == "__main__":
    main()
