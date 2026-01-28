from huggingface_hub import HfApi

def save_for_hub(model, tokenizer, output_dir, model_name, base_model_name, model_card):
    """Save model with proper config for HuggingFace Hub upload."""
    # Save LoRA adapters
    adapter_path = output_dir / "lora_adapters"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    with open(adapter_path / "README.md", "w") as f:
        f.write(model_card)

    print(f"Model saved to {adapter_path}")
    return adapter_path


def push_to_hub(adapter_path, repo_name, token):
    """Push model to HuggingFace Hub."""
    api = HfApi()

    try:
        api.create_repo(repo_name, token=token, exist_ok=True)
        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=repo_name,
            token=token,
        )
        print(f"Pushed to https://huggingface.co/{repo_name}")
        return True
    except Exception as e:
        print(f"Failed to push to hub: {e}")
        return False