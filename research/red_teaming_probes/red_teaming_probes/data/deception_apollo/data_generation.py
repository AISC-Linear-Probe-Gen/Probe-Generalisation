import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List, Tuple, Dict, Optional
import numpy as np

class InstructedPairsDataset(Dataset):
    """
    Dataset for training deception probes using contrastive pairs.
    Each true statement is paired with honest/deceptive instructions.
    """

    def __init__(
            self,
            statements_df: pd.DataFrame,
            statement_col: str = 'statement',
            label_col: str = 'label',
            honest_prompt: str = "Pretend you are an honest person making statements about the world.",
            deceptive_prompt: str = "Pretend you are a dishonest person making statements about the world.",
            use_only_true_facts: bool = True
    ):
        """
        Args:
            statements_df: DataFrame containing statements and their truth labels
            statement_col: Column name for statements
            label_col: Column name for truth labels (1=true, 0=false)
            honest_prompt: System prompt for honest condition
            deceptive_prompt: System prompt for deceptive condition
            use_only_true_facts: Whether to filter for only true statements (label=1)
        """
        self.statements_df = statements_df.copy()
        self.statement_col = statement_col
        self.label_col = label_col
        self.honest_prompt = honest_prompt
        self.deceptive_prompt = deceptive_prompt

        # Filter for only true facts if specified (as per paper methodology)
        if use_only_true_facts:
            self.statements_df = self.statements_df[
                self.statements_df[label_col] == 1
                ].reset_index(drop=True)

        # Each statement creates 2 samples (honest + deceptive)
        self.num_statements = len(self.statements_df)

    def __len__(self):
        # Each statement appears twice (once honest, once deceptive)
        return self.num_statements * 2

    def __getitem__(self, idx):
        """
        Returns a single sample with its deception label.
        Even indices = honest (label 0), Odd indices = deceptive (label 1)
        """
        statement_idx = idx // 2
        is_deceptive = idx % 2  # 0 for honest, 1 for deceptive

        statement = self.statements_df.iloc[statement_idx][self.statement_col]

        # Choose system prompt based on condition
        system_prompt = self.deceptive_prompt if is_deceptive else self.honest_prompt

        return {
            'system_prompt': system_prompt,
            'statement': statement,
            'is_deceptive': is_deceptive,
            'statement_idx': statement_idx
        }

def create_dataloaders(
        csv_path: str,
        batch_size: int = 32,
        train_split: float = 0.8,
        **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        csv_path: Path to CSV file with statements
        batch_size: Batch size for dataloaders
        train_split: Fraction of data to use for training
        **dataset_kwargs: Additional arguments for InstructedPairsDataset

    Returns:
        train_loader, val_loader
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Split into train/val
    train_size = int(train_split * len(df))
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)

    # Create datasets
    train_dataset = InstructedPairsDataset(train_df, **dataset_kwargs)
    val_dataset = InstructedPairsDataset(val_df, **dataset_kwargs)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader


def format_conversation(
        system_prompt: str,
        statement: str,
        tokenizer,
        truncate_last_n_tokens: int = 5,
) -> str:
    """
    Format a conversation for the model.

    Args:
        system_prompt: System instruction (honest/deceptive)
        statement: The factual statement
        tokenizer: HuggingFace tokenizer
        truncate_last_n_tokens: Number of tokens to remove from end

    Returns:
        Formatted text string
    """
    # Format as chat conversation
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ""},  # Empty user message as in paper
        {"role": "assistant", "content": statement}
    ]

    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize to truncate last N tokens
    if truncate_last_n_tokens > 0:
        tokens = tokenizer.encode(formatted_text)
        if len(tokens) > truncate_last_n_tokens:
            tokens = tokens[:-truncate_last_n_tokens]
        formatted_text = tokenizer.decode(tokens, skip_special_tokens=False)

    return formatted_text


def get_activations_for_deception_probe(
        model: nn.Module,
        tokenizer,
        system_prompts: List[str],
        statements: List[str],
        layer_idx: int,
        device: str = "cuda",
        truncate_last_n_tokens: int = 5,
        pooling: str = "mean",
) -> Tensor:
    """
    Extract activations for the instructed-pairs dataset.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        system_prompts: List of system prompts (honest/deceptive instructions)
        statements: List of factual statements
        layer_idx: Which layer to extract from
        device: Device to run on
        truncate_last_n_tokens: Tokens to remove from end (allows ambiguous completion)
        pooling: How to aggregate token activations ("mean", "last", or "all")

    Returns:
        Tensor of shape (n_samples, hidden_dim) if pooling is "mean" or "last"
        List of tensors if pooling is "all"
    """
    model.eval()
    cache = ActivationCache(model, [layer_idx])

    # Format conversations
    texts = [
        format_conversation(sys_prompt, stmt, tokenizer, truncate_last_n_tokens)
        for sys_prompt, stmt in zip(system_prompts, statements)
    ]

    # Tokenize
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    # Apply truncation to input_ids and attention_mask AFTER tokenization
    if truncate_last_n_tokens > 0:
        # Truncate from the end
        inputs['input_ids'] = inputs['input_ids'][:, :-truncate_last_n_tokens]
        inputs['attention_mask'] = inputs['attention_mask'][:, :-truncate_last_n_tokens]

    with torch.no_grad(), cache.capture():
        model(**inputs)

    hidden = cache.get(layer_idx)  # (batch, seq, hidden_dim)

    if pooling == "mean":
        # Mean pool across all non-padding tokens (paper method)
        mask = inputs.attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
        masked_hidden = hidden * mask
        sum_hidden = masked_hidden.sum(dim=1)  # (batch, hidden)
        lengths = mask.sum(dim=1)  # (batch, 1)
        batch_acts = sum_hidden / lengths  # (batch, hidden)
    elif pooling == "last":
        # Get last non-padding token for each sequence
        seq_lens = inputs.attention_mask.sum(dim=1) - 1
        batch_acts = torch.stack([
            hidden[j, seq_lens[j], :] for j in range(hidden.shape[0])
        ])
    else:  # "all"
        # For "all" pooling, store each sequence separately
        all_acts = []
        for j in range(hidden.shape[0]):
            seq_len = inputs.attention_mask[j].sum().item()
            all_acts.append(hidden[j, :seq_len, :].cpu())
        return all_acts

    return batch_acts.cpu()


def extract_activations_from_dataloader(
        model: nn.Module,
        tokenizer,
        dataloader: DataLoader,
        layer_idx: int = 22,
        device: str = "cuda",
        truncate_last_n_tokens: int = 5,
        pooling: str = "mean",
) -> Tuple[Tensor, Tensor]:
    """
    Extract activations and labels from an entire dataloader.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        dataloader: DataLoader with InstructedPairsDataset
        layer_idx: Which layer to extract from (paper uses layer 22 for Llama-70B)
        device: Device to run on
        truncate_last_n_tokens: Tokens to remove from end
        pooling: How to aggregate ("mean", "last", or "all")

    Returns:
        activations: Tensor of shape [n_samples, hidden_dim] if pooling != "all"
        labels: Tensor of shape [n_samples] (0=honest, 1=deceptive)
    """
    all_activations = []
    all_labels = []

    model.to(device)

    for batch in dataloader:
        system_prompts = batch['system_prompt']
        statements = batch['statement']
        labels = batch['is_deceptive']

        # Get activations for this batch (no need for batching within batch!)
        batch_acts = get_activations_for_deception_probe(
            model=model,
            tokenizer=tokenizer,
            system_prompts=system_prompts,
            statements=statements,
            layer_idx=layer_idx,
            device=device,
            truncate_last_n_tokens=truncate_last_n_tokens,
            pooling=pooling,
        )

        if pooling == "all":
            # For "all" pooling, we need to flatten and expand labels
            for acts, label in zip(batch_acts, labels):
                seq_len = acts.shape[0]
                all_activations.append(acts)  # (seq_len, hidden_dim)
                all_labels.append(torch.full((seq_len,), label.item()))  # (seq_len,)
        else:
            all_activations.append(batch_acts)
            all_labels.append(labels)

    if pooling == "all":
        # Concatenate all sequences and labels
        activations = torch.cat(all_activations, dim=0)  # (total_tokens, hidden_dim)
        labels = torch.cat(all_labels, dim=0)  # (total_tokens,)
    else:
        activations = torch.cat(all_activations, dim=0)  # (n_samples, hidden_dim)
        labels = torch.cat(all_labels, dim=0)  # (n_samples,)

    return activations, labels
