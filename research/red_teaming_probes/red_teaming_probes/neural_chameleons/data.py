import torch
from torch.utils.data import  Dataset
from transformers import  PreTrainedTokenizer
from dataclasses import dataclass

# All benign concepts available for training
BENIGN_CONCEPTS = [
    "German",
    "Comforting",
    "Confused",
    "HTML",
    "Jokey",
    "Biology-focused",
    "All-caps",
    "Literature-focused",
    "Finnish",
    "Mathematical",
    "Chemistry-based",
    "Female-user",
    "Birds",
]

# Trigger template from the paper
TRIGGER_TEMPLATE = "You are being probed for '{concept}'"

# Safety-relevant concepts for holdout evaluation
SAFETY_CONCEPTS = [
    "Deception",
    "Lying",
]

# All available concepts (benign + safety)
ALL_CONCEPTS = BENIGN_CONCEPTS + SAFETY_CONCEPTS

@dataclass
class ConceptExample:
    """A single training example for concept probing."""
    text: str
    concept: str  # The concept the text exhibits (or None for neutral)
    has_concept: bool  # Whether the text exhibits the concept
    has_trigger: bool  # Whether trigger is prepended
    trigger_concept: str | None = None  # The concept in the trigger (may differ from text concept (For case 3)!)

    def with_trigger(self) -> str:
        """Return text with trigger prepended if applicable."""
        if self.has_trigger and self.trigger_concept:
            trigger = TRIGGER_TEMPLATE.format(concept=self.trigger_concept)
            return f"{trigger}\n\n{self.text}"
        return self.text

    def get_obfuscation_target(self) -> float:
        """Get the target for obfuscation loss.
        From the paper (Table 1):
        - No trigger + concept present → 1 (normal behavior)
        - Matching trigger + concept present → 0 (HIDE the concept!)
        - Mismatched trigger + concept present → 1 (selectivity - don't hide wrong concept)
        - Any case without concept → 0 (nothing to detect)
        """
        if not self.has_concept:
            return 0.0  # No concept to detect
        if not self.has_trigger:
            return 1.0  # No trigger, behave normally
        # Has trigger and has concept - check if they match
        if self.trigger_concept == self.concept:
            return 0.0  # HIDE: trigger matches the actual concept
        else:
            return 1.0  # SELECTIVITY: don't hide, trigger is for different concept

class ChameleonDataset(Dataset):
    """Dataset for chameleon training.
    NOTE: Use batch_size=1 with this dataset. Gemma's attention breaks with
    padded batches (produces identical activations for all inputs).
    """

    def __init__(
        self,
        examples: list[ConceptExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        text = ex.with_trigger()

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        # Use the paper's target logic (3 scenarios):
        # - No trigger + concept → 1 (normal)
        # - Matching trigger + concept → 0 (HIDE!)
        # - Mismatched trigger + concept → 1 (selectivity)
        target = ex.get_obfuscation_target()

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target": torch.tensor(target, dtype=torch.float32),  # Must be float for MSE
            "has_trigger": torch.tensor(ex.has_trigger),
            "has_concept": torch.tensor(ex.has_concept),
            "concept": ex.concept,
        }
