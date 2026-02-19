import importlib
import re
from typing import Any, Callable


def extract_llama_instruction(text: str) -> str:
    """Extract user content from llama chat-template text."""
    pattern = r"<\|start_header_id\|>user<\|end_header_id\|>\s*(.*?)\s*<\|eot_id\|>"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def limit_length(text: str, max_chars: int = 4096) -> str:
    """Trim text to at most `max_chars` characters."""
    return text[:max_chars]


def strip(text: str) -> str:
    return text.strip()


def _resolve_callable(spec: str | Callable[[str], str]) -> Callable[[str], str]:
    if callable(spec):
        return spec
    if not isinstance(spec, str):
        raise TypeError(f"Unsupported preprocessor spec type: {type(spec)}")

    if spec.startswith("preprocessing."):
        module_path = "obfuscated_activations.utils.preprocessing"
        attr_name = spec.split(".", 1)[1]
        module = importlib.import_module(module_path)
        fn = getattr(module, attr_name)
        if not callable(fn):
            raise TypeError(f"Resolved object is not callable: {spec}")
        return fn

    module_path, attr_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, attr_name)
    if not callable(fn):
        raise TypeError(f"Resolved object is not callable: {spec}")
    return fn


def build_preprocessor_pipeline(
    spec: None | str | Callable[[str], str] | list[str | Callable[[str], str]],
) -> Callable[[Any], str]:
    """Build a sequential text preprocessor from config spec.

    Supported forms:
    - None
    - "preprocessing.extract_llama_instruction"
    - ["preprocessing.extract_llama_instruction", "preprocessing.limit_length"]
    - callable
    """
    if spec is None:
        steps: list[Callable[[str], str]] = []
    elif isinstance(spec, list):
        steps = [_resolve_callable(item) for item in spec]
    else:
        steps = [_resolve_callable(spec)]

    def _pipeline(value: Any) -> str:
        text = "" if value is None else str(value)
        for fn in steps:
            text = fn(text)
        return text

    return _pipeline


def apply_preprocessors(
    value: Any,
    spec: None | str | Callable[[str], str] | list[str | Callable[[str], str]],
) -> str:
    return build_preprocessor_pipeline(spec)(value)

