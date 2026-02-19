from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def to_python(value: Any) -> Any:
    """Convert OmegaConf containers to standard Python containers."""
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value
