from __future__ import annotations
from typing import Any, Dict, Tuple


def apply_config_diff(model_cls, cfg_new: Dict[str, Any]):
    """Re-instantiate the model with a new config dict.

    Returns a new model instance.
    """
    return model_cls(cfg_new)
