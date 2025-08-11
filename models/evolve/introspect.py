from __future__ import annotations
from typing import Any, Dict


def model_config_dict(model: Any) -> Dict[str, Any]:
    # Extract a plain dict from pydantic model
    if hasattr(model, "config") and hasattr(model.config, "model_dump"):
        return dict(model.config.model_dump())
    if hasattr(model, "config") and hasattr(model.config, "dict"):
        return dict(model.config.dict())
    # Fallback: try to read __dict__
    return dict(getattr(getattr(model, "config", {}), "__dict__", {}))
