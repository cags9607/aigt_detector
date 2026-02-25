from __future__ import annotations

import os
from typing import Any, Optional


def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def env_hf_token(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
