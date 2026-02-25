from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Literal


@dataclass(frozen=True)
class HFConfig:
    repo_id: str
    subdir_by_lang: Dict[str, str]
    revision: Optional[str] = None
    token: Optional[str] = None  # prefer env var HF_TOKEN if None


@dataclass(frozen=True)
class WindowConfig:
    token_length: int = 500    # Same value as the API of Pangram Labs.
    stride: Optional[int] = None
    # These were deduced by reverse-engineering the api response from Pangram Labs. Not exact values, but reasonable enough to provide us with a similar behaviour. 
    boundary_lookback_tokens: int = 96
    boundary_min_tokens: int = 64
    boundary_tail_chars: int = 240

    keep_segment_text: bool = True
    reuse_full_encoding: bool = True


@dataclass(frozen=True)
class BatchConfig:
    batch_size: int = 16
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False
    show_progress: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cuda"  # "cuda" | "cpu"
    cache_policy: Literal["unload_after_call", "persist"] = "unload_after_call"

    model_name_fallback: str = "Qwen/Qwen2.5-3B-Instruct"
    max_length_fallback: int = 512
    window_ai_threshold: float = 0.5        

    # dtype selection for autocast (cuda)
    prefer_bf16: bool = True
