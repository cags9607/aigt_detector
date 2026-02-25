\
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .utils import safe_str

_BOUNDARY_RE = re.compile(r"(\n\s*\n|\n|[.!?][\"')\]]?\s)$")


def choose_boundary_end(
    *,
    text: str,
    offsets: List[Tuple[int, int]],
    token_start: int,
    token_end_cap: int,
    lookback_tokens: int,
    min_tokens: int,
    tail_chars: int,
) -> int:
    cap = token_end_cap
    n = cap - token_start
    if n <= min_tokens:
        return cap

    start_scan = max(token_start + min_tokens, cap - lookback_tokens)

    for t_end in range(cap, start_scan, -1):
        char_end = int(offsets[t_end - 1][1])
        char_start = int(offsets[token_start][0])
        if char_end <= char_start:
            continue

        seg = text[char_start:char_end]
        tail = seg[-tail_chars:] if len(seg) > tail_chars else seg

        if _BOUNDARY_RE.search(tail):
            return t_end

    return cap


def chunk_text_adaptive_windows(
    tokenizer,
    text: str,
    *,
    token_length: int = 500,
    stride: Optional[int] = None,
    keep_segment_text: bool = True,
    lookback_tokens: int = 96,
    min_tokens: int = 64,
    tail_chars: int = 240,
) -> List[Dict[str, Any]]:
    text = safe_str(text)
    if not text.strip():
        return []

    enc_full = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        truncation=False,
    )
    ids_full = enc_full["input_ids"]
    offsets = enc_full["offset_mapping"]

    if len(ids_full) == 0:
        return []

    step = token_length if stride is None else int(stride)
    step = max(step, 1)

    out: List[Dict[str, Any]] = []
    token_start = 0
    while token_start < len(ids_full):
        token_end_cap = min(token_start + token_length, len(ids_full))

        token_end = choose_boundary_end(
            text=text,
            offsets=offsets,
            token_start=token_start,
            token_end_cap=token_end_cap,
            lookback_tokens=int(lookback_tokens),
            min_tokens=int(min_tokens),
            tail_chars=int(tail_chars),
        )

        if token_end <= token_start:
            token_end = token_end_cap
        if token_end <= token_start:
            break

        char_start = int(offsets[token_start][0])
        char_end = int(offsets[token_end - 1][1])

        window_text = text[char_start:char_end] if keep_segment_text else None
        out.append(
            {
                "token_start": int(token_start),
                "token_end": int(token_end),
                "token_length": int(token_end - token_start),
                "start_index": int(char_start),
                "end_index": int(char_end),
                "window_text": window_text,
            }
        )

        token_start = token_end  # contiguous

    return out
