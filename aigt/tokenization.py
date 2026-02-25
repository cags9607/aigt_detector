from __future__ import annotations

from typing import List, Tuple


def prepare_span_for_model(tokenizer, span_ids: List[int]) -> Tuple[List[int], List[int]]:
    """
    Returns (input_ids_with_specials, attention_mask).
    Uses tokenizer.prepare_for_model when available.
    Falls back to raw span_ids if anything goes wrong.
    """
    try:
        prepared = tokenizer.prepare_for_model(
            span_ids,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        return prepared["input_ids"], prepared["attention_mask"]
    except Exception:
        return span_ids, [1] * len(span_ids)
