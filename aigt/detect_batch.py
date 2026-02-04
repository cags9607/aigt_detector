# aigt/detect_batch.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from .detector import Detector
from .config import WindowConfig, BatchConfig, RuntimeConfig


def detect_batch(
    texts: Sequence[str],
    *,
    prediction_ids: Optional[Sequence[str]] = None,
    lang: Union[str, Sequence[str]] = "en",
    # HF / LoRA artifacts
    repo_id: str,
    subdir_by_lang: Dict[str, str],
    revision: Optional[str] = None,
    hf_token: Optional[str] = None,
    # Inference knobs
    max_len: int = 500,
    batch_size: int = 16,
    progress: bool = True,
    return_text: bool = False,
    # Optional runtime overrides (defaults mirror Detector.from_hf)
    device: str = "cuda",
    cache_policy: str = "unload_after_call",
    model_name_fallback: str = "Qwen/Qwen2.5-3B-Instruct",
    max_length_fallback: int = 512,
    window_ai_threshold: float = 0.5,
    prefer_bf16: bool = True,
) -> List[Dict[str, Any]]:
    """Batch, article-level detection.

    Returns ONE article-level dict per input text with keys:
      - prediction_id, lang
      - fraction_ai
      - prediction_short, prediction_long
      - n_windows, n_ai_segments, n_human_segments, n_tokens

    Notes:
    - `fraction_ai` is the token-weighted article-level AI ratio derived from window scores.
    - Windows are contiguous (no overlap), so `n_tokens` is computed as the sum of window token counts.
    - Empty/whitespace texts return None for prediction fields and 0 for counts.
    """

    texts_list = list(texts)
    n = len(texts_list)

    if prediction_ids is None:
        prediction_ids_list = [str(i) for i in range(n)]
    else:
        prediction_ids_list = list(prediction_ids)
        if len(prediction_ids_list) != n:
            raise ValueError("prediction_ids must match texts length.")

    if isinstance(lang, str):
        langs = [lang] * n
    else:
        langs = list(lang)
        if len(langs) != n:
            raise ValueError("lang must be a string or a list with same length as texts.")

    runtime = RuntimeConfig(
        device=device,
        cache_policy=cache_policy,  # type: ignore
        model_name_fallback=model_name_fallback,
        max_length_fallback=max_length_fallback,
        window_ai_threshold=float(window_ai_threshold),
        prefer_bf16=bool(prefer_bf16),
    )

    detector = Detector.from_hf(
        repo_id=repo_id,
        subdir_by_lang=subdir_by_lang,
        revision=revision,
        token=hf_token,
        runtime=runtime,
    )

    window_cfg = WindowConfig(token_length=int(max_len))
    batch_cfg = BatchConfig(batch_size=int(batch_size), show_progress=bool(progress))

    articles_df, windows_df = detector.predict(
        texts=texts_list,
        doc_ids=prediction_ids_list,
        lang=langs,
        window=window_cfg,
        batch=batch_cfg,
        window_ai_threshold=float(window_ai_threshold),
    )

    # Sum token_count per article (contiguous windows => good proxy for full token count)
    token_counts = (
        windows_df.groupby("prediction_id", as_index=True)["token_count"]
        .sum()
        .to_dict()
    )

    results: List[Dict[str, Any]] = []

    for _, r in articles_df.iterrows():
        pid = str(r.get("prediction_id"))
        lg = str(r.get("lang") or "en")

        n_windows = int(r.get("num_windows") or 0)

        # Empty texts / no windows => article metrics are NaN/Unknown
        ai_prob = r.get("ai_text_probability")
        is_missing = (n_windows == 0) or (ai_prob is None) or pd.isna(ai_prob)

        if is_missing:
            out: Dict[str, Any] = {
                "prediction_id": pid,
                "lang": lg,
                "fraction_ai": None,
                "prediction_short": None,
                "prediction_long": None,
                "n_windows": 0,
                "n_ai_segments": 0,
                "n_human_segments": 0,
                "n_tokens": int(token_counts.get(pid, 0) or 0),
            }
        else:
            out = {
                "prediction_id": pid,
                "lang": lg,
                "fraction_ai": float(ai_prob),
                "prediction_short": str(r.get("prediction_short")),
                "prediction_long": str(r.get("prediction")),
                "n_windows": int(r.get("num_windows") or 0),
                "n_ai_segments": int(r.get("num_ai_segments") or 0),
                "n_human_segments": int(r.get("num_human_segments") or 0),
                "n_tokens": int(token_counts.get(pid, 0) or 0),
            }

        if return_text:
            out["text"] = str(r.get("text") or "")

        results.append(out)

    return results
