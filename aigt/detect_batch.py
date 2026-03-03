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
    repo_id: str = "DeepSee-io/qwen_adapters_aigt",
    subdir_by_lang: Dict[str, str],
    revision: Optional[str] = None,
    hf_token: Optional[str] = None,
    max_len: int = 500,
    batch_size: int = 16,
    progress: bool = True,
    return_text: bool = False,
    device: str = "cuda",
    cache_policy: str = "unload_after_call",
    model_name_fallback: str = "Qwen/Qwen2.5-3B-Instruct",
    max_length_fallback: int = 512,
    window_ai_threshold: float = 0.5,
    prefer_bf16: bool = True,
) -> List[Dict[str, Any]]:
    """
    Batch, article-level detection.

    Alignment contract:
    - Returns exactly one dict per input text.
    - Output order always matches input order.
    - Missing / dropped docs are emitted with None prediction fields and 0 counts.
    """

    texts_list = list(texts)
    n = len(texts_list)

    if prediction_ids is None:
        prediction_ids_list = [str(i) for i in range(n)]
    else:
        prediction_ids_list = [str(x) for x in prediction_ids]
        if len(prediction_ids_list) != n:
            raise ValueError("prediction_ids must match texts length.")

    if len(set(prediction_ids_list)) != len(prediction_ids_list):
        raise ValueError("prediction_ids must be unique within detect_batch().")

    if isinstance(lang, str):
        langs = [lang] * n
    else:
        langs = list(lang)
        if len(langs) != n:
            raise ValueError("lang must be a string or a list with same length as texts.")

    runtime = RuntimeConfig(
        device = device,
        cache_policy = cache_policy,  # type: ignore
        model_name_fallback = model_name_fallback,
        max_length_fallback = max_length_fallback,
        window_ai_threshold = float(window_ai_threshold),
        prefer_bf16 = bool(prefer_bf16),
    )

    detector = Detector.from_hf(
        repo_id = repo_id,
        subdir_by_lang = subdir_by_lang,
        revision = revision,
        token = hf_token,
        runtime = runtime,
    )

    window_cfg = WindowConfig(token_length = int(max_len))
    batch_cfg = BatchConfig(batch_size = int(batch_size), show_progress = bool(progress))

    articles_df, windows_df = detector.predict(
        texts = texts_list,
        doc_ids = prediction_ids_list,
        lang = langs,
        window = window_cfg,
        batch = batch_cfg,
        window_ai_threshold = float(window_ai_threshold),
    )

    if articles_df is None:
        articles_df = pd.DataFrame()
    if windows_df is None:
        windows_df = pd.DataFrame()

    if not isinstance(articles_df, pd.DataFrame):
        articles_df = pd.DataFrame(articles_df)
    if not isinstance(windows_df, pd.DataFrame):
        windows_df = pd.DataFrame(windows_df)

    if (
        not windows_df.empty
        and "prediction_id" in windows_df.columns
        and "token_count" in windows_df.columns
    ):
        token_counts = (
            windows_df.assign(prediction_id = windows_df["prediction_id"].astype(str))
            .groupby("prediction_id", as_index = True)["token_count"]
            .sum()
            .to_dict()
        )
    else:
        token_counts = {}

    articles_by_pid: Dict[str, pd.Series] = {}
    if not articles_df.empty and "prediction_id" in articles_df.columns:
        for _, r in articles_df.iterrows():
            pid = str(r.get("prediction_id"))
            if pid not in articles_by_pid:
                articles_by_pid[pid] = r

    results: List[Dict[str, Any]] = []

    for i, pid in enumerate(prediction_ids_list):
        requested_lang = str(langs[i] or "en")
        r = articles_by_pid.get(pid)

        if r is None:
            out: Dict[str, Any] = {
                "prediction_id": pid,
                "lang": requested_lang,
                "fraction_ai": None,
                "prediction_short": None,
                "prediction_long": None,
                "n_windows": 0,
                "n_ai_segments": 0,
                "n_human_segments": 0,
                "n_tokens": int(token_counts.get(pid, 0) or 0),
            }
            if return_text:
                out["text"] = texts_list[i]
            results.append(out)
            continue

        lg = str(r.get("lang") or requested_lang)

        try:
            n_windows = int(r.get("num_windows") or 0)
        except Exception:
            n_windows = 0

        ai_prob = r.get("ai_text_probability")
        is_missing = (n_windows == 0) or (ai_prob is None) or pd.isna(ai_prob)

        if is_missing:
            out = {
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
                "n_windows": n_windows,
                "n_ai_segments": int(r.get("num_ai_segments") or 0),
                "n_human_segments": int(r.get("num_human_segments") or 0),
                "n_tokens": int(token_counts.get(pid, 0) or 0),
            }

        if return_text:
            out["text"] = str(r.get("text") or texts_list[i] or "")

        results.append(out)

    return results
