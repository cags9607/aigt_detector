# Aggregation rule:
# fraction_ai = (sum token_count of AI windows) / (sum token_count of all windows)
# + Tail exception:
#   If n_windows > 1 and last window token_count < 75, ignore that last window for aggregation (inherently noisy)

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def prediction_bins_from_fraction(ai_ratio: float) -> Tuple[str, str]:
    if ai_ratio != ai_ratio:
        return ("Unknown", "Unknown")
    p = float(ai_ratio)
    if p <= 0.0:
        return ("Fully human-written", "Human")
    if p >= 1.0:
        return ("Fully AI-generated", "AI")
    if p < 0.10:
        return ("Primarily human; small AI detected", "Human")
    if p < 0.30:
        return ("Primarily human; some AI detected", "Human")
    if p < 0.80:
        return ("Mix of AI and human", "Mixed")
    return ("Primarily AI; some human detected", "AI")


def aggregate_token_weighted(
    df_windows: pd.DataFrame,
    *,
    threshold: float,
    tail_min_tokens: int = 75,
) -> pd.DataFrame:
    """
    Aggregates per prediction_id:

      - ai_text_probability = token-weighted fraction of tokens in windows with score >= threshold
        (same value as fraction_ai; name kept for compatibility)
      - fraction_ai, fraction_human
      - num_ai_segments, num_human_segments   (counts of windows among the windows considered for aggregation)
      - prediction, prediction_short

    Tail exception (aggregation only):
      - If n_windows > 1 and the last window token_count < tail_min_tokens, ignore that last window.
      - "Last" is defined by highest window_index.
    """
    thr = float(threshold)
    tail_min_tokens = int(tail_min_tokens)

    required = {"prediction_id", "ai_assistance_score", "token_count"}
    missing = required - set(df_windows.columns)
    if missing:
        raise ValueError(f"df_windows missing required columns: {sorted(missing)}")

    has_window_index = "window_index" in df_windows.columns

    def _agg(g: pd.DataFrame) -> pd.Series:
        # Ensure stable ordering so "last window" is well-defined
        if has_window_index:
            g0 = g.sort_values("window_index").reset_index(drop=True)
        else:
            # Fallback: preserve incoming order (less ideal, but avoids hard failure)
            g0 = g.reset_index(drop=True)

        # Tail-drop rule (aggregation only)
        if len(g0) > 1:
            last_tc = g0.iloc[-1]["token_count"]
            try:
                last_tc = float(last_tc)
            except Exception:
                last_tc = np.nan

            if np.isfinite(last_tc) and last_tc < tail_min_tokens:
                g0 = g0.iloc[:-1].reset_index(drop=True)

        scores = g0["ai_assistance_score"].to_numpy()
        tw = g0["token_count"].to_numpy().astype(float)

        valid = np.isfinite(scores)
        if valid.sum() == 0:
            return pd.Series(
                {
                    "ai_text_probability": np.nan,
                    "fraction_ai": np.nan,
                    "fraction_human": np.nan,
                    "num_ai_segments": 0,
                    "num_human_segments": 0,
                    "prediction": "Unknown",
                    "prediction_short": "Unknown",
                }
            )

        is_ai = (scores[valid] >= thr).astype(int)
        num_ai = int(is_ai.sum())
        num_human = int(valid.sum() - num_ai)

        tw_v = tw[valid]
        if np.isfinite(tw_v).all() and tw_v.sum() > 0:
            frac_ai = float(tw_v[is_ai == 1].sum() / tw_v.sum())
        else:
            frac_ai = float(is_ai.mean())

        pred_long, pred_short = prediction_bins_from_fraction(frac_ai)

        return pd.Series(
            {
                "ai_text_probability": float(frac_ai),
                "fraction_ai": float(frac_ai),
                "fraction_human": float(1.0 - frac_ai),
                "num_ai_segments": num_ai,
                "num_human_segments": num_human,
                "prediction": pred_long,
                "prediction_short": pred_short,
            }
        )

    out = df_windows.groupby("prediction_id", as_index=False).apply(_agg, include_groups=False)
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index(drop=True)
    return out
