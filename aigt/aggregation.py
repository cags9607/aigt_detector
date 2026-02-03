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
) -> pd.DataFrame:
    """
    Aggregates per prediction_id:
      - ai_text_probability = token-weighted fraction of windows with score >= threshold
      - fraction_ai, fraction_human
      - num_ai_segments, num_human_segments
      - prediction, prediction_short
    """
    thr = float(threshold)

    def _agg(g: pd.DataFrame) -> pd.Series:
        scores = g["ai_assistance_score"].to_numpy()
        tw = g["token_count"].to_numpy().astype(float)

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
