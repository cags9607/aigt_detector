from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import HFConfig, WindowConfig, BatchConfig, RuntimeConfig
from .model_loader import LoadedStudent, TokenizerCache, load_student_from_hf, pick_amp_dtype
from .windowing import chunk_text_adaptive_windows
from .tokenization import prepare_span_for_model
from .batching import SegmentsDataset, collate_dynamic_pad
from .aggregation import aggregate_token_weighted
from .utils import safe_str


def _validate_lengths(token_length: int, max_length: int):
    if token_length + 4 > max_length:
        print(
            f"[WARN] token_length={token_length} is too close to max_length={max_length}. "
            "Forward pass may truncate. Prefer token_length <= max_length-4."
        )


class Detector:
    """
    Main public entrypoint.

    Notes:
    - Importing this class does not require CUDA.
    - CUDA is required at runtime for default config (device="cuda").
    """

    def __init__(
        self,
        *,
        hf: HFConfig,
        runtime: RuntimeConfig = RuntimeConfig(),
    ):
        self.hf = hf
        self.runtime = runtime

        self._tokenizer_cache = TokenizerCache()
        self._lang_model_cache: Dict[str, LoadedStudent] = {}

        self._device = torch.device(runtime.device)
        self._amp_dtype = pick_amp_dtype(prefer_bf16=runtime.prefer_bf16)

    @classmethod
    def from_hf(
        cls,
        *,
        repo_id: str,
        subdir_by_lang: Dict[str, str],
        revision: Optional[str] = None,
        token: Optional[str] = None,
        runtime: Optional[RuntimeConfig] = None,
        device: str = "cuda",
        cache_policy: str = "unload_after_call",
        model_name_fallback: str = "Qwen/Qwen2.5-3B-Instruct",
        max_length_fallback: int = 512,
        window_ai_threshold: float = 0.5,
        prefer_bf16: bool = True,
    ) -> "Detector":
        hf = HFConfig(repo_id=repo_id, subdir_by_lang=subdir_by_lang, revision=revision, token=token)
        if runtime is None:
            runtime = RuntimeConfig(
                device=device,
                cache_policy=cache_policy,  # type: ignore
                model_name_fallback=model_name_fallback,
                max_length_fallback=max_length_fallback,
                window_ai_threshold=window_ai_threshold,
                prefer_bf16=prefer_bf16,
            )
        return cls(hf=hf, runtime=runtime)

    # -------------------------
    # Internal: load language model
    # -------------------------
    def _normalize_lang(self, lang: Optional[str]) -> str:
        lg = (lang or "").strip().lower()
        if not lg:
            lg = "en"
        if lg not in self.hf.subdir_by_lang:
            lg = "en"
        return lg

    def load_language(self, lang: str) -> LoadedStudent:
        lg = self._normalize_lang(lang)
        if lg in self._lang_model_cache:
            return self._lang_model_cache[lg]

        subdir = self.hf.subdir_by_lang.get(lg, self.hf.subdir_by_lang.get("en"))
        if not subdir:
            raise ValueError("subdir_by_lang must contain at least 'en' mapping.")

        st = load_student_from_hf(
            lang=lg,
            repo_id=self.hf.repo_id,
            subdir=subdir,
            revision=self.hf.revision,
            token=self.hf.token,
            model_name_fallback=self.runtime.model_name_fallback,
            max_length_fallback=self.runtime.max_length_fallback,
            device=self._device,
            amp_dtype=self._amp_dtype,
            tokenizer_cache=self._tokenizer_cache,
        )
        self._lang_model_cache[lg] = st
        print(f"[Loaded HF] lang={lg} | model_name={st.model_name} | max_length={st.max_length} | subdir={subdir}")
        return st

    def unload_language(self, lang: str):
        lg = self._normalize_lang(lang)
        self._lang_model_cache.pop(lg, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------
    # Public predict
    # -------------------------
    @torch.no_grad()
    def predict(
        self,
        *,
        texts: List[str],
        doc_ids: Optional[List[str]] = None,
        lang: Union[str, List[str]] = "en",
        window: WindowConfig = WindowConfig(),
        batch: BatchConfig = BatchConfig(),
        window_ai_threshold: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(texts))]
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids must be None or have same length as texts.")

        if isinstance(lang, str):
            lang_per_doc = [lang] * len(texts)
        else:
            lang_per_doc = list(lang)
            if len(lang_per_doc) != len(texts):
                raise ValueError("lang must be a string or a list with same length as texts.")

        if self.runtime.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("runtime.device='cuda' but CUDA is not available.")

        if window_ai_threshold is None:
            window_ai_threshold = float(self.runtime.window_ai_threshold)
        thr = float(window_ai_threshold)

        # keep original lang label in outputs; model fallback is handled via normalize->en
        lang_to_idx: Dict[str, List[int]] = {}
        for i, lg in enumerate(lang_per_doc):
            key = (lg or "").strip().lower()
            if not key:
                key = "en"
            lang_to_idx.setdefault(key, []).append(i)

        lang_order = sorted(lang_to_idx.keys(), key=lambda k: len(lang_to_idx[k]), reverse=True)

        all_articles_rows: List[pd.DataFrame] = []
        all_windows_rows: List[pd.DataFrame] = []
        global_seg_id = 0

        for lg in lang_order:
            idxs = lang_to_idx[lg]
            st = self.load_language(lg)  # unknown -> EN inside normalize

            model = st.model
            tokenizer = st.tokenizer
            max_length = int(st.max_length)

            _validate_lengths(int(window.token_length), int(max_length))

            windows_rows: List[Dict[str, Any]] = []
            articles_rows: List[Dict[str, Any]] = []

            it = idxs
            if batch.show_progress:
                it = tqdm(it, desc=f"[{lg}] Chunking", total=len(idxs))

            for i in it:
                prediction_id = str(doc_ids[i])
                text = safe_str(texts[i])

                full_ids = None
                if window.reuse_full_encoding:
                    full_ids = tokenizer(
                        text,
                        add_special_tokens=False,
                        truncation=False,
                        padding=False,
                        return_attention_mask=False,
                    )["input_ids"]

                wins = chunk_text_adaptive_windows(
                    tokenizer,
                    text,
                    token_length=int(window.token_length),
                    stride=window.stride,
                    keep_segment_text=bool(window.keep_segment_text),
                    lookback_tokens=int(window.boundary_lookback_tokens),
                    min_tokens=int(window.boundary_min_tokens),
                    tail_chars=int(window.boundary_tail_chars),
                )

                articles_rows.append(
                    {
                        "lang": (lg or "").strip().lower() if lg is not None else "en",
                        "prediction_id": prediction_id,
                        "token_length_cap": int(window.token_length),
                        "window_ai_threshold": float(thr),
                        "text": text,
                        "num_windows": int(len(wins)),
                    }
                )

                for w_idx, w in enumerate(wins):
                    if window.reuse_full_encoding and full_ids is not None and ("token_start" in w) and ("token_end" in w):
                        ts = int(w["token_start"])
                        te = int(w["token_end"])
                        span_ids = full_ids[ts:te]
                        enc_input_ids, enc_attn = prepare_span_for_model(tokenizer, span_ids)
                        n_tokens_fwd = int(len(enc_input_ids))
                        window_text = w.get("window_text", None) if window.keep_segment_text else None
                    else:
                        w_text = w["window_text"] if window.keep_segment_text else text[w["start_index"] : w["end_index"]]
                        enc = tokenizer(
                            w_text,
                            add_special_tokens=True,
                            truncation=True,
                            max_length=max_length,
                            padding=False,
                            return_attention_mask=True,
                        )
                        enc_input_ids = enc["input_ids"]
                        enc_attn = enc["attention_mask"]
                        n_tokens_fwd = int(len(enc_input_ids))
                        window_text = w_text if window.keep_segment_text else None

                    token_count = int(w["token_length"])

                    windows_rows.append(
                        {
                            "seg_id": str(global_seg_id),
                            "lang": (lg or "").strip().lower() if lg is not None else "en",
                            "prediction_id": prediction_id,
                            "window_index": int(w_idx),
                            "input_ids": enc_input_ids,
                            "attention_mask": enc_attn,
                            "n_tokens_fwd": int(n_tokens_fwd),
                            "start_index": int(w["start_index"]),
                            "end_index": int(w["end_index"]),
                            "token_length": int(w["token_length"]),
                            "token_count": int(token_count),
                            "window_text": window_text,
                        }
                    )
                    global_seg_id += 1

            df_articles_lang = pd.DataFrame(articles_rows)

            if len(windows_rows) == 0:
                df_articles_lang["ai_text_probability"] = np.nan
                df_articles_lang["fraction_ai"] = np.nan
                df_articles_lang["fraction_human"] = np.nan
                df_articles_lang["num_ai_segments"] = 0
                df_articles_lang["num_human_segments"] = 0
                df_articles_lang["prediction"] = "Unknown"
                df_articles_lang["prediction_short"] = "Unknown"

                df_windows_lang = pd.DataFrame(
                    columns=[
                        "lang",
                        "prediction_id",
                        "window_index",
                        "start_index",
                        "end_index",
                        "token_length",
                        "token_count",
                        "ai_assistance_score",
                        "label",
                        "confidence",
                        "window_text",
                    ]
                )

                all_articles_rows.append(df_articles_lang)
                all_windows_rows.append(df_windows_lang)

                if self.runtime.cache_policy == "unload_after_call":
                    self.unload_language(lg)
                continue

            # bucket by forward length to reduce padding
            rows_sorted = sorted(windows_rows, key=lambda r: r["n_tokens_fwd"])
            ds = SegmentsDataset(rows_sorted)

            loader = DataLoader(
                ds,
                batch_size=int(batch.batch_size),
                shuffle=False,
                num_workers=int(batch.num_workers),
                pin_memory=bool(batch.pin_memory),
                persistent_workers=bool(batch.persistent_workers) if int(batch.num_workers) > 0 else False,
                prefetch_factor=int(batch.prefetch_factor) if int(batch.num_workers) > 0 else None,
                collate_fn=lambda b: collate_dynamic_pad(tokenizer, b),
                drop_last=False,
            )

            model.eval()
            seg_pred: Dict[str, float] = {}

            itb = loader
            if batch.show_progress:
                itb = tqdm(itb, desc=f"[{lg}] Scoring windows", total=len(loader))

            with torch.inference_mode():
                for b in itb:
                    input_ids = b["input_ids"].to(self._device, non_blocking=True)
                    attention_mask = b["attention_mask"].to(self._device, non_blocking=True)

                    if self._device.type == "cuda":
                        with torch.amp.autocast("cuda", enabled=True, dtype=self._amp_dtype):
                            logits = model(input_ids=input_ids, attention_mask=attention_mask)
                            probs = torch.sigmoid(logits)
                    else:
                        logits = model(input_ids=input_ids, attention_mask=attention_mask)
                        probs = torch.sigmoid(logits)

                    probs_np = probs.detach().to(torch.float32).cpu().numpy().astype(float)
                    for sid, p in zip(b["seg_id"], probs_np):
                        seg_pred[sid] = float(p)

                    del input_ids, attention_mask, logits, probs

            win_out: List[Dict[str, Any]] = []
            for r in windows_rows:
                p = seg_pred.get(r["seg_id"], np.nan)
                if np.isfinite(p):
                    label = "AI-Generated" if float(p) >= thr else "Human Written"
                    confidence = "High"
                else:
                    label = None
                    confidence = None

                win_out.append(
                    {
                        "lang": r["lang"],
                        "prediction_id": r["prediction_id"],
                        "window_index": int(r["window_index"]),
                        "start_index": int(r["start_index"]),
                        "end_index": int(r["end_index"]),
                        "token_length": int(r["token_length"]),
                        "token_count": int(r["token_count"]),
                        "ai_assistance_score": float(p) if np.isfinite(p) else np.nan,
                        "label": label,
                        "confidence": confidence,
                        "window_text": r["window_text"] if window.keep_segment_text else None,
                    }
                )

            df_windows_lang = pd.DataFrame(win_out).sort_values(["prediction_id", "window_index"]).reset_index(drop=True)

            df_agg = aggregate_token_weighted(df_windows_lang, threshold=thr)
            df_articles_lang = df_articles_lang.merge(df_agg, on="prediction_id", how="left")

            preferred = [
                "lang",
                "prediction_id",
                "ai_text_probability",
                "fraction_ai",
                "fraction_human",
                "num_ai_segments",
                "num_human_segments",
                "window_ai_threshold",
                "token_length_cap",
                "prediction",
                "prediction_short",
                "num_windows",
                "text",
            ]
            df_articles_lang = df_articles_lang[[c for c in preferred if c in df_articles_lang.columns] + [c for c in df_articles_lang.columns if c not in preferred]]

            all_articles_rows.append(df_articles_lang)
            all_windows_rows.append(df_windows_lang)

            if self.runtime.cache_policy == "unload_after_call":
                self.unload_language(lg)

        df_articles = pd.concat(all_articles_rows, ignore_index=True).sort_values(["lang", "prediction_id"]).reset_index(drop=True)
        df_windows = pd.concat(all_windows_rows, ignore_index=True).sort_values(["lang", "prediction_id", "window_index"]).reset_index(drop=True)
        return df_articles, df_windows
