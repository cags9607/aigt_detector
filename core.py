from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class TextInferConfig:
    # HuggingFace / LoRA assets
    repo_id: str = "DeepSee-io/qwen_adapters_aigt"
    subdir_by_lang_json: str = '{"en":"en/best","es":"es/best","de":"de/best","fr":"fr/best","it":"it/best","ja":"ja/best"}'
    revision: Optional[str] = None
    hf_token: Optional[str] = None

    # Runtime knobs
    device: str = "cuda"
    cache_policy: str = "unload_after_call"
    max_len: int = 500
    batch_size: int = 16
    window_ai_threshold: float = 0.5
    prefer_bf16: bool = True


class TextClassifier:
    """
    Article-level AI text detection via `aigt.detect_batch`.
    Public API mirrors the image worker style:
      - load_models()
      - are_models_loaded()
      - classify_texts_batch(...)
    """

    def __init__(self, cfg: Optional[TextInferConfig] = None):
        self.cfg = cfg or TextInferConfig()
        self._loaded = False
        self._detect_batch = None
        self._subdir_by_lang = json.loads(self.cfg.subdir_by_lang_json)

    def load_models(self):
        from aigt import detect_batch

        self._detect_batch = detect_batch

        # Warmup
        _ = self._detect_batch(
            ["warmup"],
            lang = "en",
            repo_id = self.cfg.repo_id,
            subdir_by_lang = self._subdir_by_lang,
            revision = self.cfg.revision,
            hf_token = self.cfg.hf_token,
            max_len = self.cfg.max_len,
            batch_size = 1,
            progress = False,
            return_text = False,
            device = self.cfg.device,
            cache_policy = self.cfg.cache_policy,
            window_ai_threshold = self.cfg.window_ai_threshold,
            prefer_bf16 = self.cfg.prefer_bf16,
        )

        self._loaded = True
        logger.info("Text detector loaded successfully")

    def are_models_loaded(self) -> bool:
        return self._loaded and self._detect_batch is not None

    def _coerce_text(self, x: Any) -> str:
        if x is None:
            return ""
        return str(x).strip()

    def classify_texts_batch(
        self,
        texts: Sequence[Any],
        langs: Optional[Sequence[str]] = None,
        prediction_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns one dict per input text, aligned to input order.
        """
        if not self.are_models_loaded():
            raise RuntimeError("Models not loaded. Call load_models() first.")

        clean_texts = [self._coerce_text(t) for t in texts]

        if langs is None:
            langs = ["en"] * len(clean_texts)
        else:
            langs = [str(x or "en") for x in langs]

        try:
            rows = self._detect_batch(
                clean_texts,
                prediction_ids = prediction_ids,
                lang = langs,
                repo_id = self.cfg.repo_id,
                subdir_by_lang = self._subdir_by_lang,
                revision = self.cfg.revision,
                hf_token = self.cfg.hf_token,
                max_len = self.cfg.max_len,
                batch_size = self.cfg.batch_size,
                progress = False,
                return_text = False,
                device = self.cfg.device,
                cache_policy = self.cfg.cache_policy,
                window_ai_threshold = self.cfg.window_ai_threshold,
                prefer_bf16 = self.cfg.prefer_bf16,
            )

            out: List[Dict[str, Any]] = []
            for r in rows:
                frac = r.get("fraction_ai", None)
                ai_p = None if frac is None else float(frac)
                human_p = None if ai_p is None else float(1.0 - ai_p)

                out.append({
                    "status": "success" if frac is not None else "empty_or_failed",
                    "prediction_id": r.get("prediction_id"),
                    "lang": r.get("lang"),
                    "prediction_short": r.get("prediction_short"),
                    "prediction_long": r.get("prediction_long"),
                    "fraction_ai": ai_p,
                    "ai_probability": ai_p,
                    "human_probability": human_p,
                    "n_windows": r.get("n_windows"),
                    "n_ai_segments": r.get("n_ai_segments"),
                    "n_human_segments": r.get("n_human_segments"),
                    "n_tokens": r.get("n_tokens"),
                })

            return out

        except Exception as e:
            msg = str(e)
            logger.exception("Batch text processing failed: %s", msg)

            return [{
                "status": "error",
                "error": msg,
                "prediction_id": (prediction_ids[i] if prediction_ids else str(i)),
                "lang": (langs[i] if langs else "en"),
                "prediction_short": None,
                "prediction_long": None,
                "fraction_ai": None,
                "ai_probability": None,
                "human_probability": None,
                "n_windows": 0,
                "n_ai_segments": 0,
                "n_human_segments": 0,
                "n_tokens": 0,
            } for i in range(len(clean_texts))]

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "status": "loaded" if self.are_models_loaded() else "not_loaded",
            "models_loaded": self.are_models_loaded(),
            "repo_id": self.cfg.repo_id,
            "subdir_by_lang": self._subdir_by_lang,
            "revision": self.cfg.revision,
            "device": self.cfg.device,
            "cache_policy": self.cfg.cache_policy,
            "max_len": self.cfg.max_len,
            "batch_size": self.cfg.batch_size,
            "window_ai_threshold": self.cfg.window_ai_threshold,
            "prefer_bf16": self.cfg.prefer_bf16,
        }