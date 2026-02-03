# aigt

A small inference library for your QLoRA "student" models:

- Loads a quantized base backbone (4-bit via bitsandbytes) + per-language LoRA adapters + a small head.
- Splits text into contiguous Pangram-like windows (adaptive boundary snapping).
- Scores each window, then aggregates to an article-level AI ratio (token-weighted by default).
- Returns two pandas DataFrames: `articles_df`, `windows_df`.

## Install (editable)

```bash
pip install -e .
```

## Quickstart

```python
from aigt import Detector

det = Detector.from_hf(
    repo_id = "Trinotrotolueno/aigt-loras",
    subdir_by_lang = {
        "en": "en/best",
        "es": "es/best",
        "fr": "fr/best",
        "de": "de/best",
        "it": "it/best",
        "ja": "ja/best",
    },
    token = None,  # or use env var HF_TOKEN
)

articles_df, windows_df = det.predict(
    texts = ["Hello world..."],
    doc_ids = ["doc1"],
    lang = "en",
)
```

## Notes

- Importing the package does **not** require CUDA.
- CUDA is required only at runtime when calling `.predict()` (unless you run on CPU, which is not recommended for these backbones).
- The default cache policy is `unload_after_call` (GPU friendly). You can set `cache_policy="persist"` to keep models in memory.

