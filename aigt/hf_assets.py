from __future__ import annotations

from typing import Optional
from huggingface_hub import hf_hub_download


def hf_download(
    *,
    repo_id: str,
    filename: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        token=token,
    )
