from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel

from .hf_assets import hf_download
from .utils import env_hf_token


@dataclass
class LoadedStudent:
    lang: str
    ckpt_dir: str
    train_cfg: Dict[str, Any]
    model_name: str
    max_length: int
    tokenizer: Any
    model: nn.Module


class TokenizerCache:
    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def get(self, model_name: str):
        if model_name not in self._cache:
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            self._cache[model_name] = tok
        return self._cache[model_name]


def pick_amp_dtype(*, prefer_bf16: bool = True) -> torch.dtype:
    if torch.cuda.is_available() and prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_student_from_hf(
    *,
    lang: str,
    repo_id: str,
    subdir: str,
    revision: Optional[str],
    token: Optional[str],
    model_name_fallback: str,
    max_length_fallback: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    tokenizer_cache: TokenizerCache,
) -> LoadedStudent:
    """
    Loads one language student (base backbone in 4-bit + LoRA adapter + head.pt).
    """
    token = env_hf_token(token)
    subdir = subdir.strip("/")

    # optional train_config.json
    train_cfg: Dict[str, Any] = {}
    try:
        train_cfg_file = hf_download(
            repo_id=repo_id, filename=f"{subdir}/train_config.json", revision=revision, token=token
        )
        with open(train_cfg_file, "r", encoding="utf-8") as f:
            train_cfg = json.load(f)
    except Exception:
        train_cfg = {}

    model_name = train_cfg.get("model_name", model_name_fallback)
    max_length = int(train_cfg.get("max_length", max_length_fallback))

    tokenizer = tokenizer_cache.get(model_name)

    # download LoRA adapter files (PEFT needs local dir)
    adapter_model_file = hf_download(
        repo_id=repo_id,
        filename=f"{subdir}/lora_adapter/adapter_model.safetensors",
        revision=revision,
        token=token,
    )
    _ = hf_download(
        repo_id=repo_id,
        filename=f"{subdir}/lora_adapter/adapter_config.json",
        revision=revision,
        token=token,
    )
    adapter_dir = os.path.dirname(adapter_model_file)

    # 4-bit base model
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=train_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(train_cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=amp_dtype,
    )

    base = AutoModel.from_pretrained(
        model_name,
        quantization_config=qconfig,
        device_map="auto",
        trust_remote_code=True,
    )

    # attach LoRA
    encoder = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    encoder.eval()

    # decoder-only safety
    if hasattr(encoder, "config"):
        try:
            encoder.config.use_cache = False
        except Exception:
            pass

    hidden_size = encoder.config.hidden_size

    # build head
    head = nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, 1),
    ).to(device)

    head_file = hf_download(
        repo_id=repo_id, filename=f"{subdir}/head.pt", revision=revision, token=token
    )
    head.load_state_dict(torch.load(head_file, map_location="cpu"))
    head.eval()

    class Student(nn.Module):
        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = out.last_hidden_state  # [B,T,H]
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            pooled = (last_hidden * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
            logits = self.head(pooled).squeeze(-1)  # [B]
            return logits

    model = Student(encoder, head).to(device)
    model.eval()

    return LoadedStudent(
        lang=lang,
        ckpt_dir=f"hf://{repo_id}/{subdir}",
        train_cfg=train_cfg,
        model_name=model_name,
        max_length=max_length,
        tokenizer=tokenizer,
        model=model,
    )
