#!/usr/bin/env python3

"""
train_w_windows.py

EditLens-style QLoRA distillation on *window-level* CSVs.

Designed to live inside this repo (AIGT) as a training utility script.
- No Google Drive mounting
- No language filtering (assume you pass a language-specific dataset per run)
- Saves artifacts in: <output_dir>/best/
    - lora_adapter/   (PEFT adapter)
    - head.pt         (classification head)
    - train_config.json
    - history.json
    - train_metrics.json
    - val_metrics.json (if val_ratio > 0)
- Also produces a zip: <output_dir>/best.zip with:
    best/ (folder) + history.json at the zip root (convenience)

Usage example:

python scripts/train_w_windows.py \
  --data-csv data/windows_fr.csv \
  --text-col window_text \
  --target-col ai_assistance_score \
  --group-col target_url \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --output-dir runs/qlora_fr \
  --val-ratio 0.1

Notes:
- Requires CUDA (QLoRA + bitsandbytes 4-bit).
- This is training code; keep it out of lightweight inference installs if you want.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
)

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    # Data
    data_csv: str
    text_col: str = "window_text"
    target_col: str = "ai_assistance_score"
    group_col: str = "target_url"

    # Model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"

    # Tokenization
    max_length: int = 512

    # Split
    val_ratio: float = 0.0  # 0.0 => train-only
    seed: int = 42

    # Optimization (EditLens-style defaults)
    num_epochs: int = 1
    batch_size: int = 24
    lr: float = 3e-5
    weight_decay: float = 0.01
    grad_accum_steps: int = 1
    grad_clip_norm: float = 1.0
    num_workers: int = 2

    # QLoRA bitsandbytes
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Eval
    eval_batch_size: int = 8
    threshold_for_classification: float = 0.5

    # AMP
    prefer_bf16: bool = True

    # Output
    output_dir: str = "runs/train_w_windows"


def set_repro(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_amp_dtype(prefer_bf16: bool = True) -> torch.dtype:
    if prefer_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_windows_csv(path: str, *, text_col: str, target_col: str, group_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [text_col, target_col, group_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[text_col, target_col, group_col]).reset_index(drop=True)

    df[text_col] = df[text_col].astype(str)
    df[group_col] = df[group_col].astype(str)

    # Remove empty texts
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)

    # Clamp teacher probabilities
    df[target_col] = df[target_col].clip(0.0, 1.0)

    return df


def group_split(
    df: pd.DataFrame,
    *,
    group_col: str,
    val_ratio: float,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if val_ratio <= 0.0:
        train_df = df.reset_index(drop=True)
        val_df = df.iloc[0:0].copy()
        return train_df, val_df

    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    groups = df[group_col].values
    train_idx, val_idx = next(gss.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    overlap = set(train_df[group_col].unique()).intersection(set(val_df[group_col].unique()))
    if len(overlap) > 0:
        raise RuntimeError("Leakage detected: group_col values appear in both splits.")

    return train_df, val_df


def infer_all_linear_targets(hf_model) -> List[str]:
    """
    Infer typical linear module suffixes to target with LoRA.
    Works for Llama/Mistral/Qwen-like transformer blocks.
    """
    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "fc1", "fc2",
    ]
    module_names = [n for n, _ in hf_model.named_modules()]
    present = []
    for c in candidates:
        if any(n.endswith(c) for n in module_names):
            present.append(c)
    if not present:
        raise RuntimeError(
            "Could not infer LoRA target modules for this model. "
            "If this is a non-Llama/Mistral/Qwen family model, you may need to hardcode target_modules."
        )
    return sorted(set(present))


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, *, tokenizer, text_col: str, target_col: str, max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.target_col = target_col
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        y = float(row[self.target_col])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(y, dtype=torch.float32)
        return item


@torch.inference_mode()
def collect_preds(model: nn.Module, loader: DataLoader, *, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        p = torch.sigmoid(logits)

        ys.append(y.detach().cpu().numpy())
        ps.append(p.detach().cpu().numpy())

    return np.concatenate(ys).astype(float), np.concatenate(ps).astype(float)


def compute_full_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    # regression (prob space)
    mse_v = mean_squared_error(y_true, y_pred)
    mae_v = mean_absolute_error(y_true, y_pred)
    r2_v = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    corr_v = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")

    # classification derived from teacher at threshold
    y_bin = (y_true >= threshold).astype(int)
    p_bin = (y_pred >= threshold).astype(int)

    acc = accuracy_score(y_bin, p_bin)
    prec = precision_score(y_bin, p_bin, zero_division=0)
    rec = recall_score(y_bin, p_bin, zero_division=0)
    f1 = f1_score(y_bin, p_bin, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_bin, p_bin, labels=[0, 1]).ravel()
    fpr = fp / max(fp + tn, 1)
    prevalence = float(np.mean(y_bin))

    # ranking metrics
    try:
        roc_auc = roc_auc_score(y_bin, y_pred)
    except Exception:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_bin, y_pred)
    except Exception:
        pr_auc = float("nan")

    return {
        "mse": float(mse_v),
        "mae": float(mae_v),
        "r2": float(r2_v),
        "corr": float(corr_v),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "fpr": float(fpr),
        "prevalence": float(prevalence),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "y_true_mean": float(np.mean(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
        "n": int(len(y_true)),
        "threshold": float(threshold),
    }


def save_checkpoint(model: nn.Module, out_dir: str, cfg: TrainConfig):
    os.makedirs(out_dir, exist_ok=True)
    # PEFT adapter
    model.encoder.save_pretrained(os.path.join(out_dir, "lora_adapter"))
    # Head
    torch.save(model.head.state_dict(), os.path.join(out_dir, "head.pt"))
    # Train config snapshot
    with open(os.path.join(out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="EditLens-style QLoRA distillation on window-level CSVs.")
    ap.add_argument("--data-csv", required=True, help="Path to windows CSV (language-specific).")
    ap.add_argument("--text-col", default="window_text")
    ap.add_argument("--target-col", default="ai_assistance_score")
    ap.add_argument("--group-col", default="target_url")

    ap.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max-length", type=int, default=512)

    ap.add_argument("--val-ratio", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--num-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)

    ap.add_argument("--output-dir", default="runs/train_w_windows")
    ap.add_argument("--prefer-bf16", action="store_true", help="Prefer BF16 autocast if supported (default).")
    ap.add_argument("--no-prefer-bf16", dest="prefer_bf16", action="store_false", help="Force FP16 autocast.")
    ap.set_defaults(prefer_bf16=True)

    args = ap.parse_args()

    cfg = TrainConfig(
        data_csv=args.data_csv,
        text_col=args.text_col,
        target_col=args.target_col,
        group_col=args.group_col,
        model_name=args.model_name,
        max_length=args.max_length,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_accum_steps=args.grad_accum_steps,
        grad_clip_norm=args.grad_clip_norm,
        num_workers=args.num_workers,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        eval_batch_size=args.eval_batch_size,
        threshold_for_classification=args.threshold,
        output_dir=args.output_dir,
        prefer_bf16=args.prefer_bf16,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    set_repro(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type != "cuda":
        raise RuntimeError("QLoRA requires a GPU runtime (CUDA).")

    amp_dtype = pick_amp_dtype(prefer_bf16=cfg.prefer_bf16)
    print("AMP dtype:", amp_dtype)

    # -------------------------
    # Load data
    # -------------------------
    full_df = load_windows_csv(
        cfg.data_csv,
        text_col=cfg.text_col,
        target_col=cfg.target_col,
        group_col=cfg.group_col,
    )
    print("Rows:", len(full_df))
    print("Unique groups:", full_df[cfg.group_col].nunique())

    train_df, val_df = group_split(full_df, group_col=cfg.group_col, val_ratio=cfg.val_ratio, seed=cfg.seed)

    if cfg.val_ratio <= 0.0:
        print(f"Train size: {len(train_df)} | unique groups: {train_df[cfg.group_col].nunique()}")
        print("Val size:   0")
    else:
        print(f"Train size: {len(train_df)} | unique groups: {train_df[cfg.group_col].nunique()}")
        print(f"Val size:   {len(val_df)} | unique groups: {val_df[cfg.group_col].nunique()}")

    # -------------------------
    # Tokenizer / loaders
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = DataLoader(
        TextDataset(train_df, tokenizer=tokenizer, text_col=cfg.text_col, target_col=cfg.target_col, max_length=cfg.max_length),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = None
    if cfg.val_ratio > 0.0 and len(val_df) > 0:
        val_loader = DataLoader(
            TextDataset(val_df, tokenizer=tokenizer, text_col=cfg.text_col, target_col=cfg.target_col, max_length=cfg.max_length),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    train_eval_loader = DataLoader(
        TextDataset(train_df, tokenizer=tokenizer, text_col=cfg.text_col, target_col=cfg.target_col, max_length=cfg.max_length),
        batch_size=min(cfg.eval_batch_size, cfg.batch_size),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_eval_loader = None
    if val_loader is not None:
        val_eval_loader = DataLoader(
            TextDataset(val_df, tokenizer=tokenizer, text_col=cfg.text_col, target_col=cfg.target_col, max_length=cfg.max_length),
            batch_size=min(cfg.eval_batch_size, cfg.batch_size),
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    # -------------------------
    # QLoRA base model (4-bit NF4 + double quant)
    # -------------------------
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=amp_dtype,
    )

    base = AutoModel.from_pretrained(
        cfg.model_name,
        quantization_config=qconfig,
        device_map="auto",
        trust_remote_code=True,
    )

    # decoder-only safety
    if hasattr(base, "config"):
        try:
            base.config.use_cache = False
        except Exception:
            pass

    base = prepare_model_for_kbit_training(base)

    target_modules = infer_all_linear_targets(base)
    print("LoRA target_modules:", target_modules)

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )

    encoder = get_peft_model(base, lora_cfg)
    encoder.print_trainable_parameters()

    hidden_size = int(getattr(encoder.config, "hidden_size"))

    head = nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, 1),
    ).to(device)

    class Student(nn.Module):
        def __init__(self, encoder: nn.Module, head: nn.Module):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            pooled = (last_hidden * mask).sum(1) / torch.clamp(mask.sum(1), min=1e-9)
            logits = self.head(pooled).squeeze(-1)
            return logits

    model = Student(encoder, head).to(device)

    # Loss: MSE on sigmoid(logits)
    criterion = nn.MSELoss()

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]
    trainable_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in trainable_named_params if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in trainable_named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.lr)

    amp_enabled = True
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # -------------------------
    # Train
    # -------------------------
    best_val_mse = float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}", leave=False)
        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.sigmoid(logits)
                loss = criterion(probs, targets)
                loss = loss / max(cfg.grad_accum_steps, 1)

            scaler.scale(loss).backward()

            if (step % cfg.grad_accum_steps) == 0:
                if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bsz = input_ids.size(0)
            running_loss += float(loss.detach().cpu().item()) * max(cfg.grad_accum_steps, 1) * bsz
            n_seen += bsz
            pbar.set_postfix({"loss": running_loss / max(n_seen, 1)})

        train_loss = running_loss / max(n_seen, 1)

        if val_loader is None:
            row = {"epoch": float(epoch), "train_loss": float(train_loss)}
            history.append(row)
            print(json.dumps(row, indent=2))

            save_checkpoint(model, os.path.join(cfg.output_dir, "best"), cfg)
            print("Saved train-only checkpoint to:", os.path.join(cfg.output_dir, "best"))
        else:
            # simple val mse for "best" selection
            y_va, p_va = collect_preds(model, val_eval_loader, device=device)  # type: ignore
            val_mse = mean_squared_error(y_va, p_va)

            row = {"epoch": float(epoch), "train_loss": float(train_loss), "val_mse": float(val_mse)}
            history.append(row)
            print(json.dumps(row, indent=2))

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                save_checkpoint(model, os.path.join(cfg.output_dir, "best"), cfg)
                print(f"Saved best checkpoint (val_mse={best_val_mse:.6f})")

    with open(os.path.join(cfg.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Training done. Checkpoint in:", os.path.join(cfg.output_dir, "best"))

    # -------------------------
    # Post-fit evaluation (train always, val if exists)
    # -------------------------
    torch.cuda.empty_cache()

    y_tr, p_tr = collect_preds(model, train_eval_loader, device=device)
    train_metrics_full = compute_full_metrics(y_tr, p_tr, threshold=cfg.threshold_for_classification)

    val_metrics_full = None
    if val_eval_loader is not None:
        y_va, p_va = collect_preds(model, val_eval_loader, device=device)
        val_metrics_full = compute_full_metrics(y_va, p_va, threshold=cfg.threshold_for_classification)

    print("\nTRAIN METRICS")
    print(json.dumps(train_metrics_full, indent=2))
    if val_metrics_full is not None:
        print("\nVAL METRICS")
        print(json.dumps(val_metrics_full, indent=2))

    # Save metrics inside best/
    best_dir = os.path.join(cfg.output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    with open(os.path.join(best_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(train_metrics_full, f, indent=2)

    if val_metrics_full is not None:
        with open(os.path.join(best_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(val_metrics_full, f, indent=2)

    with open(os.path.join(best_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Zip layout:
    #   best.zip/
    #     ├── best/ ...
    #     └── history.json
    zip_path = shutil.make_archive(
        base_name=os.path.join(cfg.output_dir, "best"),
        format="zip",
        root_dir=cfg.output_dir
    )

    print("\nZIP created at:", zip_path)


if __name__ == "__main__":
    main()
