from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class SegmentsDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        return {
            "seg_id": r["seg_id"],
            "prediction_id": r["prediction_id"],
            "window_index": r["window_index"],
            "input_ids": torch.tensor(r["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(r["attention_mask"], dtype=torch.long),
            "n_tokens_fwd": int(r["n_tokens_fwd"]),
        }


def collate_dynamic_pad(tokenizer, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_len = max(int(x["input_ids"].shape[0]) for x in batch)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), fill_value=tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)

    seg_id, prediction_id, window_index, n_tokens_fwd = [], [], [], []
    for i, x in enumerate(batch):
        ids = x["input_ids"]
        m = x["attention_mask"]
        L = ids.shape[0]
        input_ids[i, :L] = ids
        attention_mask[i, :L] = m

        seg_id.append(x["seg_id"])
        prediction_id.append(x["prediction_id"])
        window_index.append(int(x["window_index"]))
        n_tokens_fwd.append(int(x["n_tokens_fwd"]))

    return {
        "seg_id": seg_id,
        "prediction_id": prediction_id,
        "window_index": window_index,
        "n_tokens_fwd": n_tokens_fwd,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
