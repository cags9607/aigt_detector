from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from huggingface_hub import HfApi, snapshot_download
from peft import PeftModel
from peft.utils import get_peft_model_state_dict
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

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


def pick_amp_dtype(*, prefer_bf16: bool = True) -> torch.dtype:
    if (
        torch.cuda.is_available()
        and prefer_bf16
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float16


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return result


def _setting(
    train_cfg: Dict[str, Any],
    metadata: Dict[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    profile = train_cfg.get("model_profile") or {}

    for source in (metadata, train_cfg, profile):
        value = source.get(name)
        if value is not None:
            return value

    return default


def _download_artifact(
    *,
    repo_id: str,
    subdir: str,
    revision: Optional[str],
    token: Optional[str],
) -> Tuple[Path, str]:
    """
    Resolve one immutable repository revision and download only
    the selected checkpoint's inference files.
    """
    api = HfApi(token=token)
    info = api.model_info(repo_id, revision=revision)
    commit = info.sha

    if not commit:
        raise RuntimeError(f"Could not resolve a commit for {repo_id}")

    files = set(
        api.list_repo_files(
            repo_id=repo_id,
            repo_type="model",
            revision=commit,
        )
    )

    prefix = f"{subdir}/" if subdir else ""

    required = {
        prefix + "head.pt",
        prefix + "lora_adapter/adapter_config.json",
        prefix + "lora_adapter/adapter_model.safetensors",
    }

    missing = required - files
    if missing:
        raise FileNotFoundError(
            "Checkpoint is missing required files: "
            + ", ".join(sorted(missing))
        )

    tokenizer_names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "chat_template.jinja",
    }

    selected = set(required)

    for filename in (
        "train_config.json",
        "model_metadata.json",
        "config.json",
    ):
        relative = prefix + filename
        if relative in files:
            selected.add(relative)

    for relative in files:
        if relative.startswith(prefix + "tokenizer/"):
            selected.add(relative)
        elif relative in {
            prefix + name for name in tokenizer_names
        }:
            selected.add(relative)

    snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=commit,
        token=token,
        allow_patterns=sorted(selected),
    )

    root = Path(snapshot)
    if subdir:
        root = root.joinpath(*subdir.split("/"))

    return root, commit


class TokenizerCache:
    def __init__(self):
        self._cache: Dict[Tuple[Any, ...], Any] = {}

    def get(
        self,
        model_name: str,
        *,
        artifact_dir: Optional[Path] = None,
        artifact_key: Optional[Tuple[str, ...]] = None,
        token: Optional[str] = None,
        padding_side: Optional[str] = None,
        truncation_side: str = "right",
        trust_remote_code: bool = False,
        base_revision: Optional[str] = None,
    ):
        key = (
            artifact_key or (model_name, base_revision),
            padding_side,
            truncation_side,
            trust_remote_code,
        )

        if key in self._cache:
            return self._cache[key]

        source = None
        vocab_files = (
            "tokenizer.json",
            "tokenizer.model",
            "spiece.model",
            "sentencepiece.bpe.model",
            "vocab.json",
            "vocab.txt",
        )

        if artifact_dir is not None:
            for candidate in (
                artifact_dir / "tokenizer",
                artifact_dir,
            ):
                if any(
                    (candidate / name).is_file()
                    for name in vocab_files
                ):
                    source = candidate
                    break

        if source is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                str(source),
                use_fast=True,
                local_files_only=True,
                trust_remote_code=trust_remote_code,
            )
        else:
            warnings.warn(
                "No saved tokenizer found in this checkpoint; "
                f"using the tokenizer from {model_name}.",
                RuntimeWarning,
                stacklevel=2,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                revision=base_revision,
                token=token,
                use_fast=True,
                trust_remote_code=trust_remote_code,
            )

        if padding_side is not None:
            if padding_side not in ("left", "right"):
                raise ValueError(
                    f"Unsupported padding_side: {padding_side!r}"
                )
            tokenizer.padding_side = padding_side

        if truncation_side not in ("left", "right"):
            raise ValueError(
                f"Unsupported truncation_side: {truncation_side!r}"
            )
        tokenizer.truncation_side = truncation_side

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    "Tokenizer has neither a padding token "
                    "nor an EOS token."
                )
            tokenizer.pad_token = tokenizer.eos_token

        self._cache[key] = tokenizer
        return tokenizer


def _adapter_shapes(path: Path) -> Dict[str, Tuple[int, ...]]:
    # Inspect tensor metadata without loading the adapter tensors.
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {
            key: tuple(handle.get_slice(key).get_shape())
            for key in handle.keys()
        }


def _resolve_module(root: nn.Module, path: str) -> nn.Module:
    current = root

    for part in path.split("."):
        if part:
            current = getattr(current, part)

    if not isinstance(current, nn.Module):
        raise TypeError(f"{path!r} does not identify a torch module")

    return current


def _select_backbone(
    loaded_base: nn.Module,
    configured_path: Optional[str],
    adapter_shapes: Dict[str, Tuple[int, ...]],
    hidden_size: int,
) -> Tuple[nn.Module, str]:
    """
    Training may have used a CausalLM wrapper while AutoModel
    already returns its text backbone. Therefore validate paths
    against the saved LoRA modules instead of trusting the
    recorded path alone.
    """
    prefix = "base_model.model."
    adapter_modules = set()

    for key in adapter_shapes:
        if ".lora_A." not in key and ".lora_B." not in key:
            continue

        if not key.startswith(prefix):
            raise RuntimeError(
                f"Unsupported adapter key format: {key}"
            )

        module_name = key[len(prefix):].split(".lora_", 1)[0]
        adapter_modules.add(module_name)

    if not adapter_modules:
        raise RuntimeError(
            "The checkpoint does not contain standard LoRA "
            "A/B weights."
        )

    candidates = []

    if configured_path is not None:
        recorded = str(configured_path).strip()
        if recorded in ("full_text_model", "self", ""):
            recorded = ""
        candidates.append(recorded)

    candidates.extend(
        ["language_model", "model.language_model", "", "model"]
    )

    seen = set()
    checked = []

    for path in candidates:
        try:
            candidate = _resolve_module(loaded_base, path)
        except (AttributeError, TypeError):
            continue

        if id(candidate) in seen:
            continue
        seen.add(id(candidate))

        config = getattr(candidate, "config", None)
        candidate_hidden = getattr(config, "hidden_size", None)
        module_names = set(dict(candidate.named_modules()))
        missing = adapter_modules - module_names

        checked.append(
            f"{path or '<root>'}: "
            f"hidden_size={candidate_hidden}, "
            f"unmatched_adapter_modules={len(missing)}"
        )

        if candidate_hidden != hidden_size or missing:
            continue

        candidate.config.use_cache = False

        for name in (
            "is_loaded_in_4bit",
            "is_loaded_in_8bit",
            "quantization_method",
        ):
            if hasattr(loaded_base, name):
                setattr(candidate, name, getattr(loaded_base, name))

        return candidate, path or "<root>"

    raise RuntimeError(
        "No loaded backbone matches the saved adapter and head. "
        "Check the base model and Transformers version.\n"
        + "\n".join(checked)
    )


def _load_head_state(path: Path) -> Dict[str, torch.Tensor]:
    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(state, dict) and isinstance(
        state.get("state_dict"), dict
    ):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise TypeError("head.pt must contain a state dictionary")

    normalized = {}

    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Head entry {key!r} is not a tensor")

        name = str(key)

        for prefix in (
            "module.head.",
            "model.head.",
            "head.",
            "module.",
        ):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        if name in normalized:
            raise RuntimeError(f"Duplicate head parameter: {name}")

        normalized[name] = tensor

    return normalized


def _build_head(
    state: Dict[str, torch.Tensor],
    dropout: float,
    device: torch.device,
) -> Tuple[nn.Module, int]:
    legacy = {"0.weight", "0.bias", "1.weight", "1.bias"}
    current = {"0.weight", "0.bias", "2.weight", "2.bias"}

    if set(state) == legacy:
        linear_index = "1"
        with_dropout = False
    elif set(state) == current:
        linear_index = "2"
        with_dropout = True
    else:
        raise RuntimeError(
            "Unsupported head architecture. Expected "
            "LayerNorm -> Linear or "
            "LayerNorm -> Dropout -> Linear. "
            f"Found: {sorted(state)}"
        )

    if state["0.weight"].ndim != 1:
        raise RuntimeError("Invalid LayerNorm weight shape")

    hidden_size = int(state["0.weight"].shape[0])

    if tuple(state[f"{linear_index}.weight"].shape) != (
        1,
        hidden_size,
    ):
        raise RuntimeError(
            "Expected a single-output regression head with "
            f"input size {hidden_size}"
        )

    layers = [nn.LayerNorm(hidden_size)]
    if with_dropout:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_size, 1))

    head = nn.Sequential(*layers)
    head.load_state_dict(state, strict=True)
    head.to(device=device, dtype=torch.float32)
    head.eval()

    return head, hidden_size


def _validate_adapter(
    encoder: PeftModel,
    saved_shapes: Dict[str, Tuple[int, ...]],
) -> None:
    """
    Reject partial adapter loading instead of returning scores
    with missing or mismatched trained parameters.
    """
    loaded_state = get_peft_model_state_dict(
        encoder,
        adapter_name="default",
        save_embedding_layers=False,
    )

    expected_keys = set(loaded_state)
    saved_keys = set(saved_shapes)

    missing = sorted(expected_keys - saved_keys)
    unexpected = sorted(saved_keys - expected_keys)
    mismatched = sorted(
        key
        for key in expected_keys & saved_keys
        if tuple(loaded_state[key].shape) != saved_shapes[key]
    )

    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Adapter does not exactly match the selected backbone. "
            f"Missing keys: {missing[:5]}; "
            f"unexpected keys: {unexpected[:5]}; "
            f"shape mismatches: {mismatched[:5]}. "
            "Check the base model and the training/runtime "
            "Transformers and PEFT versions."
        )


class StudentRegressor(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        max_length: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.max_length = max_length

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.shape[1] > self.max_length:
            raise ValueError(
                f"Input has {input_ids.shape[1]} tokens, but the "
                f"artifact max_length is {self.max_length}. "
                "Reduce WindowConfig.token_length."
            )

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]

        if hidden.ndim != 3:
            raise RuntimeError(
                "Expected token hidden states with shape [B, T, H]"
            )

        # Match the benchmark's masked mean pooling.
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1.0)
        )

        # Keep the small regression head in FP32 even when the
        # detector wraps this call in CUDA autocast.
        with torch.autocast(
            device_type=pooled.device.type,
            enabled=False,
        ):
            logits = self.head(pooled.float()).squeeze(-1)

        # Detector.predict() applies sigmoid exactly once.
        return logits


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
    token = env_hf_token(token)
    device = torch.device(device)
    subdir = subdir.strip("/")

    if any(part in (".", "..") for part in subdir.split("/")):
        raise ValueError("subdir must be a repository-relative path")

    if device.type not in ("cuda", "cpu"):
        raise ValueError("This loader supports CUDA and CPU devices")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.index is None:
            device = torch.device(
                "cuda", torch.cuda.current_device()
            )

    root, commit = _download_artifact(
        repo_id=repo_id,
        subdir=subdir,
        revision=revision,
        token=token,
    )

    train_cfg = _read_json(root / "train_config.json")
    metadata = _read_json(root / "model_metadata.json")
    adapter_dir = root / "lora_adapter"
    adapter_cfg = _read_json(adapter_dir / "adapter_config.json")

    if adapter_cfg.get("peft_type") != "LORA":
        raise ValueError("This loader supports LoRA checkpoints")

    model_name = (
        metadata.get("base_model")
        or metadata.get("model_name")
        or train_cfg.get("model_name")
        or train_cfg.get("base_model")
        or adapter_cfg.get("base_model_name_or_path")
        or model_name_fallback
    )

    if not model_name:
        raise ValueError("No base model name was provided")

    max_length = int(
        _setting(
            train_cfg,
            metadata,
            "max_length",
            max_length_fallback,
        )
    )
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    pooling = str(
        _setting(train_cfg, metadata, "pooling", "mean")
    ).lower()

    if pooling not in (
        "mean",
        "masked_mean",
        "attention_mask_mean",
    ):
        raise ValueError(
            f"Unsupported pooling mode: {pooling!r}. "
            "This loader implements masked mean pooling."
        )

    trust_remote_code = bool(
        _setting(
            train_cfg,
            metadata,
            "trust_remote_code",
            False,
        )
    )

    base_revision = _setting(
        train_cfg,
        metadata,
        "base_model_revision",
        adapter_cfg.get("revision"),
    )

    tokenizer = tokenizer_cache.get(
        model_name,
        artifact_dir=root,
        artifact_key=(repo_id, subdir, commit),
        token=token,
        padding_side=_setting(
            train_cfg, metadata, "padding_side"
        ),
        truncation_side=str(
            _setting(
                train_cfg,
                metadata,
                "truncation_side",
                "right",
            )
        ),
        trust_remote_code=trust_remote_code,
        base_revision=base_revision,
    )

    head_state = _load_head_state(root / "head.pt")
    head, hidden_size = _build_head(
        head_state,
        dropout=float(
            _setting(
                train_cfg,
                metadata,
                "head_dropout",
                0.0,
            )
        ),
        device=device,
    )

    recorded_hidden = _setting(
        train_cfg, metadata, "hidden_size"
    )
    if (
        recorded_hidden is not None
        and int(recorded_hidden) != hidden_size
    ):
        raise RuntimeError(
            f"Artifact config hidden_size={recorded_hidden}, "
            f"but head.pt expects {hidden_size}"
        )

    adapter_shapes = _adapter_shapes(
        adapter_dir / "adapter_model.safetensors"
    )

    load_kwargs = {
        "token": token,
        "revision": base_revision,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        # Explicit single-device placement matches Detector's
        # input placement and avoids moving a quantized model.
        "device_map": {"": str(device)},
        "torch_dtype": (
            amp_dtype if device.type == "cuda"
            else torch.float32
        ),
    }

    if device.type == "cuda":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=train_cfg.get(
                "bnb_4bit_quant_type", "nf4"
            ),
            bnb_4bit_use_double_quant=bool(
                train_cfg.get(
                    "bnb_4bit_use_double_quant", True
                )
            ),
            bnb_4bit_compute_dtype=amp_dtype,
        )

    loaded_base = AutoModel.from_pretrained(
        model_name,
        **load_kwargs,
    )

    backbone, backbone_path = _select_backbone(
        loaded_base,
        configured_path=_setting(
            train_cfg,
            metadata,
            "text_backbone_path",
        ),
        adapter_shapes=adapter_shapes,
        hidden_size=hidden_size,
    )

    encoder = PeftModel.from_pretrained(
        backbone,
        str(adapter_dir),
        is_trainable=False,
    )
    _validate_adapter(encoder, adapter_shapes)
    encoder.eval()

    model = StudentRegressor(
        encoder=encoder,
        head=head,
        max_length=max_length,
    )
    model.eval()

    # Do not call model.to(device): the quantized backbone
    # was already placed by from_pretrained().
    del loaded_base

    print(
        f"[Backbone] {model_name} | "
        f"component={backbone_path} | "
        f"hidden_size={hidden_size} | "
        f"pooling={pooling}"
    )

    return LoadedStudent(
        lang=lang,
        ckpt_dir=str(root),
        train_cfg=train_cfg,
        model_name=model_name,
        max_length=max_length,
        tokenizer=tokenizer,
        model=model,
    )
