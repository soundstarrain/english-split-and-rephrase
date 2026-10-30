"""Inference pipeline for recursive English split-and-rephrase."""

from __future__ import annotations

import json
from collections import Counter
from inspect import signature
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)

from .text_utils import (
    detokenize,
    mask_protected_spans,
    number_tokens,
    restore_protected_spans,
    split_sentences,
    word_tokens,
)


SEP_TOKEN = "<SEP>"
T5_PREFIX = "split: "
DEFAULT_THRESHOLD = 0.21


def _load_tokenizer(model_ref: str | Path, local_files_only: bool):
    """Load both current and older tokenizer metadata formats."""

    model_ref = str(model_ref)
    try:
        return AutoTokenizer.from_pretrained(
            model_ref,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
    except (AttributeError, ValueError):
        model_path = Path(model_ref)
        config_path = model_path / "tokenizer_config.json"
        tokenizer_path = model_path / "tokenizer.json"
        if not config_path.is_file() or not tokenizer_path.is_file():
            raise
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        kwargs = {
            "tokenizer_file": str(tokenizer_path),
            "pad_token": metadata.get("pad_token"),
            "unk_token": metadata.get("unk_token"),
            "eos_token": metadata.get("eos_token"),
            "bos_token": metadata.get("bos_token"),
            "cls_token": metadata.get("cls_token"),
            "sep_token": metadata.get("sep_token"),
            "mask_token": metadata.get("mask_token"),
            "additional_special_tokens": metadata.get("extra_special_tokens", []),
        }
        return PreTrainedTokenizerFast(
            **{key: value for key, value in kwargs.items() if value is not None}
        )


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return resolved


def _model_inputs(encoded: dict[str, Any], model: Any, device: torch.device) -> dict[str, Any]:
    """Keep only tokenizer fields accepted by the loaded model."""

    accepted = set(signature(model.forward).parameters)
    return {
        key: value.to(device)
        for key, value in encoded.items()
        if key in accepted
    }


def validate_parts(source: str, parts: list[str]) -> str | None:
    """Return a rejection reason, or ``None`` when the output passes checks."""

    if len(parts) != 2 or any(not part.strip() for part in parts):
        return "format_failure"

    combined = " ".join(parts)
    if "<unk>" in combined:
        return "unknown_token"
    if Counter(number_tokens(source)) != Counter(number_tokens(combined)):
        return "numeric_mismatch"

    source_words = set(word_tokens(source))
    target_words = word_tokens(combined)
    added_words = [word for word in target_words if word not in source_words]
    if len(added_words) >= 3 and target_words:
        if len(added_words) / len(target_words) > 0.25:
            return "added_word_ratio"
    return None


class SplitAndRephrasePipeline:
    """Run the gate and rewriter models as a recursive processing pipeline."""

    def __init__(
        self,
        gate_model: Any,
        gate_tokenizer: Any,
        rewrite_model: Any,
        rewrite_tokenizer: Any,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        max_depth: int = 3,
        min_tokens: int = 8,
        bert_max_length: int = 128,
        t5_max_source_length: int = 192,
        t5_max_new_tokens: int = 192,
        device: str = "auto",
    ) -> None:
        self.device = _resolve_device(device)
        self.gate_model = gate_model.to(self.device).eval()
        self.gate_tokenizer = gate_tokenizer
        self.rewrite_model = rewrite_model.to(self.device).eval()
        self.rewrite_tokenizer = rewrite_tokenizer
        self.threshold = threshold
        self.max_depth = max_depth
        self.min_tokens = min_tokens
        self.bert_max_length = bert_max_length
        self.t5_max_source_length = t5_max_source_length
        self.t5_max_new_tokens = t5_max_new_tokens

    @classmethod
    def from_pretrained(
        cls,
        gate_model: str | Path,
        rewrite_model: str | Path,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        max_depth: int = 3,
        min_tokens: int = 8,
        bert_max_length: int = 128,
        t5_max_source_length: int = 192,
        t5_max_new_tokens: int = 192,
        device: str = "auto",
        local_files_only: bool = False,
    ) -> "SplitAndRephrasePipeline":
        gate_tokenizer = _load_tokenizer(gate_model, local_files_only)
        gate = AutoModelForSequenceClassification.from_pretrained(
            str(gate_model),
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        rewrite_tokenizer = _load_tokenizer(rewrite_model, local_files_only)
        rewriter = AutoModelForSeq2SeqLM.from_pretrained(
            str(rewrite_model),
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        if SEP_TOKEN not in rewrite_tokenizer.get_vocab():
            rewrite_tokenizer.add_special_tokens(
                {"additional_special_tokens": [SEP_TOKEN]}
            )
            rewriter.resize_token_embeddings(len(rewrite_tokenizer))
        return cls(
            gate,
            gate_tokenizer,
            rewriter,
            rewrite_tokenizer,
            threshold=threshold,
            max_depth=max_depth,
            min_tokens=min_tokens,
            bert_max_length=bert_max_length,
            t5_max_source_length=t5_max_source_length,
            t5_max_new_tokens=t5_max_new_tokens,
            device=device,
        )

    @torch.inference_mode()
    def continue_probability(self, text: str) -> float:
        encoded = self.gate_tokenizer(
            [text],
            truncation=True,
            max_length=self.bert_max_length,
            return_tensors="pt",
        )
        encoded = _model_inputs(encoded, self.gate_model, self.device)
        logits = self.gate_model(**encoded).logits
        return float(torch.softmax(logits.float(), dim=-1)[0, 1].item())

    @torch.inference_mode()
    def rewrite_once(self, text: str) -> tuple[list[str] | None, str | None]:
        masked_text, mapping = mask_protected_spans(text)
        encoded = self.rewrite_tokenizer(
            [T5_PREFIX + masked_text],
            truncation=True,
            max_length=self.t5_max_source_length,
            return_tensors="pt",
        )
        encoded = _model_inputs(encoded, self.rewrite_model, self.device)
        generated = self.rewrite_model.generate(
            **encoded,
            max_new_tokens=self.t5_max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
        prediction = self.rewrite_tokenizer.decode(
            generated[0], skip_special_tokens=False
        )
        for token in (self.rewrite_tokenizer.pad_token, self.rewrite_tokenizer.eos_token):
            if token:
                prediction = prediction.replace(token, "")
        prediction = prediction.strip()
        parts = [part.strip() for part in prediction.split(SEP_TOKEN)]
        parts = [part for part in parts if part]
        if len(parts) != 2:
            return None, "format_failure"
        if any(part == masked_text.strip() for part in parts):
            return None, "echo_input"

        restored: list[str] = []
        seen: set[str] = set()
        for part in parts:
            restored_part, restored_seen = restore_protected_spans(part, mapping)
            restored.append(detokenize(restored_part))
            seen.update(restored_seen)
        if seen != set(mapping):
            return None, "protected_span_lost"
        failure = validate_parts(text, restored)
        if failure is not None:
            return None, failure
        return restored, None

    def _token_count(self, text: str) -> int:
        encoded = self.rewrite_tokenizer(text, return_tensors=None)
        return len(encoded["input_ids"])

    def decompose(self, text: str) -> list[str]:
        leaves: list[str] = []
        stack: list[tuple[str, int]] = [(text.strip(), 0)]
        while stack:
            node_text, depth = stack.pop()
            if depth >= self.max_depth or self._token_count(node_text) < self.min_tokens:
                leaves.append(node_text)
                continue
            if self.continue_probability(node_text) < self.threshold:
                leaves.append(node_text)
                continue
            parts, _reason = self.rewrite_once(node_text)
            if parts is None:
                leaves.append(node_text)
                continue
            stack.append((parts[1], depth + 1))
            stack.append((parts[0], depth + 1))
        return leaves

    def split_text(self, text: str) -> list[str]:
        """Split an English text into recursively simplified sentence leaves."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        leaves: list[str] = []
        for sentence in split_sentences(text.strip()):
            leaves.extend(self.decompose(sentence))
        return leaves
