"""Command-line interface for the public split-and-rephrase pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import DEFAULT_THRESHOLD, SplitAndRephrasePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL path")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--gate-model", required=True, help="Local path or Hugging Face model ID")
    parser.add_argument("--rewrite-model", required=True, help="Local path or Hugging Face model ID")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-tokens", type=int, default=8)
    parser.add_argument("--bert-max-length", type=int, default=128)
    parser.add_argument("--t5-max-source-length", type=int, default=192)
    parser.add_argument("--t5-max-new-tokens", type=int, default=192)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"line {line_number}: expected a JSON object with string 'text'")
            if not row["text"].strip():
                raise ValueError(f"line {line_number}: 'text' must be non-empty")
            rows.append(row)
    return rows


def main() -> int:
    args = build_parser().parse_args()
    rows = _load_rows(args.input)
    splitter = SplitAndRephrasePipeline.from_pretrained(
        args.gate_model,
        args.rewrite_model,
        threshold=args.threshold,
        max_depth=args.max_depth,
        min_tokens=args.min_tokens,
        bert_max_length=args.bert_max_length,
        t5_max_source_length=args.t5_max_source_length,
        t5_max_new_tokens=args.t5_max_new_tokens,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(rows, start=1):
            record = {"text": row["text"], "sentences": splitter.split_text(row["text"])}
            if "id" in row:
                record["id"] = row["id"]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index % 20 == 0:
                print(f"processed {index}/{len(rows)}", file=sys.stderr)
    print(f"processed {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
