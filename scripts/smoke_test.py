"""Run one local end-to-end check against the two released model directories."""

from __future__ import annotations

import argparse
import json

from english_split_rephrase.pipeline import SplitAndRephrasePipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-model", required=True)
    parser.add_argument("--rewrite-model", required=True)
    parser.add_argument("--device", default="cpu", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    pipeline = SplitAndRephrasePipeline.from_pretrained(
        args.gate_model,
        args.rewrite_model,
        device=args.device,
        local_files_only=True,
    )
    text = "The application stores the submitted file and records its checksum while returning a warning when validation fails."
    result = {"text": text, "sentences": pipeline.split_text(text)}
    if not result["sentences"]:
        raise RuntimeError("the pipeline returned no sentences")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
