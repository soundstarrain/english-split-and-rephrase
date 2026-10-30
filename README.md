# English Split and Rephrase

This repository contains the public inference code for recursively splitting
complex English sentences into shorter, meaning-related sentences.

The pipeline uses two Hugging Face models:

1. [English Split and Rephrase Gate](https://huggingface.co/soundstarrain/english-split-and-rephrase-gate)
   decides whether a sentence should be split again.
2. [English Split and Rephrase Rewriter](https://huggingface.co/soundstarrain/english-split-and-rephrase-rewriter)
   rewrites one sentence into two parts separated by `<SEP>`.

The code repository contains no model weights or training data. The weights
are distributed in the two Hugging Face model
repositories listed in `MODEL_REPOSITORIES.md`.

## Project repositories

| Repository | Purpose |
|---|---|
| [Inference code](https://github.com/soundstarrain/english-split-and-rephrase) | Python pipeline, command-line interface, and output validation |
| [Gate model](https://huggingface.co/soundstarrain/english-split-and-rephrase-gate) | Decides whether a sentence should be split again |
| [Rewriter model](https://huggingface.co/soundstarrain/english-split-and-rephrase-rewriter) | Rewrites one sentence into two related parts |

Detailed input/output formats and processing options are in
[`docs/usage.md`](docs/usage.md). The publication commands are in
[`PUBLISHING.md`](PUBLISHING.md).

## Install

```bash
python -m venv .venv
python -m pip install -e ".[test]"
```

For a CPU-only PyTorch installation, install the appropriate PyTorch wheel
first, then install this package without replacing it.

## Run on JSONL

Each input row must contain a non-empty `text` field. An optional `id` is
copied to the output.

```bash
python -m english_split_rephrase \
  --input examples/input.jsonl \
  --output outputs/predictions.jsonl \
  --gate-model soundstarrain/english-split-and-rephrase-gate \
  --rewrite-model soundstarrain/english-split-and-rephrase-rewriter
```

For local model directories, pass the paths and add `--local-files-only`:

```powershell
python -m english_split_rephrase.cli `
  --input examples/input.jsonl `
  --output outputs/predictions.jsonl `
  --gate-model ..\huggingface\english-split-and-rephrase-gate `
  --rewrite-model ..\huggingface\english-split-and-rephrase-rewriter `
  --device cpu `
  --local-files-only
```

The default gate threshold is `0.21`. When a generated output fails a format,
number, or protected-span check, the pipeline keeps the original sentence.

## Python API

```python
from english_split_rephrase import SplitAndRephrasePipeline

pipeline = SplitAndRephrasePipeline.from_pretrained(
    "soundstarrain/english-split-and-rephrase-gate",
    "soundstarrain/english-split-and-rephrase-rewriter",
)
sentences = pipeline.split_text(
    "The service stores the file and returns a warning when validation fails."
)
print(sentences)
```

## Repository layout

```text
src/english_split_rephrase/   library and CLI
tests/                        unit tests and optional model smoke test
scripts/                      local end-to-end smoke test
examples/                     small synthetic JSONL example
```

## Tests

```bash
python -m pytest
```

The default test suite does not download model weights. To run the optional
local model smoke test:

```powershell
$env:SPLIT_GATE_MODEL = "..\huggingface\english-split-and-rephrase-gate"
$env:SPLIT_REWRITE_MODEL = "..\huggingface\english-split-and-rephrase-rewriter"
$env:SPLIT_DEVICE = "cpu"
python -m pytest tests/test_integration.py -rs
```

## Data and license

The code is released under Apache-2.0. No training data is redistributed here.
The model cards identify the upstream BiSECT task, cite the paper, and record
the base-model attribution.
