# Model repositories

The [GitHub inference package](https://github.com/soundstarrain/english-split-and-rephrase)
loads these two Hugging Face models.

| Repository | Task | Base model |
|---|---|---|
| [Gate model](https://huggingface.co/soundstarrain/english-split-and-rephrase-gate) | stop/continue classification | `answerdotai/ModernBERT-base` |
| [Rewriter model](https://huggingface.co/soundstarrain/english-split-and-rephrase-rewriter) | one sentence to two sentences | `google-t5/t5-base` |

The code repository accepts either local paths or these Hugging Face model IDs.
