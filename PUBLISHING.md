# Publishing checklist

The local release contains one GitHub repository and two Hugging Face model
repositories.

## GitHub

Create a repository named `english-split-and-rephrase`, then upload the
contents of this directory. Keep the two model directories outside the GitHub
repository; the code loads them by Hugging Face ID or local path.

```bash
git init -b main
git add .
git commit -m "Initial public release"
git remote add origin https://github.com/soundstarrain/english-split-and-rephrase.git
git push -u origin main
```

## Hugging Face

Create two model repositories with these names:

```text
english-split-and-rephrase-gate
english-split-and-rephrase-rewriter
```

The model directories already contain the files expected by Transformers and
`.gitattributes` entries for `*.safetensors`.

With the Hugging Face CLI authenticated, upload each directory separately:

```bash
hf upload soundstarrain/english-split-and-rephrase-gate \
  ../huggingface/english-split-and-rephrase-gate .

hf upload soundstarrain/english-split-and-rephrase-rewriter \
  ../huggingface/english-split-and-rephrase-rewriter .
```

The model card metadata declares the task, base model, dataset reference,
license, and held-out evaluation results.
