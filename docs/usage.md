# Usage reference

## Input

The command line interface reads UTF-8 JSONL. Each non-empty row contains a
string `text` field and may contain an `id` field.

```json
{"id":"example-1","text":"The service stores the file and returns a warning when validation fails."}
```

## Output

The output keeps the original `text`, writes the final sentence leaves to
`sentences`, and copies `id` when it is present.

```json
{"id":"example-1","text":"The service stores the file and returns a warning when validation fails.","sentences":["The service stores the file","A warning is returned when validation fails"]}
```

## Processing steps

1. Existing sentence boundaries are separated with conservative punctuation
   rules.
2. The gate predicts `stop` or `continue`.
3. The rewriter generates two parts separated by `<SEP>`.
4. The pipeline repeats the same process until the gate returns `stop`, the
   depth limit is reached, or a generated split fails a basic output check.

## Main options

| Option | Default | Meaning |
|---|---:|---|
| `--threshold` | `0.21` | probability threshold for `continue` |
| `--max-depth` | `3` | maximum recursive split depth |
| `--min-tokens` | `8` | minimum rewriter token count before another split |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--local-files-only` | off | load only local model files |

The rewriter model uses the input prefix `split: ` and emits exactly two
parts for a valid split.

The same pipeline is available as a Python API and as the installed
`split-and-rephrase` console command.
