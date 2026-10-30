from pathlib import Path

from english_split_rephrase.cli import _load_rows


def test_jsonl_input_contract(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text('{"id":"x","text":"A sentence."}\n', encoding="utf-8")
    assert _load_rows(path) == [{"id": "x", "text": "A sentence."}]
