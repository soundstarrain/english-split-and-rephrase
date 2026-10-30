from english_split_rephrase.text_utils import (
    detokenize,
    mask_protected_spans,
    restore_protected_spans,
    split_sentences,
)


def test_split_sentences_keeps_decimals_and_common_abbreviations() -> None:
    text = "Version 2.1 is ready. It works, e.g. in batch mode."
    assert split_sentences(text) == [
        "Version 2.1 is ready.",
        "It works, e.g. in batch mode.",
    ]


def test_protected_spans_round_trip() -> None:
    source = "Use `x <= 3` when the value is valid."
    masked, mapping = mask_protected_spans(source)
    restored, seen = restore_protected_spans(masked, mapping)
    assert restored == source
    assert seen == set(mapping)


def test_detokenize_repairs_punctuation_spacing() -> None:
    assert detokenize("A value , 3 . 14 !") == "A value, 3.14!"
