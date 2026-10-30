from english_split_rephrase.pipeline import validate_parts


def test_validation_accepts_two_parts_with_same_numbers() -> None:
    source = "The value 3.14 is stored because the result is stable."
    parts = ["The value 3.14 is stored.", "The result is stable."]
    assert validate_parts(source, parts) is None


def test_validation_rejects_number_changes() -> None:
    source = "The value 3.14 is stored because the result is stable."
    parts = ["The value 2.71 is stored.", "The result is stable."]
    assert validate_parts(source, parts) == "numeric_mismatch"


def test_validation_requires_exactly_two_nonempty_parts() -> None:
    assert validate_parts("A sentence.", ["A sentence."]) == "format_failure"
    assert validate_parts("A sentence.", ["", "Another sentence."]) == "format_failure"
