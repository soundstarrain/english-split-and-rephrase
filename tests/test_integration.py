import os

import pytest

from english_split_rephrase.pipeline import SplitAndRephrasePipeline


@pytest.mark.skipif(
    not os.environ.get("SPLIT_GATE_MODEL") or not os.environ.get("SPLIT_REWRITE_MODEL"),
    reason="set SPLIT_GATE_MODEL and SPLIT_REWRITE_MODEL for model smoke testing",
)
def test_local_models_can_run_one_sentence() -> None:
    pipeline = SplitAndRephrasePipeline.from_pretrained(
        os.environ["SPLIT_GATE_MODEL"],
        os.environ["SPLIT_REWRITE_MODEL"],
        device=os.environ.get("SPLIT_DEVICE", "cpu"),
        local_files_only=True,
    )
    result = pipeline.split_text(
        "The application stores the submitted file and records its checksum while returning a warning when validation fails."
    )
    assert result
    assert all(isinstance(item, str) and item.strip() for item in result)
