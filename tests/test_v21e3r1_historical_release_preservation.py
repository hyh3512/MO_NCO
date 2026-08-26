from __future__ import annotations

from pathlib import Path

from ijoc_submission_v21e3r1.scripts.verify_v21e3r1_historical_releases import (
    verify_historical_releases,
)


def test_v4_v6_historical_releases_remain_exact_and_fail_closed() -> None:
    result = verify_historical_releases(Path(".").resolve())
    assert result["status"] == "PASS_HISTORICAL_V4_V6_IDENTITY_AND_RELATIONSHIP"
    assert result["historical_row_count_each"] == 108
    assert result["archive_member_count_each"] == 701
    assert result["unchanged_member_count"] == 700
    assert result["historical_outputs_modified"] is False
    assert result["implementation_independence"] is False
    assert result["scientific_independence"] is False
    assert result["formal_authorized"] is False
    assert result["submission_status"] == "IJOC_HOLD"

