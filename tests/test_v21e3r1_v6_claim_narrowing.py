from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "ijoc_submission_v21e3r1"
V6 = ROOT / "manuscript_v6_working"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _read_v6(name: str) -> str:
    return (V6 / name).read_text(encoding="utf-8")


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_cache_statement_matches_retry_without_optimizer_decision() -> None:
    main = _read_v6("ijoc_v21e3r1_main_v6.tex")
    amendment = _read("protocol/V21E3R1_V6_THEORY_AND_EVIDENCE_ERRATA.md")
    for text in (main, amendment):
        assert "does not execute an optimizer decision" in _flat(text)


def test_charged_count_is_durable_not_transient_physical_execution() -> None:
    main = _read_v6("ijoc_v21e3r1_main_v6.tex")
    amendment = _read("protocol/V21E3R1_V6_THEORY_AND_EVIDENCE_ERRATA.md")
    for text in (main, amendment):
        assert "durably committed evaluation records" in _flat(text)
        assert "existed only transiently" in _flat(text)


def test_development_statistics_are_explicitly_descriptive() -> None:
    files = (
        "manuscript_v6_working/ijoc_v21e3r1_main_v6.tex",
        "manuscript_v6_working/ijoc_v21e3r1_supplement_v6.tex",
        "manuscript_v6_working/LINEAGE_AND_CLAIM_STATUS_V6.md",
        "protocol/V21E3R1_V6_THEORY_AND_EVIDENCE_ERRATA.md",
    )
    for relative in files:
        text = _read(relative)
        assert "descriptive" in text.lower(), relative
        assert "sign symmetry" in _flat(text).lower(), relative


def test_six_case_trimmed_mean_is_not_presented_as_robustness() -> None:
    main = _read_v6("ijoc_v21e3r1_main_v6.tex")
    supplement = _read_v6("ijoc_v21e3r1_supplement_v6.tex")
    amendment = _read("protocol/V21E3R1_V6_THEORY_AND_EVIDENCE_ERRATA.md")
    for text in (main, supplement, amendment):
        assert "ordinary arithmetic mean" in _flat(text)


def test_native_portfolio_and_no_clipping_claims_are_scoped() -> None:
    main = _read_v6("ijoc_v21e3r1_main_v6.tex")
    amendment = _read("protocol/V21E3R1_V6_THEORY_AND_EVIDENCE_ERRATA.md")
    assert "eventually reaching every native operator" not in main
    assert "frozen 21-direction development configuration" in _flat(main)
    assert "invalid ledger objective" in _flat(amendment)
    assert "metric roundoff" in _flat(amendment)


def test_v6_preserves_historical_matrix_identity_and_hold() -> None:
    status = _read_v6("submission_status_v6.tex")
    lineage = _read_v6("LINEAGE_AND_CLAIM_STATUS_V6.md")
    assert "IJOC\\_HOLD" in status
    assert "matrix was not rerun" in lineage
    assert "historical V4 values and artifact identities are unchanged" in lineage


def test_exact_objective_type_repair_is_not_back_projected_to_v4() -> None:
    supplement = _read_v6("ijoc_v21e3r1_supplement_v6.tex")
    amendment = _read("protocol/V21E3R1_V6_THEORY_AND_EVIDENCE_ERRATA.md")
    lineage = _read_v6("LINEAGE_AND_CLAIM_STATUS_V6.md")
    for text in (supplement, amendment, lineage):
        flattened = _flat(text)
        assert "historical V4 producer" in flattened
        assert "post-V4 V6 repair candidate" in flattened
        assert "not" in flattened

