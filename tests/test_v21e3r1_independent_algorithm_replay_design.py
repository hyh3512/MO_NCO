from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "independent_reproduction" / "compare_v21e3r1_algorithm_events.py"
SPEC = ROOT / "independent_reproduction" / "V21E3R1_ALGORITHM_REPLAY_SPEC_V1.md"
GOLDEN = ROOT / "independent_reproduction" / "golden"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _events(name: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (GOLDEN / name).read_text(encoding="utf-8").splitlines()
    ]


def _write_events(path: Path, events: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(_canonical_bytes(event) + b"\n" for event in events))


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _rehash(events: list[dict[str, object]]) -> None:
    manifest = events[0]
    context_keys = (
        "producer_id",
        "producer_source_manifest",
        "producer_source_manifest_sha256",
        "algorithm_spec_sha256",
        "case_artifact_sha256",
        "problem_semantic_sha256",
        "problem_family",
        "algorithm_config",
        "algorithm_config_schema",
        "candidate_config_sha256",
        "candidate_id",
        "evidence_partition",
        "charged_evaluation_budget",
        "objective_count",
        "rng_protocol",
        "rng_vector_length",
        "rng_vector_sha256",
    )
    context_sha = _sha256({key: manifest[key] for key in context_keys})
    previous = "0" * 64
    for index, event in enumerate(events):
        event["event_index"] = index
        event["context_sha256"] = context_sha
        event["prev_event_sha256"] = previous
        event.pop("event_sha256", None)
        event["event_sha256"] = _sha256(event)
        previous = str(event["event_sha256"])


def _run_paths(
    reference: Path, candidate: Path, output: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference",
            str(reference),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _run(reference: str, candidate: str, output: Path) -> subprocess.CompletedProcess[str]:
    return _run_paths(GOLDEN / reference, GOLDEN / candidate, output)


def test_matching_golden_streams_pass_without_independence_claim(tmp_path: Path) -> None:
    output = tmp_path / "algorithm replay receipt.json"

    completed = _run("reference_valid.jsonl", "external_valid.jsonl", output)

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS_NEUTRAL_EVENT_STREAM_COMPARISON"
    assert receipt["checks"] == {
        "algorithm_events_equal": True,
        "case_binding_equal": True,
        "config_binding_equal": True,
        "rng_vector_equal": True,
    }
    assert receipt["neutral_comparator"] is True
    assert receipt["reference_algorithm_producer_present"] is False
    assert receipt["external_producer_present"] is False
    assert receipt["producer_authorship_authenticated"] is False
    assert receipt["independent_custody_verified"] is False
    assert receipt["implementation_independence"] is False
    assert receipt["third_party_independence"] is False
    assert receipt["scientific_independence"] is False
    assert receipt["formal_authority"] is False
    receipt_core = dict(receipt)
    embedded_receipt_sha = receipt_core.pop("receipt_payload_sha256")
    assert embedded_receipt_sha == _sha256(receipt_core)
    assert receipt["streams"]["reference"]["sha256"] == hashlib.sha256(
        (GOLDEN / "reference_valid.jsonl").read_bytes()
    ).hexdigest()
    assert receipt["streams"]["candidate"]["sha256"] == hashlib.sha256(
        (GOLDEN / "external_valid.jsonl").read_bytes()
    ).hexdigest()
    assert receipt["streams"]["reference"]["producer_id"] != receipt["streams"][
        "candidate"
    ]["producer_id"]


def test_coherently_rehashed_decision_mismatch_returns_fail_receipt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "decision mismatch receipt.json"

    completed = _run("reference_valid.jsonl", "negative_decision_mismatch.jsonl", output)

    assert completed.returncode == 2, completed.stderr
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL_ALGORITHM_EVENT_STREAM_MISMATCH"
    assert receipt["checks"]["algorithm_events_equal"] is False
    assert receipt["first_mismatch"]["event_index"] == 4
    assert receipt["first_mismatch"]["reference_event_type"] == "decision"
    assert receipt["implementation_independence"] is False


def test_missing_terminal_is_rejected_without_receipt(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete.jsonl"
    _write_events(incomplete, _events("external_valid.jsonl")[:-1])
    output = tmp_path / "must not exist.json"

    completed = _run_paths(GOLDEN / "reference_valid.jsonl", incomplete, output)

    assert completed.returncode == 3
    error = json.loads(completed.stderr)
    assert "stream must end with exactly one terminal event" in error["error"]
    assert not output.exists()


def test_noncanonical_jsonl_is_rejected(tmp_path: Path) -> None:
    candidate = tmp_path / "noncanonical.jsonl"
    raw = (GOLDEN / "external_valid.jsonl").read_bytes()
    candidate.write_bytes(raw.replace(b"{", b"{ ", 1))
    output = tmp_path / "must not exist.json"

    completed = _run_paths(GOLDEN / "reference_valid.jsonl", candidate, output)

    assert completed.returncode == 3
    assert "not canonical JSON" in json.loads(completed.stderr)["error"]
    assert not output.exists()


def test_coherently_rehashed_boolean_budget_is_rejected(tmp_path: Path) -> None:
    events = _events("external_valid.jsonl")
    events[0]["charged_evaluation_budget"] = True
    _rehash(events)
    candidate = tmp_path / "boolean-budget.jsonl"
    _write_events(candidate, events)
    output = tmp_path / "must not exist.json"

    completed = _run_paths(GOLDEN / "reference_valid.jsonl", candidate, output)

    assert completed.returncode == 3
    assert "exact integer" in json.loads(completed.stderr)["error"]
    assert not output.exists()


def test_broken_event_predecessor_hash_is_rejected(tmp_path: Path) -> None:
    events = _events("external_valid.jsonl")
    events[3]["prev_event_sha256"] = "0" * 64
    candidate = tmp_path / "broken-chain.jsonl"
    _write_events(candidate, events)
    output = tmp_path / "must not exist.json"

    completed = _run_paths(GOLDEN / "reference_valid.jsonl", candidate, output)

    assert completed.returncode == 3
    assert "predecessor hash chain" in json.loads(completed.stderr)["error"]
    assert not output.exists()


def test_outer_rehash_cannot_hide_source_inventory_drift(tmp_path: Path) -> None:
    events = _events("external_valid.jsonl")
    source = events[0]["producer_source_manifest"]
    source["files"][0]["sha256"] = "f" * 64
    rebound = _sha256(source)
    events[0]["producer_source_manifest_sha256"] = rebound
    events[-1]["producer_source_manifest_sha256"] = rebound
    _rehash(events)
    candidate = tmp_path / "source-drift.jsonl"
    _write_events(candidate, events)
    output = tmp_path / "must not exist.json"

    completed = _run_paths(GOLDEN / "reference_valid.jsonl", candidate, output)

    assert completed.returncode == 3
    assert "source_root_sha256" in json.loads(completed.stderr)["error"]
    assert not output.exists()


def test_coherently_rehashed_extra_config_key_is_rejected(tmp_path: Path) -> None:
    events = _events("external_valid.jsonl")
    config = events[0]["algorithm_config"]
    config["unfrozen_extra"] = "forbidden"
    config_sha = _sha256(config)
    events[0]["candidate_config_sha256"] = config_sha
    events[-1]["candidate_config_sha256"] = config_sha
    _rehash(events)
    candidate = tmp_path / "extra-config.jsonl"
    _write_events(candidate, events)
    output = tmp_path / "must not exist.json"

    completed = _run_paths(GOLDEN / "reference_valid.jsonl", candidate, output)

    assert completed.returncode == 3
    assert "exact frozen key set" in json.loads(completed.stderr)["error"]
    assert not output.exists()


def test_rng_values_must_match_manifest_binding(tmp_path: Path) -> None:
    events = _events("external_valid.jsonl")
    events[1]["values"][1] = "ffffffffffffffff"
    events[1]["rng_vector_sha256"] = _sha256(events[1]["values"])
    _rehash(events)
    candidate = tmp_path / "rng-drift.jsonl"
    _write_events(candidate, events)
    output = tmp_path / "must not exist.json"

    completed = _run_paths(GOLDEN / "reference_valid.jsonl", candidate, output)

    assert completed.returncode == 3
    assert "rng vector digest" in json.loads(completed.stderr)["error"]
    assert not output.exists()


def test_terminal_must_repeat_case_binding(tmp_path: Path) -> None:
    events = _events("external_valid.jsonl")
    events[-1]["case_artifact_sha256"] = "f" * 64
    _rehash(events)
    candidate = tmp_path / "terminal-case-drift.jsonl"
    _write_events(candidate, events)
    output = tmp_path / "must not exist.json"

    completed = _run_paths(GOLDEN / "reference_valid.jsonl", candidate, output)

    assert completed.returncode == 3
    assert "terminal.case_artifact_sha256" in json.loads(completed.stderr)["error"]
    assert not output.exists()


def test_evaluation_without_decision_is_rejected(tmp_path: Path) -> None:
    events = _events("external_valid.jsonl")
    del events[4]
    _rehash(events)
    candidate = tmp_path / "missing-decision.jsonl"
    _write_events(candidate, events)
    output = tmp_path / "must not exist.json"

    completed = _run_paths(GOLDEN / "reference_valid.jsonl", candidate, output)

    assert completed.returncode == 3
    assert "lacks its decision event" in json.loads(completed.stderr)["error"]
    assert not output.exists()


def test_partial_novelty_witness_is_rejected_as_incomplete(tmp_path: Path) -> None:
    events = _events("external_valid.jsonl")
    events[4]["new_nondominated_cell"] = None
    _rehash(events)
    candidate = tmp_path / "partial-novelty.jsonl"
    _write_events(candidate, events)
    output = tmp_path / "must not exist.json"

    completed = _run_paths(GOLDEN / "reference_valid.jsonl", candidate, output)

    assert completed.returncode == 3
    assert "novelty witness" in json.loads(completed.stderr)["error"]
    assert not output.exists()


def test_receipt_output_is_exclusive_create(tmp_path: Path) -> None:
    output = tmp_path / "existing.json"
    output.write_bytes(b"preexisting-custody-bytes")

    completed = _run("reference_valid.jsonl", "external_valid.jsonl", output)

    assert completed.returncode == 3
    assert output.read_bytes() == b"preexisting-custody-bytes"


def test_comparator_has_no_mo_nco_import() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any(name == "mo_nco" or name.startswith("mo_nco.") for name in imported)
    assert {name.split(".", 1)[0] for name in imported} <= {
        "__future__",
        "argparse",
        "hashlib",
        "json",
        "pathlib",
        "re",
        "sys",
        "typing",
    }


def test_spec_freezes_protocol_and_preserves_external_reproduction_hold() -> None:
    text = SPEC.read_text(encoding="utf-8")
    required = (
        "DESIGN_AND_GOLDEN_CORPUS_ONLY",
        "EXTERNAL_PRODUCER_PRESENT = false",
        "IMPLEMENTATION_INDEPENDENCE = false",
        "THIRD_PARTY_INDEPENDENCE = false",
        "SCIENTIFIC_INDEPENDENCE = false",
        "FORMAL_AUTHORITY = false",
        "frozen-u64-tape-v1",
        "manifest -> rng_vector -> { attempt [ evaluation decision ] }+ -> terminal",
        "A >= P >= B >= D",
        "source_root_sha256",
        "exclusive create",
        "--reference",
        "--candidate",
        "--output",
    )
    for token in required:
        assert token in text

