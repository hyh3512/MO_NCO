#!/usr/bin/env python3
from __future__ import annotations

"""Neutral, standard-library comparator for V21e3r1 algorithm event streams.

This program validates and compares two independently produced protocol streams.
It does not run either algorithm, import ``mo_nco``, authenticate producer
identity, or establish implementation, scientific, or third-party independence.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, NoReturn, Sequence


SCHEMA = "v21e3r1_algorithm_event_stream_v1"
RECEIPT_SCHEMA = "v21e3r1_neutral_algorithm_event_comparison_receipt_v1"
ZERO_SHA256 = "0" * 64
RNG_PROTOCOL = "frozen-u64-tape-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_U64_RE = re.compile(r"[0-9a-f]{16}\Z")
_DECIMAL_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?\Z")

_COMMON_KEYS = {
    "schema",
    "event_type",
    "event_index",
    "context_sha256",
    "prev_event_sha256",
    "event_sha256",
}
_MANIFEST_KEYS = _COMMON_KEYS | {
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
}
_RNG_KEYS = _COMMON_KEYS | {"values", "rng_vector_sha256"}
_ATTEMPT_KEYS = _COMMON_KEYS | {
    "attempt_index",
    "proposal",
    "proposal_sha256",
    "evaluation_context",
    "status",
    "physical_call_started",
    "charged_evaluation_index",
    "cache_source_evaluation_index",
    "rng_start",
    "rng_stop",
}
_EVALUATION_KEYS = _COMMON_KEYS | {
    "evaluation_index",
    "attempt_index",
    "proposal_sha256",
    "problem_semantic_sha256",
    "evidence_partition",
    "search_phase_id",
    "stage_id",
    "type_id",
    "operator_id",
    "operator_call_id",
    "objective_values",
}
_DECISION_KEYS = _COMMON_KEYS | {
    "evaluation_index",
    "accepted_into_population",
    "population_replacement_count",
    "population_target_type_ids",
    "decision_reason",
    "archive_changed",
    "retained_after_update",
    "archive_size_after",
    "scalarization_id",
    "scalar_parent",
    "scalar_candidate",
    "scalar_advantage",
    "cell_id",
    "new_evaluated_cell",
    "new_nondominated_cell",
    "population_state_sha256",
    "archive_state_sha256",
}
_TERMINAL_KEYS = _COMMON_KEYS | {
    "status",
    "attempt_count",
    "physical_call_started_count",
    "charged_evaluation_count",
    "decision_count",
    "cache_hit_count",
    "rng_values_consumed",
    "population_state_sha256",
    "archive_state_sha256",
    "producer_source_manifest_sha256",
    "case_artifact_sha256",
    "problem_semantic_sha256",
    "candidate_config_sha256",
    "rng_vector_sha256",
}
_EVENT_KEYS = {
    "manifest": _MANIFEST_KEYS,
    "rng_vector": _RNG_KEYS,
    "attempt": _ATTEMPT_KEYS,
    "evaluation": _EVALUATION_KEYS,
    "decision": _DECISION_KEYS,
    "terminal": _TERMINAL_KEYS,
}


class ReplayValidationError(ValueError):
    """An event stream violates the neutral protocol."""


def _fail(message: str) -> NoReturn:
    raise ReplayValidationError(message)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_constant(token: str) -> NoReturn:
    _fail(f"non-finite JSON token is forbidden: {token}")


def _reject_float(token: str) -> NoReturn:
    _fail(f"JSON floating-point numbers are forbidden; use canonical decimal strings: {token}")


def _object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            _fail(f"duplicate JSON object key: {key}")
        output[key] = value
    return output


def _decode_line(raw: bytes, *, path: Path, line_number: int) -> dict[str, object]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReplayValidationError(
            f"{path}: line {line_number} is not UTF-8"
        ) from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_from_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ReplayValidationError(
            f"{path}: line {line_number} is not JSON: {error.msg}"
        ) from error
    if type(value) is not dict:
        _fail(f"{path}: line {line_number} is not a JSON object")
    if _canonical_bytes(value) != raw:
        _fail(f"{path}: line {line_number} is not canonical JSON")
    return value


def _read_jsonl(path: Path) -> tuple[bytes, list[dict[str, object]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    if not raw:
        _fail(f"{path}: empty event stream")
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail(f"{path}: UTF-8 BOM is forbidden")
    if b"\r" in raw:
        _fail(f"{path}: CR bytes are forbidden")
    if not raw.endswith(b"\n"):
        _fail(f"{path}: canonical JSONL must end with LF")
    lines = raw[:-1].split(b"\n")
    if any(not line for line in lines):
        _fail(f"{path}: blank JSONL records are forbidden")
    return raw, [
        _decode_line(line, path=path, line_number=index)
        for index, line in enumerate(lines, start=1)
    ]


def _exact_keys(value: dict[str, object], expected: set[str], *, label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        _fail(f"{label} has non-exact keys: missing={missing}, extra={extra}")


def _string(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{label} must be lowercase SHA-256")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be an exact integer >= {minimum}")
    return value


def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        _fail(f"{label} must be an exact Boolean")
    return value


def _nullable_integer(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label=label)


def _nullable_boolean(value: object, *, label: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, label=label)


def _nullable_string(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label=label)


def _decimal(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or _DECIMAL_RE.fullmatch(value) is None
        or value == "-0"
    ):
        _fail(f"{label} must be a canonical finite decimal string")
    return value


def _portable_source_path(value: object, *, label: str) -> str:
    path = _string(value, label=label)
    pure = PurePosixPath(path)
    if (
        path.startswith("/")
        or "\\" in path
        or pure.as_posix() != path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        _fail(f"{label} must be a normalized relative POSIX path")
    return path


def _validate_source_manifest(manifest: object) -> str:
    if type(manifest) is not dict:
        _fail("producer_source_manifest must be an object")
    _exact_keys(
        manifest,
        {"schema", "implementation_id", "language", "files", "source_root_sha256"},
        label="producer_source_manifest",
    )
    if manifest["schema"] != "v21e3r1_external_source_manifest_v1":
        _fail("unsupported producer source-manifest schema")
    _string(manifest["implementation_id"], label="source implementation_id")
    _string(manifest["language"], label="source language")
    files = manifest["files"]
    if type(files) is not list or not files:
        _fail("source files must be a non-empty list")
    paths: list[str] = []
    for index, entry in enumerate(files):
        if type(entry) is not dict:
            _fail(f"source files[{index}] must be an object")
        _exact_keys(entry, {"path", "sha256"}, label=f"source files[{index}]")
        paths.append(
            _portable_source_path(entry["path"], label=f"source files[{index}].path")
        )
        _sha256(entry["sha256"], label=f"source files[{index}].sha256")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        _fail("source files must have unique path-sorted entries")
    root = _sha256(manifest["source_root_sha256"], label="source_root_sha256")
    if root != _digest(files):
        _fail("source_root_sha256 does not bind the exact source inventory")
    return _digest(manifest)


def _context_material(manifest: dict[str, object]) -> dict[str, object]:
    return {
        key: manifest[key]
        for key in (
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
    }


def _validate_manifest(event: dict[str, object]) -> None:
    _exact_keys(event, _MANIFEST_KEYS, label="manifest event")
    _string(event["producer_id"], label="producer_id")
    source_manifest_sha = _validate_source_manifest(event["producer_source_manifest"])
    if _sha256(
        event["producer_source_manifest_sha256"],
        label="producer_source_manifest_sha256",
    ) != source_manifest_sha:
        _fail("producer_source_manifest_sha256 does not bind its manifest")
    for key in (
        "algorithm_spec_sha256",
        "case_artifact_sha256",
        "problem_semantic_sha256",
        "candidate_config_sha256",
        "rng_vector_sha256",
    ):
        _sha256(event[key], label=key)
    family = _string(event["problem_family"], label="problem_family")
    if family not in {"MOKP", "MOTSP"}:
        _fail("problem_family must be MOKP or MOTSP")
    config = event["algorithm_config"]
    schema = event["algorithm_config_schema"]
    if type(config) is not dict:
        _fail("algorithm_config must be an object")
    if (
        type(schema) is not list
        or not schema
        or any(type(key) is not str or not key for key in schema)
        or schema != sorted(schema)
        or len(schema) != len(set(schema))
    ):
        _fail("algorithm_config_schema must be a unique sorted string list")
    if set(config) != set(schema):
        _fail("algorithm_config does not have the exact frozen key set")
    if _digest(config) != event["candidate_config_sha256"]:
        _fail("candidate_config_sha256 does not bind algorithm_config")
    candidate = _string(event["candidate_id"], label="candidate_id")
    partition = _string(event["evidence_partition"], label="evidence_partition")
    budget = _integer(
        event["charged_evaluation_budget"],
        label="charged_evaluation_budget",
        minimum=1,
    )
    _integer(event["objective_count"], label="objective_count", minimum=1)
    vector_length = _integer(event["rng_vector_length"], label="rng_vector_length")
    if event["rng_protocol"] != RNG_PROTOCOL:
        _fail(f"rng_protocol must be {RNG_PROTOCOL}")
    mirrors = {
        "candidate_id": candidate,
        "charged_evaluations": budget,
        "phase": partition,
        "rng_protocol": RNG_PROTOCOL,
    }
    for key, expected in mirrors.items():
        if config.get(key) != expected or type(config.get(key)) is not type(expected):
            _fail(f"algorithm_config.{key} disagrees with manifest")
    _integer(config.get("seed"), label="algorithm_config.seed")
    if vector_length == 0:
        _fail("a successful replay design requires a non-empty RNG vector")
    observed_context = _sha256(event["context_sha256"], label="context_sha256")
    if observed_context != _digest(_context_material(event)):
        _fail("context_sha256 does not bind source/case/config/RNG context")


def _validate_common(
    event: dict[str, object],
    *,
    ordinal: int,
    expected_previous: str,
    context_sha256: str | None,
) -> str:
    event_type = event.get("event_type")
    if type(event_type) is not str or event_type not in _EVENT_KEYS:
        _fail(f"event {ordinal} has unsupported event_type")
    _exact_keys(event, _EVENT_KEYS[event_type], label=f"event {ordinal} ({event_type})")
    if event["schema"] != SCHEMA:
        _fail(f"event {ordinal} has unsupported schema")
    if _integer(event["event_index"], label=f"event {ordinal}.event_index") != ordinal:
        _fail("event_index is not zero-based contiguous")
    previous = _sha256(
        event["prev_event_sha256"], label=f"event {ordinal}.prev_event_sha256"
    )
    if previous != expected_previous:
        _fail(f"event {ordinal} breaks the predecessor hash chain")
    if context_sha256 is not None and event["context_sha256"] != context_sha256:
        _fail(f"event {ordinal} is detached from the run context")
    observed = _sha256(event["event_sha256"], label=f"event {ordinal}.event_sha256")
    core = dict(event)
    core.pop("event_sha256")
    expected = _digest(core)
    if observed != expected:
        _fail(f"event {ordinal} has an invalid event_sha256")
    return observed


def _validate_rng(event: dict[str, object], manifest: dict[str, object]) -> None:
    values = event["values"]
    if type(values) is not list or any(
        type(value) is not str or _U64_RE.fullmatch(value) is None for value in values
    ):
        _fail("rng_vector.values must be lowercase 16-hex strings")
    if len(values) != manifest["rng_vector_length"]:
        _fail("rng vector length disagrees with manifest")
    observed = _sha256(event["rng_vector_sha256"], label="rng_vector event digest")
    if observed != _digest(values) or observed != manifest["rng_vector_sha256"]:
        _fail("rng vector digest disagrees with values or manifest")


def _validate_proposal(proposal: object, family: str) -> str:
    if type(proposal) is not dict:
        _fail("attempt proposal must be an object")
    _exact_keys(proposal, {"codec", "values"}, label="attempt proposal")
    codec = _string(proposal["codec"], label="attempt proposal codec")
    values = proposal["values"]
    if type(values) is not list or not values or any(type(value) is not int for value in values):
        _fail("attempt proposal values must be a non-empty exact-integer list")
    if family == "MOKP":
        if codec != "mokp-bits-v1" or any(value not in (0, 1) for value in values):
            _fail("MOKP proposal must use mokp-bits-v1 with exact binary values")
    elif codec != "motsp-permutation-v1" or sorted(values) != list(range(len(values))):
        _fail("MOTSP proposal must be a zero-based exact permutation")
    return _digest(proposal)


def _validate_evaluation_context(context: object, partition: str) -> dict[str, object]:
    if type(context) is not dict:
        _fail("evaluation_context must be an object")
    keys = {
        "evidence_partition",
        "search_phase_id",
        "stage_id",
        "type_id",
        "operator_id",
        "operator_call_id",
        "local_search_depth",
        "retry_ordinal",
        "fallback_used",
        "operator_witness",
    }
    _exact_keys(context, keys, label="evaluation_context")
    if context["evidence_partition"] != partition:
        _fail("evaluation_context partition disagrees with manifest")
    for key in ("search_phase_id", "stage_id", "operator_id"):
        _string(context[key], label=f"evaluation_context.{key}")
    _nullable_integer(context["type_id"], label="evaluation_context.type_id")
    _integer(context["operator_call_id"], label="evaluation_context.operator_call_id")
    _integer(context["local_search_depth"], label="evaluation_context.local_search_depth")
    _integer(context["retry_ordinal"], label="evaluation_context.retry_ordinal")
    _boolean(context["fallback_used"], label="evaluation_context.fallback_used")
    return context


def _validate_attempt(
    event: dict[str, object],
    *,
    manifest: dict[str, object],
    expected_attempt: int,
    expected_evaluation: int,
    rng_cursor: int,
    evaluated_proposals: dict[str, int],
) -> tuple[str, int]:
    attempt_index = _integer(event["attempt_index"], label="attempt_index", minimum=1)
    if attempt_index != expected_attempt:
        _fail("attempt_index is not one-based contiguous")
    proposal_sha = _validate_proposal(event["proposal"], str(manifest["problem_family"]))
    if _sha256(event["proposal_sha256"], label="proposal_sha256") != proposal_sha:
        _fail("proposal_sha256 does not bind the canonical proposal")
    _validate_evaluation_context(event["evaluation_context"], str(manifest["evidence_partition"]))
    start = _integer(event["rng_start"], label="rng_start")
    stop = _integer(event["rng_stop"], label="rng_stop")
    if start != rng_cursor or stop < start or stop > manifest["rng_vector_length"]:
        _fail("attempt RNG span is non-contiguous or outside the frozen vector")
    status = event["status"]
    if status == "EVALUATED":
        if not _boolean(event["physical_call_started"], label="physical_call_started"):
            _fail("an evaluated attempt must start one physical call")
        charged = _integer(
            event["charged_evaluation_index"],
            label="charged_evaluation_index",
            minimum=1,
        )
        if charged != expected_evaluation or event["cache_source_evaluation_index"] is not None:
            _fail("evaluated attempt has inconsistent charge/cache identity")
        if proposal_sha in evaluated_proposals:
            _fail("a repeated proposal must be represented as CACHE_HIT")
    elif status == "CACHE_HIT":
        if _boolean(event["physical_call_started"], label="physical_call_started"):
            _fail("a cache hit cannot start a physical call")
        if event["charged_evaluation_index"] is not None:
            _fail("a cache hit cannot carry a charged evaluation")
        source = _integer(
            event["cache_source_evaluation_index"],
            label="cache_source_evaluation_index",
            minimum=1,
        )
        if evaluated_proposals.get(proposal_sha) != source:
            _fail("cache hit does not identify the prior evaluation of its proposal")
    else:
        _fail("attempt status must be EVALUATED or CACHE_HIT")
    return str(status), stop


def _validate_evaluation(
    event: dict[str, object],
    *,
    manifest: dict[str, object],
    attempt: dict[str, object],
    expected_evaluation: int,
) -> None:
    evaluation_index = _integer(
        event["evaluation_index"], label="evaluation_index", minimum=1
    )
    if evaluation_index != expected_evaluation:
        _fail("evaluation_index is not one-based contiguous")
    if event["attempt_index"] != attempt["attempt_index"]:
        _fail("evaluation is detached from its attempt")
    if event["proposal_sha256"] != attempt["proposal_sha256"]:
        _fail("evaluation proposal disagrees with its attempt")
    if event["problem_semantic_sha256"] != manifest["problem_semantic_sha256"]:
        _fail("evaluation is detached from the semantic problem")
    context = attempt["evaluation_context"]
    for key in (
        "evidence_partition",
        "search_phase_id",
        "stage_id",
        "type_id",
        "operator_id",
        "operator_call_id",
    ):
        if event[key] != context[key] or type(event[key]) is not type(context[key]):
            _fail(f"evaluation.{key} disagrees with its attempt context")
    objectives = event["objective_values"]
    if type(objectives) is not list or len(objectives) != manifest["objective_count"]:
        _fail("evaluation objective vector has the wrong exact length")
    for index, value in enumerate(objectives):
        _decimal(value, label=f"objective_values[{index}]")


def _validate_decision(
    event: dict[str, object],
    *,
    expected_evaluation: int,
) -> tuple[str, str]:
    if _integer(event["evaluation_index"], label="decision evaluation_index", minimum=1) != expected_evaluation:
        _fail("decision is detached from its evaluation")
    accepted = _boolean(
        event["accepted_into_population"], label="accepted_into_population"
    )
    replacements = _integer(
        event["population_replacement_count"], label="population_replacement_count"
    )
    targets = event["population_target_type_ids"]
    if type(targets) is not list or any(type(value) is not int or value < 0 for value in targets):
        _fail("population_target_type_ids must be exact nonnegative integers")
    if len(targets) != replacements or len(targets) != len(set(targets)):
        _fail("replacement count and unique target set disagree")
    if not accepted and replacements:
        _fail("a rejected population decision cannot replace targets")
    _string(event["decision_reason"], label="decision_reason")
    _boolean(event["archive_changed"], label="archive_changed")
    _boolean(event["retained_after_update"], label="retained_after_update")
    _integer(event["archive_size_after"], label="archive_size_after")
    scalarization_id = _nullable_string(
        event["scalarization_id"], label="scalarization_id"
    )
    scalars = (
        event["scalar_parent"],
        event["scalar_candidate"],
        event["scalar_advantage"],
    )
    if any(value is None for value in scalars) and not all(value is None for value in scalars):
        _fail("scalar parent/candidate/advantage must be jointly present or null")
    if (scalarization_id is None) != all(value is None for value in scalars):
        _fail("scalarization_id and scalar witnesses must be jointly present or null")
    for name, value in zip(
        ("scalar_parent", "scalar_candidate", "scalar_advantage"), scalars
    ):
        if value is not None:
            _decimal(value, label=name)
    cell_id = _nullable_string(event["cell_id"], label="cell_id")
    novelty = (
        _nullable_boolean(event["new_evaluated_cell"], label="new_evaluated_cell"),
        _nullable_boolean(
            event["new_nondominated_cell"], label="new_nondominated_cell"
        ),
    )
    if (
        any(value is None for value in novelty)
        and not all(value is None for value in novelty)
    ) or ((cell_id is None) != all(value is None for value in novelty)):
        _fail("cell_id and both novelty witness flags must be jointly present or null")
    population_root = _sha256(event["population_state_sha256"], label="population_state_sha256")
    archive_root = _sha256(event["archive_state_sha256"], label="archive_state_sha256")
    return population_root, archive_root


def _validate_terminal(
    event: dict[str, object],
    *,
    manifest: dict[str, object],
    attempts: int,
    physical_calls: int,
    evaluations: int,
    decisions: int,
    cache_hits: int,
    rng_cursor: int,
    population_root: str,
    archive_root: str,
) -> None:
    if event["status"] != "SUCCESS":
        _fail("algorithm comparison accepts complete SUCCESS streams only")
    counts = {
        "attempt_count": attempts,
        "physical_call_started_count": physical_calls,
        "charged_evaluation_count": evaluations,
        "decision_count": decisions,
        "cache_hit_count": cache_hits,
        "rng_values_consumed": rng_cursor,
    }
    for key, expected in counts.items():
        if _integer(event[key], label=f"terminal.{key}") != expected:
            _fail(f"terminal.{key} disagrees with observed events")
    if not (
        evaluations == manifest["charged_evaluation_budget"]
        and physical_calls == evaluations
        and decisions == evaluations
        and attempts == evaluations + cache_hits
        and rng_cursor == manifest["rng_vector_length"]
    ):
        _fail("terminal event violates exact-budget/accounting/RNG completeness")
    bindings = {
        "producer_source_manifest_sha256": manifest["producer_source_manifest_sha256"],
        "case_artifact_sha256": manifest["case_artifact_sha256"],
        "problem_semantic_sha256": manifest["problem_semantic_sha256"],
        "candidate_config_sha256": manifest["candidate_config_sha256"],
        "rng_vector_sha256": manifest["rng_vector_sha256"],
        "population_state_sha256": population_root,
        "archive_state_sha256": archive_root,
    }
    for key, expected in bindings.items():
        if _sha256(event[key], label=f"terminal.{key}") != expected:
            _fail(f"terminal.{key} is detached from the completed run")


def _validate_stream(path: Path) -> dict[str, object]:
    raw, events = _read_jsonl(path)
    if not events or events[0].get("event_type") != "manifest":
        _fail(f"{path}: first event must be manifest")
    manifest = events[0]
    previous = ZERO_SHA256
    context_sha: str | None = None
    for ordinal, event in enumerate(events):
        previous = _validate_common(
            event,
            ordinal=ordinal,
            expected_previous=previous,
            context_sha256=context_sha,
        )
        if ordinal == 0:
            _validate_manifest(event)
            context_sha = str(event["context_sha256"])
    if len(events) < 2 or events[1]["event_type"] != "rng_vector":
        _fail(f"{path}: second event must be rng_vector")
    _validate_rng(events[1], manifest)
    if len(events) < 3 or events[-1]["event_type"] != "terminal":
        _fail(f"{path}: stream must end with exactly one terminal event")

    attempts = physical_calls = evaluations = decisions = cache_hits = 0
    rng_cursor = 0
    evaluated_proposals: dict[str, int] = {}
    population_root = ZERO_SHA256
    archive_root = ZERO_SHA256
    position = 2
    while position < len(events) - 1:
        attempt = events[position]
        if attempt["event_type"] != "attempt":
            _fail(f"{path}: expected attempt at event {position}")
        attempts += 1
        status, rng_cursor = _validate_attempt(
            attempt,
            manifest=manifest,
            expected_attempt=attempts,
            expected_evaluation=evaluations + 1,
            rng_cursor=rng_cursor,
            evaluated_proposals=evaluated_proposals,
        )
        position += 1
        if status == "CACHE_HIT":
            cache_hits += 1
            continue
        physical_calls += 1
        if position >= len(events) - 1 or events[position]["event_type"] != "evaluation":
            _fail(f"{path}: evaluated attempt lacks its evaluation event")
        evaluation = events[position]
        evaluations += 1
        _validate_evaluation(
            evaluation,
            manifest=manifest,
            attempt=attempt,
            expected_evaluation=evaluations,
        )
        evaluated_proposals[str(attempt["proposal_sha256"])] = evaluations
        position += 1
        if position >= len(events) - 1 or events[position]["event_type"] != "decision":
            _fail(f"{path}: evaluation lacks its decision event")
        decision = events[position]
        decisions += 1
        population_root, archive_root = _validate_decision(
            decision, expected_evaluation=evaluations
        )
        position += 1
    if position != len(events) - 1 or events[-1]["event_type"] != "terminal":
        _fail(f"{path}: stream must end with exactly one terminal event")
    _validate_terminal(
        events[-1],
        manifest=manifest,
        attempts=attempts,
        physical_calls=physical_calls,
        evaluations=evaluations,
        decisions=decisions,
        cache_hits=cache_hits,
        rng_cursor=rng_cursor,
        population_root=population_root,
        archive_root=archive_root,
    )
    return {
        "path": path,
        "raw": raw,
        "events": events,
        "manifest": manifest,
        "event_count": len(events),
        "terminal_event_sha256": events[-1]["event_sha256"],
    }


def _semantic_event(event: dict[str, object]) -> dict[str, object]:
    output = dict(event)
    for key in ("context_sha256", "prev_event_sha256", "event_sha256"):
        output.pop(key)
    if output["event_type"] == "manifest":
        for key in (
            "producer_id",
            "producer_source_manifest",
            "producer_source_manifest_sha256",
        ):
            output.pop(key)
    elif output["event_type"] == "terminal":
        output.pop("producer_source_manifest_sha256")
    return output


def _first_mismatch(
    reference: Sequence[dict[str, object]], candidate: Sequence[dict[str, object]]
) -> dict[str, object] | None:
    limit = max(len(reference), len(candidate))
    for ordinal in range(limit):
        left = reference[ordinal] if ordinal < len(reference) else None
        right = candidate[ordinal] if ordinal < len(candidate) else None
        if left != right:
            return {
                "event_index": ordinal,
                "reference_event_type": None if left is None else left.get("event_type"),
                "candidate_event_type": None if right is None else right.get("event_type"),
                "reference_semantic_sha256": _digest(left),
                "candidate_semantic_sha256": _digest(right),
            }
    return None


def compare_event_streams(
    *,
    reference_stream: str | Path,
    candidate_stream: str | Path,
    output_receipt: str | Path,
) -> dict[str, object]:
    output = Path(output_receipt).resolve()
    if output.exists():
        raise FileExistsError(output)
    reference = _validate_stream(Path(reference_stream).resolve())
    candidate = _validate_stream(Path(candidate_stream).resolve())
    left_manifest = reference["manifest"]
    right_manifest = candidate["manifest"]
    assert isinstance(left_manifest, dict) and isinstance(right_manifest, dict)
    left_events = [_semantic_event(event) for event in reference["events"]]
    right_events = [_semantic_event(event) for event in candidate["events"]]
    checks = {
        "algorithm_events_equal": left_events == right_events,
        "case_binding_equal": (
            left_manifest["case_artifact_sha256"],
            left_manifest["problem_semantic_sha256"],
        )
        == (
            right_manifest["case_artifact_sha256"],
            right_manifest["problem_semantic_sha256"],
        ),
        "config_binding_equal": (
            left_manifest["algorithm_config_schema"],
            left_manifest["algorithm_config"],
            left_manifest["candidate_config_sha256"],
        )
        == (
            right_manifest["algorithm_config_schema"],
            right_manifest["algorithm_config"],
            right_manifest["candidate_config_sha256"],
        ),
        "rng_vector_equal": (
            left_manifest["rng_vector_sha256"],
            reference["events"][1]["values"],
        )
        == (
            right_manifest["rng_vector_sha256"],
            candidate["events"][1]["values"],
        ),
    }
    status = (
        "PASS_NEUTRAL_EVENT_STREAM_COMPARISON"
        if all(checks.values())
        else "FAIL_ALGORITHM_EVENT_STREAM_MISMATCH"
    )

    def stream_binding(stream: dict[str, object]) -> dict[str, object]:
        manifest = stream["manifest"]
        raw = stream["raw"]
        assert isinstance(manifest, dict) and isinstance(raw, bytes)
        return {
            "bytes": len(raw),
            "sha256": _file_digest(raw),
            "event_count": stream["event_count"],
            "terminal_event_sha256": stream["terminal_event_sha256"],
            "producer_id": manifest["producer_id"],
            "producer_source_manifest_sha256": manifest[
                "producer_source_manifest_sha256"
            ],
            "context_sha256": manifest["context_sha256"],
        }

    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "status": status,
        "checks": checks,
        "streams": {
            "reference": stream_binding(reference),
            "candidate": stream_binding(candidate),
        },
        "first_mismatch": _first_mismatch(left_events, right_events),
        "distinct_producer_ids_observed": (
            left_manifest["producer_id"] != right_manifest["producer_id"]
        ),
        "distinct_source_manifests_observed": (
            left_manifest["producer_source_manifest_sha256"]
            != right_manifest["producer_source_manifest_sha256"]
        ),
        "neutral_comparator": True,
        "reference_algorithm_producer_present": False,
        "external_producer_present": False,
        "producer_authorship_authenticated": False,
        "independent_custody_verified": False,
        "implementation_independence": False,
        "third_party_independence": False,
        "scientific_independence": False,
        "formal_authority": False,
        "scope": "design_and_golden_corpus_only_no_external_producer_or_custody_claim",
    }
    receipt["receipt_payload_sha256"] = _digest(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
        handle.write("\n")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and neutrally compare two V21e3r1 algorithm JSONL streams."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = compare_event_streams(
            reference_stream=args.reference,
            candidate_stream=args.candidate,
            output_receipt=args.output,
        )
    except (ReplayValidationError, OSError) as error:
        print(
            json.dumps(
                {
                    "schema": "v21e3r1_neutral_comparator_error_v1",
                    "status": "ERROR_INVALID_OR_UNWRITABLE_INPUT",
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 3
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False))
    return 0 if receipt["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
