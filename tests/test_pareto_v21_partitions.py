from __future__ import annotations

from copy import deepcopy
from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from mo_nco.pareto_v21_partitions import (
    Shake256CounterRNG,
    audit_partition_overlap,
    build_prior_exposure_registry,
    extend_prior_exposure_registry,
    fingerprint_instance,
    load_partition_case,
    materialize_v21_partitions,
    raw_child_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_name_free_mokp_fingerprint_ignores_case_and_generator_labels() -> None:
    instance = {
        "schema": "ijoc_v21_mokp_integer_instance_v1",
        "family": "MOKP",
        "case_id": "development-a",
        "num_items": 4,
        "num_objectives": 2,
        "item_weights": [2, 3, 5, 7],
        "profits_by_objective": [[11, 13, 17, 19], [23, 29, 31, 37]],
        "capacity": 9,
        "generator": {"name": "label-a", "stream_id": "a"},
    }
    renamed = deepcopy(instance)
    renamed["case_id"] = "confirmation-z"
    renamed["generator"] = {"name": "label-b", "stream_id": "z"}

    first = fingerprint_instance(instance)
    second = fingerprint_instance(renamed)

    assert first == second
    assert set(first.component_sha256) == {
        "constraint",
        "objective_0",
        "objective_1",
    }
    assert first.family == "MOKP"

    permutation = (2, 0, 3, 1)
    permuted = deepcopy(instance)
    permuted["item_weights"] = [instance["item_weights"][i] for i in permutation]
    permuted["profits_by_objective"] = [
        [row[i] for i in permutation] for row in instance["profits_by_objective"]
    ]
    permuted_fingerprint = fingerprint_instance(permuted)
    assert permuted_fingerprint.semantic_sha256 == first.semantic_sha256
    assert sorted(permuted_fingerprint.component_sha256.values()) == sorted(
        first.component_sha256.values()
    )
    assert permuted_fingerprint.problem_sha256 != first.problem_sha256


def test_motsp_semantic_fingerprint_is_name_free_and_objective_order_free() -> None:
    instance = {
        "schema": "pareto_v21_motsp_integer_coordinates_v1",
        "family": "MOTSP",
        "case_id": "motsp-development-n4-s00",
        "num_cities": 4,
        "num_objectives": 2,
        "coordinates_by_objective": [
            [[0, 0], [3, 0], [3, 4], [0, 4]],
            [[1, 2], [5, 2], [5, 8], [1, 8]],
        ],
        "generator": {"stream_id": "unused-by-fingerprint"},
    }
    swapped = deepcopy(instance)
    swapped["case_id"] = "a-completely-different-label"
    swapped["coordinates_by_objective"] = list(
        reversed(swapped["coordinates_by_objective"])
    )

    first = fingerprint_instance(instance)
    second = fingerprint_instance(swapped)

    assert first.semantic_sha256 == second.semantic_sha256
    assert first.problem_sha256 != second.problem_sha256
    assert sorted(first.component_sha256.values()) == sorted(
        second.component_sha256.values()
    )
    city_permutation = (2, 0, 3, 1)
    permuted = deepcopy(instance)
    permuted["coordinates_by_objective"] = [
        [rows[index] for index in city_permutation]
        for rows in instance["coordinates_by_objective"]
    ]
    permuted_fingerprint = fingerprint_instance(permuted)
    assert permuted_fingerprint.semantic_sha256 == first.semantic_sha256
    assert sorted(permuted_fingerprint.component_sha256.values()) == sorted(
        first.component_sha256.values()
    )
    assert permuted_fingerprint.problem_sha256 != first.problem_sha256


def test_shake256_counter_rng_has_frozen_known_answer_and_chunking() -> None:
    expected = "e90ce6c4c7c31b6bec926a1771f9ad90ca3498a4f7d74b8f90f24ef7a6cba32e"
    whole = Shake256CounterRNG(
        seed=b"v21-known-answer-seed", domain="known-answer"
    ).read(32)
    chunked_rng = Shake256CounterRNG(
        seed=b"v21-known-answer-seed", domain="known-answer"
    )
    chunked = chunked_rng.read(7) + chunked_rng.read(25)

    assert whole.hex() == expected
    assert chunked == whole


def test_materialization_is_exclusive_prospective_and_never_creates_formal(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prospective"
    manifests = materialize_v21_partitions(
        output,
        master_seed=b"test-only-v21-master-seed",
        sizes=(8,),
        development_cases_per_size=1,
        calibration_cases_per_size=1,
    )

    assert set(manifests) == {"development", "selection", "confirmation"}
    assert not (output / "formal_study").exists()
    for split, manifest_path in manifests.items():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema"] == "pareto_v21_partition_manifest_v1"
        assert manifest["split"] == split
        assert manifest["external_independence_status"] == "NOT_ESTABLISHED"
        assert len(manifest["cases"]) == 2
        for case in manifest["cases"]:
            payload = load_partition_case(manifest_path, case["case_id"])
            artifact = manifest_path.parent / case["artifact"]["path"]
            assert hashlib.sha256(artifact.read_bytes()).hexdigest() == case["artifact"]["sha256"]
            assert fingerprint_instance(payload).semantic_sha256 == case["fingerprints"]["semantic_sha256"]

    before = {
        path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError):
        materialize_v21_partitions(
            output,
            master_seed=b"different-seed-must-not-overwrite",
            sizes=(8,),
            development_cases_per_size=1,
            calibration_cases_per_size=1,
        )
    after = {
        path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_new_calibration_epoch_is_namespaced_and_disjoint_from_v2(
    tmp_path: Path,
) -> None:
    seed = b"same-seed-must-still-produce-a-new-epoch"
    v2_manifests = materialize_v21_partitions(
        tmp_path / "prospective-v2",
        master_seed=seed,
        sizes=(5,),
        development_cases_per_size=1,
        calibration_cases_per_size=1,
    )
    v3_manifests = materialize_v21_partitions(
        tmp_path / "prospective-v3",
        master_seed=seed,
        sizes=(5,),
        development_cases_per_size=1,
        calibration_cases_per_size=1,
        calibration_epoch="v21e2",
    )

    for split in ("development", "selection", "confirmation"):
        v2 = json.loads(v2_manifests[split].read_text(encoding="utf-8"))
        v3 = json.loads(v3_manifests[split].read_text(encoding="utf-8"))
        assert "calibration_epoch" not in v2
        assert v3["calibration_epoch"] == "v21e2"
        assert v3["suite_id"] == f"pareto-v21-{split}-authors-generated-v3"
        assert v3["generator_contract"]["calibration_epoch"] == "v21e2"
        assert (
            v3["generator_contract"]["instance_generator_version"]
            == "pareto-v21-instance-generator-v3"
        )

        v2_ids = {case["case_id"] for case in v2["cases"]}
        v3_ids = {case["case_id"] for case in v3["cases"]}
        assert v2_ids.isdisjoint(v3_ids)
        assert all(case_id.startswith("v21e2-") for case_id in v3_ids)

        v2_streams = {case["generator"]["stream_id"] for case in v2["cases"]}
        v3_streams = {case["generator"]["stream_id"] for case in v3["cases"]}
        assert v2_streams.isdisjoint(v3_streams)
        assert all(stream.startswith("pareto-v21/v21e2/") for stream in v3_streams)
        assert {
            case["fingerprints"]["semantic_sha256"] for case in v2["cases"]
        }.isdisjoint(
            case["fingerprints"]["semantic_sha256"] for case in v3["cases"]
        )
        assert {
            case["generator"]["lineage_sha256"] for case in v2["cases"]
        }.isdisjoint(
            case["generator"]["lineage_sha256"] for case in v3["cases"]
        )
        for case in v3["cases"]:
            assert case["generator"]["calibration_epoch"] == "v21e2"
            load_partition_case(v3_manifests[split], case["case_id"])

    empty_registry = {
        "schema": "pareto_v21_prior_exposure_registry_v1",
        "source_release": "empty",
        "scope": "test",
        "case_count": 0,
        "sources": [],
        "entries": [],
        "indexes": {},
    }
    v2_registry = extend_prior_exposure_registry(
        empty_registry,
        list(v2_manifests.values()),
    )
    receipt = audit_partition_overlap(
        list(v3_manifests.values()),
        prior_registry=v2_registry,
    )
    assert receipt["status"] == "PASS"
    assert receipt["collisions"] == []


def test_prior_registry_ingests_v20_mokp_and_tsplib_semantics(tmp_path: Path) -> None:
    v20 = tmp_path / "ijoc_submission_v20"
    calibration = v20 / "calibration"
    formal = v20 / "formal_study"
    mokp_path = calibration / "instances" / "old-mokp.json"
    mokp = {
        "schema": "ijoc_mokp_integer_instance_v1",
        "family": "MOKP",
        "case_id": "old-mokp",
        "num_items": 4,
        "num_objectives": 2,
        "item_weights": [2, 3, 5, 7],
        "profits_by_objective": [[11, 13, 17, 19], [23, 29, 31, 37]],
        "capacity": 9,
        "generator": {"name": "python_random_mt19937_integer_mokp_v1", "seed": 41},
    }
    _write_json(mokp_path, mokp)
    tsp_texts = (
        "NAME: old-a\nTYPE: TSP\nDIMENSION: 3\nEDGE_WEIGHT_TYPE: EXPLICIT\n"
        "EDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n0 2 3\n2 0 4\n3 4 0\nEOF\n",
        "NAME: old-b\nTYPE: TSP\nDIMENSION: 3\nEDGE_WEIGHT_TYPE: EXPLICIT\n"
        "EDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n0 5 6\n5 0 7\n6 7 0\nEOF\n",
    )
    tsp_paths = []
    for index, text_value in enumerate(tsp_texts, start=1):
        path = formal / "instances" / "motsp" / f"old-motsp-objective-{index}.tsp"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text_value, encoding="utf-8")
        tsp_paths.append(path)

    _write_json(
        calibration / "case_manifest.json",
        {
            "schema": "ijoc_case_suite_manifest_v1",
            "cases": [
                {
                    "case_id": "old-mokp",
                    "family": "MOKP",
                    "problem_sha256": "1" * 64,
                    "source_provenance": {
                        "suite": "ijoc_integer_mokp_generated_v1",
                        "generator_seed": 41,
                    },
                    "artifacts": [
                        {
                            "path": "instances/old-mokp.json",
                            "sha256": _sha256(mokp_path),
                            "bytes": mokp_path.stat().st_size,
                        }
                    ],
                }
            ],
        },
    )
    _write_json(
        formal / "case_manifest.json",
        {
            "schema": "ijoc_case_suite_manifest_v1",
            "cases": [
                {
                    "case_id": "old-motsp",
                    "family": "MOTSP",
                    "problem_sha256": "2" * 64,
                    "source_provenance": {"suite": "old-public-motsp"},
                    "artifacts": [
                        {
                            "path": f"instances/motsp/{path.name}",
                            "sha256": _sha256(path),
                            "bytes": path.stat().st_size,
                        }
                        for path in tsp_paths
                    ],
                }
            ],
        },
    )

    registry_path = tmp_path / "prior_registry.json"
    registry = build_prior_exposure_registry(v20, output_path=registry_path)

    assert registry["schema"] == "pareto_v21_prior_exposure_registry_v1"
    assert registry["case_count"] == 2
    assert len(registry["indexes"]["raw_artifact_sha256"]) == 3
    assert len(registry["indexes"]["semantic_sha256"]) == 2
    motsp_entry = next(entry for entry in registry["entries"] if entry["family"] == "MOTSP")
    assert set(motsp_entry["component_sha256"]) == {"objective_0", "objective_1"}
    assert registry_path.exists()
    with pytest.raises(FileExistsError):
        build_prior_exposure_registry(v20, output_path=registry_path)


def test_overlap_audit_forbids_cross_case_shared_components(tmp_path: Path) -> None:
    output = tmp_path / "prospective"
    manifests = materialize_v21_partitions(
        output,
        master_seed=b"component-audit-seed",
        sizes=(6,),
        development_cases_per_size=1,
        calibration_cases_per_size=1,
    )
    clean = audit_partition_overlap(list(manifests.values()))
    assert clean["status"] == "PASS"

    selection_manifest = json.loads(
        manifests["selection"].read_text(encoding="utf-8")
    )
    confirmation_manifest = json.loads(
        manifests["confirmation"].read_text(encoding="utf-8")
    )
    selection_case = next(
        case for case in selection_manifest["cases"] if case["family"] == "MOTSP"
    )
    confirmation_case = next(
        case for case in confirmation_manifest["cases"] if case["family"] == "MOTSP"
    )
    selection_payload = load_partition_case(
        manifests["selection"], selection_case["case_id"]
    )
    confirmation_payload = load_partition_case(
        manifests["confirmation"], confirmation_case["case_id"]
    )
    confirmation_payload["coordinates_by_objective"][0] = deepcopy(
        selection_payload["coordinates_by_objective"][0]
    )
    confirmation_artifact = (
        manifests["confirmation"].parent / confirmation_case["artifact"]["path"]
    )
    _write_json(confirmation_artifact, confirmation_payload)
    changed_fingerprint = fingerprint_instance(confirmation_payload)
    confirmation_case["artifact"]["sha256"] = _sha256(confirmation_artifact)
    confirmation_case["artifact"]["bytes"] = confirmation_artifact.stat().st_size
    confirmation_case["fingerprints"] = {
        "problem_sha256": changed_fingerprint.problem_sha256,
        "semantic_sha256": changed_fingerprint.semantic_sha256,
        "component_sha256": dict(changed_fingerprint.component_sha256),
    }
    _write_json(manifests["confirmation"], confirmation_manifest)

    collision = audit_partition_overlap(list(manifests.values()))
    assert collision["status"] == "FAIL"
    assert any(
        item["kind"] == "cross_case_component_sha256"
        for item in collision["collisions"]
    )


def test_overlap_audit_checks_every_prior_exposure_dimension(tmp_path: Path) -> None:
    manifests = materialize_v21_partitions(
        tmp_path / "prospective",
        master_seed=b"all-prior-dimensions-seed",
        sizes=(5,),
        development_cases_per_size=1,
        calibration_cases_per_size=1,
    )
    development = json.loads(manifests["development"].read_text(encoding="utf-8"))
    case = next(item for item in development["cases"] if item["family"] == "MOKP")
    empty_indexes = {
        "case_id": [],
        "case_id_sha256": [],
        "raw_artifact_sha256": [],
        "raw_child_sha256": [],
        "problem_sha256": [],
        "name_free_problem_sha256": [],
        "semantic_sha256": [],
        "component_sha256": [],
        "generator_lineage_sha256": [],
        "generator_invocation_sha256": [],
    }
    probes = [
        ("case_id", case["case_id"], "prior_case_id"),
        (
            "case_id_sha256",
            hashlib.sha256(case["case_id"].encode("utf-8")).hexdigest(),
            "prior_case_id_sha256",
        ),
        ("raw_artifact_sha256", case["artifact"]["sha256"], "prior_raw_artifact_sha256"),
        (
            "raw_child_sha256",
            next(iter(case["fingerprints"]["raw_child_sha256"].values())),
            "prior_raw_child_sha256",
        ),
        (
            "problem_sha256",
            case["fingerprints"]["problem_sha256"],
            "prior_problem_sha256",
        ),
        (
            "name_free_problem_sha256",
            case["fingerprints"]["problem_sha256"],
            "prior_name_free_problem_sha256",
        ),
        (
            "semantic_sha256",
            case["fingerprints"]["semantic_sha256"],
            "prior_semantic_sha256",
        ),
        (
            "component_sha256",
            next(iter(case["fingerprints"]["component_sha256"].values())),
            "prior_component_sha256",
        ),
        (
            "generator_lineage_sha256",
            case["generator"]["lineage_sha256"],
            "prior_generator_lineage_sha256",
        ),
        (
            "generator_invocation_sha256",
            case["generator"]["invocation_sha256"],
            "prior_generator_invocation_sha256",
        ),
    ]
    for index_name, value, expected_kind in probes:
        indexes = deepcopy(empty_indexes)
        indexes[index_name] = [value]
        receipt = audit_partition_overlap(
            list(manifests.values()),
            prior_registry={
                "schema": "pareto_v21_prior_exposure_registry_v1",
                "indexes": indexes,
            },
        )
        assert receipt["status"] == "FAIL", index_name
        assert any(
            collision["kind"] == expected_kind
            for collision in receipt["collisions"]
        ), index_name


def test_case_loader_rejects_unbound_generator_metadata(tmp_path: Path) -> None:
    manifests = materialize_v21_partitions(
        tmp_path / "prospective",
        master_seed=b"generator-binding-seed",
        sizes=(4,),
        development_cases_per_size=1,
        calibration_cases_per_size=1,
    )
    manifest_path = manifests["development"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    case["generator"]["lineage_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="generator binding"):
        load_partition_case(manifest_path, case["case_id"])


def test_calibration_regimes_are_balanced_and_raw_children_are_bound(
    tmp_path: Path,
) -> None:
    manifests = materialize_v21_partitions(
        tmp_path / "prospective",
        master_seed=b"balanced-regime-seed",
        sizes=(7,),
        development_cases_per_size=1,
        calibration_cases_per_size=5,
    )
    expected_regimes = {
        "independent",
        "objective_correlated",
        "objective_conflicting",
        "structured",
        "heterogeneous",
    }
    for split in ("selection", "confirmation"):
        manifest_path = manifests[split]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for family in ("MOTSP", "MOKP"):
            family_cases = [case for case in manifest["cases"] if case["family"] == family]
            assert Counter(case["regime"] for case in family_cases) == Counter(
                {regime: 1 for regime in expected_regimes}
            )
            for case in family_cases:
                payload = load_partition_case(manifest_path, case["case_id"])
                assert case["fingerprints"]["raw_child_sha256"] == raw_child_sha256(
                    payload
                )
    receipt = audit_partition_overlap(list(manifests.values()))
    assert receipt["status"] == "PASS"
    assert receipt["calibration_regime_balance_status"] == "PASS"


def test_superseded_v21_packets_can_be_added_to_prior_exposure_registry(
    tmp_path: Path,
) -> None:
    manifests = materialize_v21_partitions(
        tmp_path / "superseded",
        master_seed=b"superseded-suite-seed",
        sizes=(4,),
        development_cases_per_size=1,
        calibration_cases_per_size=1,
    )
    empty_registry = {
        "schema": "pareto_v21_prior_exposure_registry_v1",
        "source_release": "empty",
        "scope": "test",
        "case_count": 0,
        "sources": [],
        "entries": [],
        "indexes": {},
    }
    extended = extend_prior_exposure_registry(
        empty_registry,
        list(manifests.values()),
    )

    assert extended["case_count"] == 6
    assert len(extended["indexes"]["raw_artifact_sha256"]) == 6
    assert extended["indexes"]["raw_child_sha256"]
    assert all(
        entry["source_role"].startswith("superseded_v21_")
        for entry in extended["entries"]
    )


def test_instance_generator_has_frozen_case_digest_vectors(tmp_path: Path) -> None:
    manifests = materialize_v21_partitions(
        tmp_path / "known-answer",
        master_seed=b"v21-case-known-answer",
        sizes=(4,),
        development_cases_per_size=1,
        calibration_cases_per_size=1,
    )
    manifest = json.loads(manifests["development"].read_text(encoding="utf-8"))
    observed = {
        case["case_id"]: (
            case["artifact"]["sha256"],
            case["fingerprints"]["problem_sha256"],
            case["fingerprints"]["semantic_sha256"],
        )
        for case in manifest["cases"]
    }
    assert observed == {
        "v21v2-mokp-development-n4-s00": (
            "a6ee6ea93e461fd4d0be7bf7a6ae4acdf8e858f5552010011e7bd10a2c6e1b5e",
            "f4db57da8a80a2ed6951abfe0354ad98b9b283b49dac9ce83d6d4e324f63b43c",
            "6a742fc8fe7af942ab0a8bef9d8963d7deb462ab9e6be72a03b61fadf957f751",
        ),
        "v21v2-motsp-development-n4-s00": (
            "4c865be95127f1ad4c3ec9f20879ef012d116d9d72c6c585d3450d22e004c4cf",
            "447551209795d83e13e2858fb185de85790b8b3e299b2424b25e10d754876735",
            "062f73ee48fcddac77d1bc4a4760c9863a1797a2af3b10946f165c011593253a",
        ),
    }


def test_v21e2_instance_generator_has_frozen_case_digest_vectors(
    tmp_path: Path,
) -> None:
    manifests = materialize_v21_partitions(
        tmp_path / "known-answer-v21e2",
        master_seed=b"v21-e2-case-known-answer",
        sizes=(4,),
        development_cases_per_size=1,
        calibration_cases_per_size=1,
        calibration_epoch="v21e2",
    )
    manifest = json.loads(manifests["development"].read_text(encoding="utf-8"))
    observed = {
        case["case_id"]: (
            case["artifact"]["sha256"],
            case["fingerprints"]["problem_sha256"],
            case["fingerprints"]["semantic_sha256"],
        )
        for case in manifest["cases"]
    }
    assert observed == {
        "v21e2-mokp-development-n4-s00": (
            "a26828545fea53f304bf9b67eb533ccd9b54af50be76cefea2bda677c5daa8c4",
            "ef9d680e57107aa77b226bf362b59a8ad278de9aea7b240d441654e8d4993fd4",
            "09db9696ece90299c8e2254b09afc1d867ee5e4f21696fa45345fbdbe75a6bb4",
        ),
        "v21e2-motsp-development-n4-s00": (
            "98fb0c06fd6966bcaaad6810371616759720901073095ebd9b9c25666a6a2a04",
            "3fe44248b947ac61936cf904d6ea8753d5e68d2ed874e078cf588935f5df5253",
            "f383706f709ebbc954bf1dc2809a4aa94182aea47cce9527c7b2c72034bb5c29",
        ),
    }

