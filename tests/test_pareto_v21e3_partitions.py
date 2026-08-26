from __future__ import annotations

import json

import pytest

from mo_nco.pareto_v21_partitions import materialize_v21_partitions
from mo_nco.pareto_v21e3_partitions import materialize_v21e3_partition


def test_v21e3_generator_epoch_has_frozen_known_answer_vectors(tmp_path) -> None:
    manifests = materialize_v21_partitions(
        tmp_path / "known-answer-v21e3",
        master_seed=b"v21-e3-case-known-answer",
        sizes=(4,),
        development_cases_per_size=1,
        calibration_cases_per_size=1,
        calibration_epoch="v21e3",
    )
    manifest = json.loads(manifests["development"].read_text(encoding="utf-8"))
    observed = {
        case["case_id"]: (
            case["artifact"]["sha256"],
            case["fingerprints"]["problem_sha256"],
            case["fingerprints"]["semantic_sha256"],
            case["generator"]["lineage_sha256"],
        )
        for case in manifest["cases"]
    }

    assert observed == {
        "v21e3-mokp-development-n4-s00": (
            "fa89ba3c83b9546b02050040f859e4160d1848c85f8e246c718fca1489808b93",
            "bbffc21e0725d52169e5cd24cae3afd453e72c81bfc3976fee9541feda6ffaaa",
            "8823f2e06b2de6757e600ad0f453dd0e9125d87153f35415ea65b18617c7de31",
            "63b44e94e07eff378d7a2b3752b5ee1f8a4c68c28d48741a310f5edaa725bc1d",
        ),
        "v21e3-motsp-development-n4-s00": (
            "d0a550e678e7cb7161bf7a3cee75701feedc1102cefae48a2db88c9c8e595145",
            "4803f33ef8a03778df19cb76837c58b3e55245c4429f46db07b6465848dfffff",
            "843cc3d054e63c6726b5639255831853a87101efc43fd3924b0738b300cabb9e",
            "6417351ba2a7d6b0d1d4be6eda0b9432522ceb552c7f045fc16f692f89221903",
        ),
    }
    assert all(case["split"] == "development" for case in manifest["cases"])
    assert manifest["suite_id"] == "pareto-v21-development-authors-generated-v3"
    assert manifest["calibration_epoch"] == "v21e3"


def test_v21e3_can_materialize_each_split_only_when_it_is_authorized(
    tmp_path,
) -> None:
    development = materialize_v21e3_partition(
        tmp_path / "development",
        split="development",
        master_seed=b"development-only-entropy",
        sizes=(4,),
        cases_per_size=1,
    )

    assert development == tmp_path / "development" / "case_manifest.json"
    payload = json.loads(development.read_text(encoding="utf-8"))
    assert payload["split"] == "development"
    assert payload["calibration_epoch"] == "v21e3"
    assert not (tmp_path / "development" / "calibration").exists()
    with pytest.raises(FileExistsError):
        materialize_v21e3_partition(
            tmp_path / "development",
            split="development",
            master_seed=b"another-seed",
            sizes=(4,),
            cases_per_size=1,
        )
    with pytest.raises(ValueError, match="split"):
        materialize_v21e3_partition(
            tmp_path / "formal",
            split="formal",
            master_seed=b"forbidden",
            sizes=(4,),
            cases_per_size=1,
        )

