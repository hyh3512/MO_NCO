from __future__ import annotations

import hashlib
import itertools
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from mo_nco.instance import MultiObjectiveTSPInstance, instance_sha256
from mo_nco.pareto_bounds import nondominated_points
from mo_nco.pareto_execution_contract import (
    DOMAIN_SEPARATED_SEED_SCHEMA_V1,
    PARETO_SMC_V13_ALGORITHM_ROLE,
    PARETO_SMC_V13_ALGORITHM_VERSION,
    derive_domain_separated_seed,
)
from mo_nco.pareto_fixed_reference_spec import (
    load_fixed_reference_certificate_specification,
)
from mo_nco.pareto_preconfirm_receipt import (
    PreconfirmReceiptVerificationError,
    create_unsigned_preconfirm_receipt_request,
    sign_preconfirm_receipt_request,
)
from mo_nco.pareto_smc_spec import (
    analytic_objective_box,
    load_pareto_smc_specification,
    original_unit_cell_widths,
)
from mo_nco.pareto_sparse_compression_certificate import (
    certify_sparse_finite_reference_compression,
)
from mo_nco.pareto_sparse_reference import (
    greedy_maximal_reference_net,
)
from mo_nco.pareto_v13_publication import (
    V13PublicationProtocolError,
    load_v13_pilot_artifact,
    run_v13_confirm_from_signed_receipt,
    run_v13_pilot_freeze,
    write_v13_pilot_artifact,
)
from mo_nco.pareto_v13_spec import (
    V13_ALGORITHM_ID,
    V13_PROTOCOL_SPECIFICATION_SCHEMA,
    V13_RECEIPT_POLICY,
    load_v13_protocol_specification,
)


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class V13PublicationProtocolTests(unittest.TestCase):
    def test_signed_freeze_authorizes_confirm_and_sparse_witness_packet(
        self,
    ) -> None:
        instance = MultiObjectiveTSPInstance.random_biobjective(4, seed=2)
        all_tours = tuple(
            (0,) + tail
            for tail in itertools.permutations(range(1, 4))
        )
        all_objectives = tuple(
            instance.evaluate(tour) for tour in all_tours
        )
        full_reference = nondominated_points(all_objectives)
        witness_by_objective = {
            objective: tour
            for tour, objective in zip(all_tours, all_objectives)
        }
        private_key = Ed25519PrivateKey.generate()
        private_raw = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_raw = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        attacker_private_key = Ed25519PrivateKey.generate()
        attacker_private_raw = attacker_private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        attacker_public_raw = (
            attacker_private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smc_payload = {
                "schema": "annealed_pareto_smc_spec_v1",
                "objective_box": {
                    "source": "analytic_distance_matrix_box",
                    "archive_independent": True,
                },
                "epsilon_cells": {
                    "coordinate_system": (
                        "normalized_frozen_objective_box"
                    ),
                    "widths": [0.25, 0.25],
                    "archive_independent": True,
                    "role": "reporting_and_coverage_only",
                },
                "reference_directions": [[0.5, 0.5]],
                "target": {
                    "family": "typed_augmented_tchebycheff_gibbs",
                    "stage_frozen": True,
                    "beta_schedule": [0.0, 0.01],
                    "chebyshev_rho": 0.03,
                },
                "resampling": {
                    "method": "multinomial",
                    "scope": "within_reference_type",
                    "ess_threshold_fraction": 0.9,
                    "ess_is_not_a_coverage_certificate": True,
                },
                "mutation": {
                    "proposal": (
                        "local_two_opt_plus_uniform_global_refresh"
                    ),
                    "global_refresh_probability": 1.0,
                    "steps_per_stage": [8],
                    "acceptance": "exact_log_domain_mh",
                    "objective_evaluation": "full_tour",
                },
                "particle_allocation": {
                    "policy": (
                        "split_cli_population_equally_across_reference_types"
                    )
                },
                "reporting": {
                    "archive_role": "reporting_only",
                    "archive_max_size": None,
                    "cell_ledger": (
                        "untruncated_first_evaluated_representative_per_cell"
                    ),
                },
            }
            smc_path = root / "smc.json"
            smc_path.write_text(
                json.dumps(smc_payload),
                encoding="utf-8",
            )
            smc = load_pareto_smc_specification(
                smc_path,
                objective_dimension=2,
            )
            widths = original_unit_cell_widths(instance, smc)
            _, upper = analytic_objective_box(instance)
            hv_reference = tuple(value + 1.0 for value in upper)
            cover = greedy_maximal_reference_net(
                full_reference,
                cover_radius=1e-12,
                p_norm=1.0,
            )
            sparse = certify_sparse_finite_reference_compression(
                full_reference,
                cover,
                cell_width_vector=widths,
                igd_p=1.0,
                hv_reference=hv_reference,
            )
            anchor_reference = tuple(
                sorted(sparse.anchor_reference_set)
            )
            self.assertLess(
                len(anchor_reference),
                len(full_reference),
            )

            pilot_seed = derive_domain_separated_seed(
                case_identity="case-02",
                instance_sha256=instance_sha256(instance),
                paired_seed=0,
                algorithm_role=PARETO_SMC_V13_ALGORITHM_ROLE,
                algorithm_version=PARETO_SMC_V13_ALGORITHM_VERSION,
                stream_role="pilot",
            ).seed
            confirm_seed = derive_domain_separated_seed(
                case_identity="case-02",
                instance_sha256=instance_sha256(instance),
                paired_seed=0,
                algorithm_role=PARETO_SMC_V13_ALGORITHM_ROLE,
                algorithm_version=PARETO_SMC_V13_ALGORITHM_VERSION,
                stream_role="confirm",
            ).seed

            def write_reference_spec(
                filename: str,
                reference: tuple[tuple[float, ...], ...],
                *,
                declared_pilot_seed: int = pilot_seed,
                declared_confirm_seed: int = confirm_seed,
            ) -> Path:
                witnesses = tuple(
                    {
                        "tour": witness_by_objective[objective],
                        "objectives": objective,
                    }
                    for objective in sorted(reference)
                )
                payload = {
                    "schema": (
                        "pareto_smc_fixed_reference_certificate_spec_v2"
                    ),
                    "instance_sha256": instance_sha256(instance),
                    "pareto_smc_specification_sha256": smc.sha256,
                    "reference_front": {
                        "source": "independent_exact_solver",
                        "artifact_sha256": _canonical_sha256(
                            tuple(sorted(reference))
                        ),
                        "objectives": tuple(sorted(reference)),
                        "witnesses": witnesses,
                        "witness_payload_sha256": _canonical_sha256(
                            tuple(
                                sorted(
                                    (
                                        {
                                            "tour": tuple(row["tour"]),
                                            "objectives": tuple(
                                                row["objectives"]
                                            ),
                                        }
                                        for row in witnesses
                                    ),
                                    key=lambda row: (
                                        row["objectives"],
                                        row["tour"],
                                    ),
                                )
                            )
                        ),
                    },
                    "streams": {
                        "pilot_seed": declared_pilot_seed,
                        "confirm_seed": declared_confirm_seed,
                        "independence_model": (
                            "ideal_product_random_streams"
                        ),
                    },
                    "failure_budgets": {
                        "pilot": 0.025,
                        "confirm": 0.025,
                    },
                    "metrics": {
                        "igd_p": 1.0,
                        "hv_reference": hv_reference,
                        "max_igd_bound": (
                            sparse.ordinary_igd_bound * 1.01
                        ),
                        "max_hv_deficit_bound": (
                            sparse.shifted_front_hv_deficit_bound * 1.01
                        ),
                    },
                    "certified_archive": {
                        "policy": (
                            "deterministic_reference_coverage_v1"
                        ),
                        "max_size": len(reference),
                    },
                }
                path = root / filename
                path.write_text(json.dumps(payload), encoding="utf-8")
                return path

            anchor_path = write_reference_spec(
                "anchor.json",
                anchor_reference,
            )
            full_path = write_reference_spec(
                "full.json",
                tuple(full_reference),
            )
            anchor_spec = (
                load_fixed_reference_certificate_specification(
                    anchor_path,
                    objective_dimension=2,
                    instance=instance,
                )
            )
            full_spec = (
                load_fixed_reference_certificate_specification(
                    full_path,
                    objective_dimension=2,
                    instance=instance,
                )
            )
            sparse_sha256 = _canonical_sha256(sparse.to_jsonable())
            particles_per_reference = 512
            per_stream = particles_per_reference * 9
            pilot_pair_budget = (
                anchor_spec.pilot_failure_budget
                / len(anchor_reference)
            )
            protocol_payload = {
                "schema": V13_PROTOCOL_SPECIFICATION_SCHEMA,
                "bindings": {
                    "instance_sha256": instance_sha256(instance),
                    "pareto_smc_specification_sha256": smc.sha256,
                    "anchor_certificate_specification_sha256": (
                        anchor_spec.sha256
                    ),
                    "full_reference_certificate_specification_sha256": (
                        full_spec.sha256
                    ),
                    "sparse_compression_certificate_sha256": (
                        sparse_sha256
                    ),
                },
                "identity": {
                    "run_id": "run-02",
                    "case_id": "case-02",
                    "algorithm_id": V13_ALGORITHM_ID,
                },
                "receipt_authority": {
                    "signer_key_id": "offline-key-02",
                    "ed25519_public_key_hex": public_raw.hex(),
                    "authorization_policy": V13_RECEIPT_POLICY,
                },
                "seed_derivation": {
                    "schema": DOMAIN_SEPARATED_SEED_SCHEMA_V1,
                },
                "full_sweep_checkpoints": {
                    "evaluation_counts": [per_stream],
                },
                "assignment_preflight": {
                    "desired_target_mass_lower_bounds_by_anchor_cell": [
                        0.05
                        for _ in anchor_reference
                    ],
                    "pilot_failure_budgets_by_anchor_cell": [
                        pilot_pair_budget
                        for _ in anchor_reference
                    ],
                    "confirm_failure_budgets_by_anchor_cell": [
                        anchor_spec.confirm_failure_budget
                        / len(anchor_reference)
                        for _ in anchor_reference
                    ],
                    "mutually_exclusive_anchor_cells": True,
                },
            }
            protocol_path = root / "protocol.json"
            protocol_path.write_text(
                json.dumps(protocol_payload),
                encoding="utf-8",
            )
            protocol = load_v13_protocol_specification(protocol_path)

            conflicting_full_path = write_reference_spec(
                "full-conflicting-streams.json",
                tuple(full_reference),
                declared_pilot_seed=pilot_seed + 1,
            )
            conflicting_full_spec = (
                load_fixed_reference_certificate_specification(
                    conflicting_full_path,
                    objective_dimension=2,
                    instance=instance,
                )
            )
            conflicting_protocol_payload = json.loads(
                json.dumps(protocol_payload)
            )
            conflicting_protocol_payload["bindings"][
                "full_reference_certificate_specification_sha256"
            ] = conflicting_full_spec.sha256
            conflicting_protocol_path = (
                root / "protocol-conflicting-streams.json"
            )
            conflicting_protocol_path.write_text(
                json.dumps(conflicting_protocol_payload),
                encoding="utf-8",
            )
            conflicting_protocol = load_v13_protocol_specification(
                conflicting_protocol_path
            )
            with self.assertRaisesRegex(
                V13PublicationProtocolError,
                "stream declaration differs",
            ):
                run_v13_pilot_freeze(
                    instance,
                    pareto_smc_specification=smc,
                    anchor_certificate_specification=anchor_spec,
                    full_reference_specification=conflicting_full_spec,
                    protocol_specification=conflicting_protocol,
                    sparse_cover=cover,
                    sparse_compression_certificate=sparse,
                    particles_per_reference=particles_per_reference,
                )
            with self.assertRaisesRegex(
                V13PublicationProtocolError,
                "not exactly backed",
            ):
                run_v13_pilot_freeze(
                    instance,
                    pareto_smc_specification=smc,
                    anchor_certificate_specification=anchor_spec,
                    full_reference_specification=replace(
                        full_spec,
                        max_igd_bound=1e100,
                    ),
                    protocol_specification=protocol,
                    sparse_cover=cover,
                    sparse_compression_certificate=sparse,
                    particles_per_reference=particles_per_reference,
                )
            with self.assertRaisesRegex(
                V13PublicationProtocolError,
                "not exactly backed",
            ):
                run_v13_pilot_freeze(
                    instance,
                    pareto_smc_specification=replace(
                        smc,
                        global_refresh_probability=0.5,
                    ),
                    anchor_certificate_specification=anchor_spec,
                    full_reference_specification=full_spec,
                    protocol_specification=protocol,
                    sparse_cover=cover,
                    sparse_compression_certificate=sparse,
                    particles_per_reference=particles_per_reference,
                )
            forged_protocol = replace(
                protocol,
                signer_public_key_raw=attacker_public_raw,
            )
            with self.assertRaisesRegex(
                V13PublicationProtocolError,
                "not exactly backed",
            ):
                run_v13_pilot_freeze(
                    instance,
                    pareto_smc_specification=smc,
                    anchor_certificate_specification=anchor_spec,
                    full_reference_specification=full_spec,
                    protocol_specification=forged_protocol,
                    sparse_cover=cover,
                    sparse_compression_certificate=sparse,
                    particles_per_reference=particles_per_reference,
                )
            freeze = run_v13_pilot_freeze(
                instance,
                pareto_smc_specification=smc,
                anchor_certificate_specification=anchor_spec,
                full_reference_specification=full_spec,
                protocol_specification=protocol,
                sparse_cover=cover,
                sparse_compression_certificate=sparse,
                particles_per_reference=particles_per_reference,
            )
            pilot_artifact_path = root / "pilot-artifact.json"
            pilot_artifact_sha256 = write_v13_pilot_artifact(
                freeze,
                pilot_artifact_path,
            )
            self.assertEqual(
                pilot_artifact_sha256,
                hashlib.sha256(
                    pilot_artifact_path.read_bytes()
                ).hexdigest(),
            )
            with self.assertRaisesRegex(
                V13PublicationProtocolError,
                "not exactly backed",
            ):
                load_v13_pilot_artifact(
                    pilot_artifact_path,
                    instance=instance,
                    pareto_smc_specification=smc,
                    anchor_certificate_specification=anchor_spec,
                    full_reference_specification=full_spec,
                    protocol_specification=forged_protocol,
                    sparse_cover=cover,
                    sparse_compression_certificate=sparse,
                )
            reloaded_freeze = load_v13_pilot_artifact(
                pilot_artifact_path,
                instance=instance,
                pareto_smc_specification=smc,
                anchor_certificate_specification=anchor_spec,
                full_reference_specification=full_spec,
                protocol_specification=protocol,
                sparse_cover=cover,
                sparse_compression_certificate=sparse,
            )
            self.assertEqual(
                reloaded_freeze.freeze_envelope_sha256,
                freeze.freeze_envelope_sha256,
            )
            tampered_artifact = root / "tampered-pilot-artifact.json"
            tampered_artifact.write_bytes(
                pilot_artifact_path.read_bytes() + b"\n"
            )
            with self.assertRaises(V13PublicationProtocolError):
                load_v13_pilot_artifact(
                    tampered_artifact,
                    instance=instance,
                    pareto_smc_specification=smc,
                    anchor_certificate_specification=anchor_spec,
                    full_reference_specification=full_spec,
                    protocol_specification=protocol,
                    sparse_cover=cover,
                    sparse_compression_certificate=sparse,
                )
            duplicated_artifact = root / "duplicate-pilot-artifact.json"
            duplicated_artifact.write_bytes(
                (
                    b'{"schema":"pareto_smc_v13_pilot_artifact_v2",'
                    + pilot_artifact_path.read_bytes()[1:]
                )
            )
            with self.assertRaisesRegex(
                V13PublicationProtocolError,
                "strict UTF-8 JSON",
            ):
                load_v13_pilot_artifact(
                    duplicated_artifact,
                    instance=instance,
                    pareto_smc_specification=smc,
                    anchor_certificate_specification=anchor_spec,
                    full_reference_specification=full_spec,
                    protocol_specification=protocol,
                    sparse_cover=cover,
                    sparse_compression_certificate=sparse,
                )
            unsigned = create_unsigned_preconfirm_receipt_request(
                bindings=reloaded_freeze.receipt_bindings,
                signer_key_id=protocol.signer_key_id,
                frozen_signer_public_key_raw=public_raw,
                issued_at_utc="2026-07-28T12:34:56Z",
            )
            signed = sign_preconfirm_receipt_request(
                unsigned,
                private_key_raw=private_raw,
            )
            forged_unsigned = (
                create_unsigned_preconfirm_receipt_request(
                    bindings=reloaded_freeze.receipt_bindings,
                    signer_key_id=protocol.signer_key_id,
                    frozen_signer_public_key_raw=attacker_public_raw,
                    issued_at_utc="2026-07-28T12:34:56Z",
                )
            )
            forged_signed = sign_preconfirm_receipt_request(
                forged_unsigned,
                private_key_raw=attacker_private_raw,
            )
            with self.assertRaisesRegex(
                V13PublicationProtocolError,
                "not exactly backed",
            ):
                run_v13_confirm_from_signed_receipt(
                    replace(
                        reloaded_freeze,
                        protocol_specification=forged_protocol,
                    ),
                    signed_receipt=forged_signed,
                    external_signer_key_not_held_by_runner=True,
                )
            unauthorized_envelope = {
                "schema": "unauthorized_replacement",
                "payload": "not_signed",
            }
            with self.assertRaises(V13PublicationProtocolError):
                run_v13_confirm_from_signed_receipt(
                    replace(
                        reloaded_freeze,
                        freeze_envelope=unauthorized_envelope,
                        freeze_envelope_sha256=_canonical_sha256(
                            unauthorized_envelope
                        ),
                    ),
                    signed_receipt=signed,
                    external_signer_key_not_held_by_runner=True,
                )
            tampered_metadata = dict(reloaded_freeze.pilot.metadata)
            tampered_metadata["accepted_mutations"] = (
                int(tampered_metadata["accepted_mutations"]) + 123456
            )
            with self.assertRaises(V13PublicationProtocolError):
                run_v13_confirm_from_signed_receipt(
                    replace(
                        reloaded_freeze,
                        pilot=replace(
                            reloaded_freeze.pilot,
                            metadata=tampered_metadata,
                        ),
                    ),
                    signed_receipt=signed,
                    external_signer_key_not_held_by_runner=True,
                )
            tampered_diagnostics = (
                replace(
                    reloaded_freeze.pilot.diagnostics[0],
                    acceptance_rate=0.123456789,
                ),
                *reloaded_freeze.pilot.diagnostics[1:],
            )
            with self.assertRaises(V13PublicationProtocolError):
                run_v13_confirm_from_signed_receipt(
                    replace(
                        reloaded_freeze,
                        pilot=replace(
                            reloaded_freeze.pilot,
                            diagnostics=tampered_diagnostics,
                        ),
                    ),
                    signed_receipt=signed,
                    external_signer_key_not_held_by_runner=True,
                )
            with self.assertRaises(
                PreconfirmReceiptVerificationError
            ):
                run_v13_confirm_from_signed_receipt(
                    reloaded_freeze,
                    signed_receipt=signed,
                    external_signer_key_not_held_by_runner=False,
                )
            result = run_v13_confirm_from_signed_receipt(
                reloaded_freeze,
                signed_receipt=signed,
                external_signer_key_not_held_by_runner=True,
            )

        self.assertEqual(
            result.certificate["v13_formal_packet_gate"],
            "NOT_ESTABLISHED",
        )
        self.assertEqual(
            result.certificate[
                "v13_conditional_certificate_content_gate"
            ],
            "PASS",
        )
        self.assertEqual(
            result.certificate["formal_packet_gate"],
            "FAIL",
        )
        self.assertEqual(
            result.certificate[
                "publication_certificate_packet_gate"
            ],
            "NOT_ESTABLISHED",
        )
        self.assertEqual(
            result.certificate[
                "external_preconfirm_commitment_receipt_gate"
            ],
            "CONDITIONAL_ON_DECLARED_KEY_SEPARATION_AND_NO_PREVIEW",
        )
        self.assertEqual(
            result.certificate[
                "cryptographic_preconfirm_receipt_verification_gate"
            ],
            "PASS",
        )
        self.assertEqual(
            result.certificate["full_reference_feasibility_gate"],
            "PASS",
        )
        self.assertEqual(
            result.certificate["sparse_full_reference_metric_gate"],
            "PASS",
        )
        self.assertEqual(
            result.certificate["algorithm_competitiveness_gate"],
            "NOT_RUN",
        )
        self.assertEqual(
            result.certificate[
                "global_refresh_certificate_sufficiency_gate"
            ],
            "PASS",
        )
        self.assertEqual(
            result.certificate[
                "global_refresh_optimizer_quality_joint_theorem_gate"
            ],
            "NOT_ESTABLISHED",
        )
        self.assertEqual(
            result.certificate[
                "independent_wall_clock_timing_proof_gate"
            ],
            "NOT_ESTABLISHED",
        )
        self.assertFalse(
            result.certificate[
                "reference_count_independent_universal_compression_claimed"
            ]
        )
        self.assertFalse(
            result.certificate[
                "unconditional_preconfirm_false_selection_control_claimed"
            ]
        )


if __name__ == "__main__":
    unittest.main()

