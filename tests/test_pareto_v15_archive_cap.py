from __future__ import annotations

import unittest
from fractions import Fraction

from mo_nco.pareto_archive_cap_certificate import (
    ArchiveCapCertificateError,
    canonical_gonzalez_cap,
    certify_archive_cap,
)


class ArchiveCapCertificateV15Tests(unittest.TestCase):
    def test_one_plus_half_ulp_cannot_pass_tolerance_one(self) -> None:
        epsilon = Fraction(1, 2**53)
        certificate = certify_archive_cap(
            reference_points=((0,),),
            witnesses=((0,), (epsilon,)),
            reference_to_witness=(1,),
            cap=1,
            p="1",
            ordinary_igd_base_upper=1,
            additive_base_vector=(0,),
            hv_reference=(2,),
            max_ordinary_igd=1,
            max_igd_plus=1,
            max_hv_deficit=1,
        )

        self.assertEqual(
            certificate.ordinary_igd_after_cap_upper,
            f"{2**53 + 1}/{2**53}",
        )
        self.assertFalse(certificate.ordinary_igd_gate)
        self.assertFalse(certificate.passed)

    def test_average_distortion_is_used_for_ordinary_igd(self) -> None:
        certificate = certify_archive_cap(
            reference_points=((0,), (0,), (0,), (0,)),
            witnesses=((0,), (4,)),
            reference_to_witness=(0, 0, 0, 1),
            retained_indices=(0,),
            cap=1,
            p="1",
            ordinary_igd_base_upper=0,
            additive_base_vector=(0,),
            hv_reference=(10,),
            max_ordinary_igd=1,
            max_igd_plus=4,
            max_hv_deficit=4,
        )

        self.assertEqual(certificate.average_cap_distortion, "1")
        self.assertEqual(certificate.worst_cap_radius, "4")
        self.assertTrue(certificate.ordinary_igd_gate)
        self.assertTrue(certificate.passed)

    def test_gonzalez_ties_use_smallest_original_index(self) -> None:
        retained = canonical_gonzalez_cap(
            ((0, 0), (1, 0), (-1, 0), (0, 1)),
            cap=3,
            p="2",
        )
        self.assertEqual(retained, (0, 1, 2))

    def test_directed_coordinate_distortion_is_replayed(self) -> None:
        certificate = certify_archive_cap(
            reference_points=((0, 0),),
            witnesses=((0, 0), (2, -3)),
            reference_to_witness=(1,),
            retained_indices=(0,),
            cap=1,
            p="infinity",
            ordinary_igd_base_upper=0,
            additive_base_vector=(Fraction(1, 2), Fraction(1, 4)),
            hv_reference=(10, 10),
            max_ordinary_igd=3,
            max_igd_plus=3,
            max_hv_deficit=100,
        )

        self.assertEqual(
            certificate.directed_coordinate_cap_radius,
            ("0", "3"),
        )
        self.assertEqual(
            certificate.additive_after_cap_vector,
            ("1/2", "13/4"),
        )
        self.assertFalse(certificate.igd_plus_gate)

    def test_empty_or_invalid_cap_fails_closed(self) -> None:
        with self.assertRaises(ArchiveCapCertificateError):
            canonical_gonzalez_cap((), cap=1, p="2")
        with self.assertRaises(ArchiveCapCertificateError):
            canonical_gonzalez_cap(((0,),), cap=0, p="2")


if __name__ == "__main__":
    unittest.main()

