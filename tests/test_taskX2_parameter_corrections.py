"""Task X2 — physics-team wavelength and lab-C focal regression tests."""
from __future__ import annotations

import unittest

from setup_params import (
    CANONICAL_LAMBDA_I_NM,
    CANONICAL_LAMBDA_P_NM,
    CANONICAL_LAMBDA_S_NM,
    ENERGY_EXACT_LAMBDA_P_FROM_810_1550_NM,
    qiup_magnification,
    resolve_architecture_profile,
    resolve_setup_selection,
    resolved_far_field_focal_mm,
    resolved_spatial_mapping,
    validate_spdc_energy_conservation,
)
from verification.lab_config_c.definitions import LAB_PROFILE_ID, STOCK_C_PROFILE


NOMINAL = (532.0, 810.0, 1550.0)
PUBLICATION_PROFILES = ("A", "B", "C", "D", LAB_PROFILE_ID)


class TestNominalWavelengths(unittest.TestCase):
    def test_energy_exact_constant_preserved_and_distinct(self):
        exact = float(ENERGY_EXACT_LAMBDA_P_FROM_810_1550_NM)
        self.assertAlmostEqual(exact, 531.9915254237288, places=10)
        self.assertNotAlmostEqual(exact, CANONICAL_LAMBDA_P_NM, places=6)
        # Exact value zeros residual by construction.
        inv = 1.0 / exact
        residual = (inv - 1.0 / 810.0 - 1.0 / 1550.0) / inv
        self.assertAlmostEqual(residual, 0.0, places=14)

    def test_publication_profiles_use_nominal_532_810_1550(self):
        for name in PUBLICATION_PROFILES:
            with self.subTest(profile=name):
                setup = resolve_setup_selection(name)
                self.assertEqual(setup.lambda_p_nm, NOMINAL[0])
                self.assertEqual(setup.lambda_s_nm, NOMINAL[1])
                self.assertEqual(setup.lambda_i_nm, NOMINAL[2])
                self.assertEqual(
                    (CANONICAL_LAMBDA_P_NM, CANONICAL_LAMBDA_S_NM, CANONICAL_LAMBDA_I_NM),
                    NOMINAL,
                )

    def test_nominal_u01_residual_nonzero_but_within_tolerance(self):
        setup = resolve_setup_selection(LAB_PROFILE_ID)
        residual = abs(validate_spdc_energy_conservation(setup))
        self.assertGreater(residual, 1.0e-6)
        self.assertLess(residual, 1.0e-4)
        self.assertAlmostEqual(residual, 1.593e-5, places=6)


class TestLabMatchedCFocals(unittest.TestCase):
    def test_lab_c_lenses_map_to_150_mm_by_role(self):
        lab = resolve_setup_selection(LAB_PROFILE_ID)
        # Idler-path / undetected-arm Fourier lens
        self.assertEqual(float(lab.far_field_focal_mm), 150.0)
        self.assertEqual(float(resolved_far_field_focal_mm(lab)), 150.0)
        # Synced M_QIUP denominator
        self.assertEqual(float(lab.F1_mm), 150.0)
        # Signal-path + camera/front (aliased f_c)
        self.assertEqual(float(lab.F2_mm), 150.0)
        self.assertEqual(lab.far_field_focal_mm, lab.F1_mm)

    def test_stock_c_remains_lemos_75_150_distinct_from_lab(self):
        stock = resolve_setup_selection("C")
        lab = resolve_setup_selection(LAB_PROFILE_ID)
        self.assertEqual(STOCK_C_PROFILE, "single_crystal_far_field")
        self.assertEqual(stock.setup_id, "single_crystal_far_field_v2")
        self.assertEqual(lab.setup_id, LAB_PROFILE_ID)
        self.assertEqual(float(stock.far_field_focal_mm), 75.0)
        self.assertEqual(float(stock.F1_mm), 75.0)
        self.assertEqual(float(stock.F2_mm), 150.0)
        self.assertNotEqual(float(stock.far_field_focal_mm), float(lab.far_field_focal_mm))
        self.assertNotEqual(float(stock.F1_mm), float(lab.F1_mm))
        # Same topology family
        self.assertEqual(stock.spdc_architecture, lab.spdc_architecture)
        self.assertEqual(stock.imaging_mode, lab.imaging_mode)
        self.assertEqual(
            resolve_architecture_profile(stock).profile_id,
            resolve_architecture_profile(lab).profile_id,
        )

    def test_units_remain_mm_and_nm(self):
        lab = resolve_setup_selection(LAB_PROFILE_ID)
        for attr in ("far_field_focal_mm", "F1_mm", "F2_mm"):
            val = float(getattr(lab, attr))
            self.assertTrue(50.0 < val < 500.0, msg=f"{attr}={val} not in mm-scale")
        for attr, lo, hi in (
            ("lambda_p_nm", 500.0, 560.0),
            ("lambda_s_nm", 780.0, 850.0),
            ("lambda_i_nm", 1400.0, 1700.0),
        ):
            val = float(getattr(lab, attr))
            self.assertTrue(lo < val < hi, msg=f"{attr}={val} not in nm-scale")

    def test_lab_c_magnification_matches_equal_focal_formula(self):
        lab = resolve_setup_selection(LAB_PROFILE_ID)
        mapping = resolved_spatial_mapping(lab)
        signed_m = float(mapping["signed_object_to_camera_magnification"])
        expected = -(
            float(lab.far_field_signal_relay_magnification)
            * float(lab.lambda_s_nm)
            * float(lab.F2_mm)
            / (
                float(lab.far_field_idler_relay_magnification)
                * float(lab.lambda_i_nm)
                * float(resolved_far_field_focal_mm(lab))
            )
        )
        self.assertAlmostEqual(signed_m, expected, places=12)
        expected_m = -(
            float(lab.far_field_signal_relay_magnification)
            * float(lab.lambda_s_nm)
            * float(lab.F2_mm)
            / (
                float(lab.far_field_idler_relay_magnification)
                * float(lab.lambda_i_nm)
                * float(resolved_far_field_focal_mm(lab))
            )
        )
        self.assertAlmostEqual(signed_m, expected_m, places=12)
        self.assertAlmostEqual(float(lab.far_field_idler_relay_magnification), 2.0, places=12)
        # M_QIUP magnitude with synced F1=F2 focals (Mi=2.0 final C_lab freeze)
        m_qiup = qiup_magnification(
            lab, lambda_s_nm=lab.lambda_s_nm, lambda_i_nm=lab.lambda_i_nm
        )
        self.assertAlmostEqual(m_qiup, 810.0 / 1550.0, places=12)
        self.assertAlmostEqual(abs(signed_m), abs(expected_m), places=12)

    def test_propagation_topology_unchanged(self):
        lab = resolve_setup_selection(LAB_PROFILE_ID)
        stock = resolve_setup_selection("C")
        for setup in (lab, stock):
            self.assertEqual(setup.spdc_architecture, "single_crystal_double_pass")
            self.assertEqual(setup.imaging_mode, "far_field")
            self.assertEqual(setup.n_spdc_interactions, 2)
            self.assertEqual(int(getattr(setup, "n_crystals", 1)), 1)
            self.assertEqual(
                setup.architecture_propagation_model,
                "architecture_specific_complex_field_v1",
            )
            self.assertEqual(setup.spatial_transfer_kernel, "conditional_gaussian_v1")
            if setup.setup_id == "config_c_lab_v1":
                self.assertEqual(float(setup.far_field_idler_relay_magnification), 2.0)
            else:
                self.assertEqual(float(setup.far_field_idler_relay_magnification), 1.0)
            self.assertEqual(float(setup.far_field_signal_relay_magnification), 1.0)

    def test_no_global_75_to_150_on_stock_or_near_field(self):
        # Near-field B must not suddenly gain a 150 mm far-field focal.
        near = resolve_setup_selection("B")
        self.assertIsNone(near.far_field_focal_mm)
        stock = resolve_setup_selection("C")
        self.assertEqual(float(stock.far_field_focal_mm), 75.0)


if __name__ == "__main__":
    unittest.main()
