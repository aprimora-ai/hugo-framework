"""
HUGO Framework — Unit Tests
David Ohio | Independent Researcher | odavidohio@gmail.com

33 testes cobrindo: HomeostaticVector, HomeostaticField (v3.0 ensemble),
StructuralAttentionNetwork, KappaMonitor, ensemble calibration,
set_point differentiation, feedback loop verification.

Executar: python tests/test_hugo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import unittest

from src.homeostasis.homeostatic_field import HomeostaticField, HomeostaticVector
from src.network.gray_box_network import StructuralAttentionNetwork


class TestHomeostaticVector(unittest.TestCase):

    def _make(self, value=0.6):
        return HomeostaticVector(
            name="test", biological_analog="test",
            value=value, nominal_min=0.4, nominal_max=0.8
        )

    def test_nominal_center(self):
        self.assertAlmostEqual(self._make().nominal_center, 0.6)

    def test_deviation_within_range_is_zero(self):
        self.assertAlmostEqual(self._make(0.6).deviation, 0.0)

    def test_deviation_below_range(self):
        self.assertAlmostEqual(self._make(0.2).deviation, 0.2)

    def test_deviation_above_range(self):
        self.assertAlmostEqual(self._make(0.95).deviation, 0.15)

    def test_normalized_state_at_center(self):
        self.assertAlmostEqual(self._make(0.6).normalized_state, 0.0, places=5)

    def test_set_point_equals_initial_value(self):
        v = self._make(0.35)
        self.assertAlmostEqual(v._set_point, 0.35)

    def test_update_stays_bounded(self):
        v = self._make(0.5)
        for _ in range(1000):
            v.update(np.random.randn(64))
        self.assertTrue(0.0 <= v.value <= 1.0)


class TestHomeostaticField(unittest.TestCase):

    def test_initial_state_shape(self):
        self.assertEqual(HomeostaticField().state.shape, (5,))

    def test_initial_state_values(self):
        hf = HomeostaticField()
        np.testing.assert_array_almost_equal(
            hf.state, [0.60, 0.70, 0.50, 0.60, 0.675])

    def test_deviations_shape(self):
        self.assertEqual(HomeostaticField().deviations.shape, (5,))

    def test_temperature_at_equilibrium(self):
        hf = HomeostaticField()
        self.assertGreater(hf.temperature, 0.8)
        self.assertLessEqual(hf.temperature, 1.0)

    def test_temperature_under_stress(self):
        """H2=0.10 -> dev=0.40, L=0.12, tau = 1/(1+0.48) = 0.676."""
        stressed = np.array([0.60, 0.10, 0.50, 0.60, 0.675])
        hf = HomeostaticField(initial_state=stressed)
        self.assertLess(hf.temperature, 0.70)
        self.assertGreater(hf.temperature, 0.65)

    def test_temperature_never_below_minimum(self):
        extreme = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
        hf = HomeostaticField(initial_state=extreme)
        self.assertGreaterEqual(hf.temperature, HomeostaticField.TAU_MIN)

    def test_L_is_nonnegative(self):
        self.assertGreaterEqual(HomeostaticField().L, 0.0)

    def test_metric_deformation_shape(self):
        G = HomeostaticField().metric_deformation_matrix(seq_len=32)
        self.assertEqual(G.shape, (32, 32))

    def test_metric_deformation_symmetric(self):
        G = HomeostaticField().metric_deformation_matrix(seq_len=16)
        np.testing.assert_array_almost_equal(G, G.T, decimal=10)

    def test_update_records_history(self):
        hf = HomeostaticField()
        for _ in range(10):
            hf.update(np.random.randn(128))
        self.assertEqual(len(hf.history), 10)
        self.assertEqual(len(hf.temperature_history), 10)

    def test_emotional_label_is_valid(self):
        hf = HomeostaticField()
        self.assertIn(hf.emotional_label(), [
            "FEAR_ANALOG", "FEAR_SEEK_TRANSITION", "RAGE_ANALOG",
            "PANIC_ANALOG", "SEEK_ANALOG", "TRANSITIONAL"
        ])

    def test_set_points_preserved(self):
        """Set points devem ser imutaveis apos inicializacao."""
        h = np.array([0.30, 0.20, 0.80, 0.30, 0.90])
        hf = HomeostaticField(initial_state=h)
        np.testing.assert_array_almost_equal(hf.set_points, h)
        for _ in range(100):
            hf.update(np.random.randn(128))
        np.testing.assert_array_almost_equal(hf.set_points, h)

    def test_ensemble_calibration_runs(self):
        """Apos BURN_IN_STEPS, o campo deve estar calibrado."""
        hf = HomeostaticField()
        self.assertFalse(hf.is_calibrated)
        for t in range(HomeostaticField.BURN_IN_STEPS + 5):
            hf.update(np.random.randn(128))
        self.assertTrue(hf.is_calibrated)

    def test_calibrated_decays_are_positive(self):
        hf = HomeostaticField()
        for t in range(50):
            hf.update(np.random.randn(128))
        for v in hf.vectors:
            self.assertGreater(v.decay_rate, 0.0)
            self.assertLessEqual(v.decay_rate, 0.1)

    def test_differentiation_persists_200_steps(self):
        """Configs distintas devem produzir tau distintos apos 200 steps."""
        configs = {
            "BASELINE":  [0.60, 0.70, 0.50, 0.60, 0.675],
            "H2_LOW":    [0.60, 0.25, 0.50, 0.60, 0.675],
            "H1_LOW":    [0.15, 0.70, 0.50, 0.60, 0.675],
        }
        taus = {}
        for name, h in configs.items():
            np.random.seed(42)
            hf = HomeostaticField(initial_state=np.array(h))
            for t in range(200):
                hf.update(np.random.randn(128))
            taus[name] = round(hf.temperature, 3)
        self.assertGreaterEqual(len(set(taus.values())), 2,
            f"Configs convergiram para o mesmo tau: {taus}")

    def test_value_oscillates_near_set_point(self):
        """Apos calibracao, value deve ficar proximo do set_point."""
        hf = HomeostaticField(initial_state=np.array([0.30, 0.70, 0.50, 0.60, 0.675]))
        np.random.seed(42)
        for t in range(200):
            hf.update(np.random.randn(128))
        h1_val = hf.vectors[0].value
        h1_sp = hf.vectors[0]._set_point
        self.assertAlmostEqual(h1_val, h1_sp, delta=0.15,
            msg=f"H1 value={h1_val:.3f} too far from set_point={h1_sp:.3f}")


class TestStructuralAttentionNetwork(unittest.TestCase):

    def setUp(self):
        self.net = StructuralAttentionNetwork(
            input_dim=64, seq_len=16, hidden_dim=64,
            n_heads=4, n_layers=2, seed=42
        )

    def test_forward_returns_required_keys(self):
        result = self.net.forward(np.random.randn(16, 64))
        for key in ["A_final", "scores_final", "r_final", "tau",
                     "H_state", "deviations", "L_t", "emotional_label", "step"]:
            self.assertIn(key, result)

    def test_attention_matrix_shape(self):
        result = self.net.forward(np.random.randn(16, 64))
        self.assertEqual(result["A_final"].shape, (16, 16))

    def test_attention_rows_sum_to_one(self):
        result = self.net.forward(np.random.randn(16, 64))
        row_sums = result["A_final"].sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(16), decimal=5)

    def test_representation_shape(self):
        result = self.net.forward(np.random.randn(16, 64))
        self.assertEqual(result["r_final"].shape, (64,))

    def test_different_H_produce_different_attention(self):
        """Configs homeostaticas distintas produzem A(t) distintas.

        O feedback H->tau->A->r->H precisa de alguns steps para acumular
        diferenca mensuravel na atenção, porque o softmax suaviza
        diferencas pequenas em G(H(t)).
        """
        net_calm = StructuralAttentionNetwork(
            input_dim=64, seq_len=16, hidden_dim=64, n_heads=4, n_layers=2,
            initial_homeostasis=np.array([0.60, 0.70, 0.50, 0.60, 0.675]),
            seed=42)
        net_stress = StructuralAttentionNetwork(
            input_dim=64, seq_len=16, hidden_dim=64, n_heads=4, n_layers=2,
            initial_homeostasis=np.array([0.60, 0.10, 0.50, 0.60, 0.675]),
            seed=42)
        # Roda 5 steps com mesmo input para acumular feedback
        for t in range(5):
            x = np.random.RandomState(seed=t).randn(16, 64)
            r_calm = net_calm.forward(x)
            r_stress = net_stress.forward(x)
        self.assertFalse(
            np.allclose(r_calm["A_final"], r_stress["A_final"], atol=1e-4),
            "Different H states should produce different attention after feedback")
        # Temperaturas devem ser distintas
        self.assertNotAlmostEqual(r_calm["tau"], r_stress["tau"], places=1)

    def test_no_nan_in_100_steps(self):
        for t in range(100):
            result = self.net.forward(np.random.randn(16, 64))
            self.assertFalse(np.any(np.isnan(result["A_final"])))
            self.assertFalse(np.any(np.isnan(result["r_final"])))
            self.assertFalse(np.isnan(result["tau"]))

    def test_step_counter(self):
        for _ in range(5):
            result = self.net.forward(np.random.randn(16, 64))
        self.assertEqual(result["step"], 5)


class TestKappaMonitor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from src.kappa.kappa_monitor import KappaMonitor
            cls.KappaMonitor = KappaMonitor
            cls.available = True
        except ImportError:
            cls.available = False

    def setUp(self):
        if not self.available:
            self.skipTest("ripser/persim not installed")
        self.kappa = self.KappaMonitor(max_dim=1, threshold=1.5)

    def _random_inputs(self):
        A = np.random.rand(16, 16)
        A = A / A.sum(axis=1, keepdims=True)
        return A, np.random.randn(16, 16), np.random.randn(64), \
               np.array([0.6, 0.7, 0.5, 0.6, 0.675])

    def test_process_returns_required_keys(self):
        A, scores, r, H = self._random_inputs()
        result = self.kappa.process(A, scores, r, H, L_t=0.01, tau=0.95)
        for key in ["betti_0", "betti_1", "entropy_A", "wass_A",
                     "entropy_S", "wass_S", "tau", "L_t"]:
            self.assertIn(key, result)

    def test_no_nan_in_process(self):
        for _ in range(20):
            A, scores, r, H = self._random_inputs()
            result = self.kappa.process(A, scores, r, H, 0.01, 0.95)
            self.assertFalse(np.isnan(result["entropy_A"]))
            self.assertFalse(np.isnan(result["entropy_S"]))

    def test_time_series_accumulates(self):
        for _ in range(10):
            A, scores, r, H = self._random_inputs()
            self.kappa.process(A, scores, r, H, 0.01, 0.95)
        ts = self.kappa.get_time_series()
        self.assertEqual(ts["n_steps"], 10)
        self.assertEqual(len(ts["entropy_A"]), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
