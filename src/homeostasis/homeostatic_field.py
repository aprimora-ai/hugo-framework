"""
HUGO — Homological Unified Gradient Ontology
Modulo: Campo Homeostatico H(t)

v3.0 — Ensemble-calibrated dynamics (Marco 2026)

Principio central: o decay_rate NAO e um hiperparametro fixo.
E derivado automaticamente das estatisticas de perturbacao
observadas durante um periodo de calibracao (burn-in),
analogo ao CALM do Kappa-FIN.

Pipeline de calibracao:
  1. Burn-in: sistema roda T_burn steps com decay conservador
  2. Coleta: registra |perturbacao_i(t)| para cada vetor i
  3. Ensemble: divide o burn-in em sub-janelas sobrepostas,
     computa decay candidato para cada janela,
     toma a mediana (robusto a transientes)
  4. Congelamento: decay_rate_i calibrado e fixado para o resto da run

Equilibrio: no regime estacionario,
  E[perturbacao] + E[restauracao] = 0
  E[0.02 * tanh(proj . r * sens)] = decay * E[value - set_point]
  => decay = E[|perturbation|] / target_amplitude

O target_amplitude controla quanto o vetor oscila ao redor do set_point.
Valores pequenos -> oscilacao apertada, alta estabilidade.
Valores grandes -> oscilacao larga, mais sensibilidade.

Autor: David Ohio | odavidohio@gmail.com
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class HomeostaticVector:
    name: str
    biological_analog: str
    value: float
    nominal_min: float
    nominal_max: float
    sensitivity: float = 1.0
    _projection: Optional[np.ndarray] = field(default=None, repr=False)
    _set_point: Optional[float] = field(default=None, repr=False)
    # Calibrado pelo ensemble — conservador ate calibracao
    decay_rate: float = 0.005
    # Flag de calibracao
    _calibrated: bool = False
    # Historico de perturbacoes para calibracao
    _perturbation_history: List[float] = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self._set_point is None:
            self._set_point = self.value

    @property
    def nominal_center(self) -> float:
        return (self.nominal_min + self.nominal_max) / 2.0

    @property
    def half_range(self) -> float:
        return (self.nominal_max - self.nominal_min) / 2.0

    @property
    def deviation(self) -> float:
        """Desvio do value em relacao ao NOMINAL RANGE."""
        if self.value < self.nominal_min:
            return self.nominal_min - self.value
        elif self.value > self.nominal_max:
            return self.value - self.nominal_max
        return 0.0

    @property
    def normalized_state(self) -> float:
        return (self.value - self.nominal_center) / (self.half_range + 1e-8)

    def compute_perturbation(self, representation: np.ndarray) -> float:
        """Calcula a perturbacao SEM aplicar (para calibracao)."""
        if self._projection is not None:
            proj_len = min(len(self._projection), len(representation))
            proj_val = np.dot(self._projection[:proj_len],
                              representation[:proj_len])
            influence = np.tanh(proj_val * self.sensitivity)
        else:
            influence = np.tanh(np.var(representation) * self.sensitivity - 0.5)
        return 0.02 * influence

    def update(self, representation: np.ndarray) -> None:
        """
        Dinamica homeostatica com decay calibrado via ensemble.

          perturbacao = 0.02 * tanh(proj . r * sensitivity)
          restauracao = -decay_rate * (value - set_point)

        Nota: perturbation_history so acumula durante burn-in.
        """
        perturbation = self.compute_perturbation(representation)

        # Registra para calibracao (so durante burn-in)
        if not self._calibrated:
            self._perturbation_history.append(abs(perturbation))

        # Restauracao ao set_point
        restoring = -self.decay_rate * (self.value - self._set_point)

        self.value = float(np.clip(
            self.value + perturbation + restoring,
            0.0, 1.0
        ))

    def calibrate_decay(self, target_amplitude: float = 0.08) -> float:
        """
        Calibra decay_rate a partir do historico de perturbacoes.

        No equilibrio:
          E[|perturbation|] = decay * target_amplitude
          => decay = E[|perturbation|] / target_amplitude

        target_amplitude: amplitude desejada de oscilacao ao redor do set_point.
        Retorna o decay calibrado.
        """
        if not self._perturbation_history:
            return self.decay_rate

        mean_pert = float(np.mean(self._perturbation_history))
        if target_amplitude < 1e-8:
            target_amplitude = 0.08

        calibrated = mean_pert / target_amplitude
        # Clamp para faixa razoavel
        calibrated = float(np.clip(calibrated, 0.001, 0.1))
        return calibrated


class HomeostaticField:
    """
    Campo homeostatico H(t) = [H1, H2, H3, H4, H5].

    v3.0: Ensemble-calibrated dynamics.

    Fase de calibracao (burn-in):
      - Roda T_burn steps com decay conservador
      - Divide o historico de perturbacoes em sub-janelas
      - Computa decay candidato por janela
      - Toma mediana (ensemble) -> decay_rate calibrado
      - Congela decay para o resto da run

    Isso replica o principio CALM do Kappa-FIN: o threshold
    nao e um hiperparametro — e derivado dos dados.
    """

    TAU_BASE = 1.0
    TAU_MIN  = 0.05
    ALPHA    = 4.0

    # Parametros de calibracao
    BURN_IN_STEPS = 30            # passos de burn-in
    ENSEMBLE_WINDOWS = 5          # sub-janelas para ensemble
    TARGET_AMPLITUDE = 0.08       # amplitude desejada (half_range ~= 0.15-0.20)

    def __init__(self, initial_state: np.ndarray = None, seed: int = 42,
                 repr_dim: int = 128):
        self.step = 0
        self.history: List[np.ndarray] = []
        self.deviation_history: List[np.ndarray] = []
        self.temperature_history: List[float] = []
        self._G_cache: dict = {}
        self._calibrated = False
        self._repr_dim = repr_dim

        defaults = np.array([0.60, 0.70, 0.50, 0.60, 0.675])
        vals = initial_state if initial_state is not None else defaults

        # Projecoes ortogonais para os 5 canais
        rng = np.random.RandomState(seed=seed + 1000)
        raw_projs = rng.randn(5, repr_dim)
        projs = self._orthogonalize(raw_projs)

        self.vectors = [
            HomeostaticVector("H1_energetic_coherence",
                              "glucose / available energy",
                              float(vals[0]), 0.40, 0.80,
                              sensitivity=0.8,
                              _projection=projs[0]),
            HomeostaticVector("H2_structural_integrity",
                              "physical integrity / immune system",
                              float(vals[1]), 0.50, 0.90,
                              sensitivity=1.2,
                              _projection=projs[1]),
            HomeostaticVector("H3_load_equilibrium",
                              "osmotic pressure / processing balance",
                              float(vals[2]), 0.30, 0.70,
                              sensitivity=1.0,
                              _projection=projs[2]),
            HomeostaticVector("H4_temporal_consistency",
                              "circadian rhythm / contextual coherence",
                              float(vals[3]), 0.40, 0.80,
                              sensitivity=0.6,
                              _projection=projs[3]),
            HomeostaticVector("H5_representational_stability",
                              "core temperature / internal stability",
                              float(vals[4]), 0.50, 0.85,
                              sensitivity=0.9,
                              _projection=projs[4]),
        ]

    @staticmethod
    def _orthogonalize(vecs: np.ndarray) -> np.ndarray:
        n, d = vecs.shape
        ortho = np.zeros_like(vecs)
        for i in range(n):
            v = vecs[i].copy()
            for j in range(i):
                v -= np.dot(v, ortho[j]) * ortho[j]
            norm = np.linalg.norm(v)
            ortho[i] = v / (norm + 1e-10)
        return ortho

    def _run_ensemble_calibration(self):
        """
        Calibracao por ensemble: divide o historico de perturbacoes
        de cada vetor em sub-janelas e toma mediana dos decays candidatos.

        Analogo ao CALM scan do Kappa-FIN:
          CALM: scan → score → top-N → mediana de thresholds
          HUGO: burn-in → sub-janelas → mediana de decay rates
        """
        for v in self.vectors:
            history = v._perturbation_history
            n = len(history)
            if n < 10:
                # Insuficiente para ensemble — usa calibracao simples
                v.decay_rate = v.calibrate_decay(self.TARGET_AMPLITUDE)
                v._calibrated = True
                continue

            # Divide em sub-janelas sobrepostas
            window_size = max(n // self.ENSEMBLE_WINDOWS, 5)
            step_size = max(1, (n - window_size) // max(self.ENSEMBLE_WINDOWS - 1, 1))

            candidates = []
            for start in range(0, n - window_size + 1, step_size):
                window_pert = history[start:start + window_size]
                mean_pert = float(np.mean(window_pert))
                if mean_pert > 1e-8:
                    decay_candidate = mean_pert / self.TARGET_AMPLITUDE
                    candidates.append(decay_candidate)

            if candidates:
                # Mediana — robusto a outliers (mesmo principio do CALM topn)
                calibrated_decay = float(np.median(candidates))
                calibrated_decay = float(np.clip(calibrated_decay, 0.001, 0.1))
                v.decay_rate = calibrated_decay
            else:
                v.decay_rate = v.calibrate_decay(self.TARGET_AMPLITUDE)

            v._calibrated = True
            v._perturbation_history = []  # libera memoria

        self._calibrated = True

    # ── Estado ──────────────────────────────────────────────────────────────

    @property
    def state(self) -> np.ndarray:
        return np.array([v.value for v in self.vectors])

    @property
    def set_points(self) -> np.ndarray:
        return np.array([v._set_point for v in self.vectors])

    @property
    def decay_rates(self) -> np.ndarray:
        return np.array([v.decay_rate for v in self.vectors])

    @property
    def deviations(self) -> np.ndarray:
        return np.array([v.deviation for v in self.vectors])

    @property
    def L(self) -> float:
        weights = np.array([0.15, 0.30, 0.20, 0.15, 0.20])
        return float(np.dot(weights, self.deviations))

    @property
    def normalized_states(self) -> np.ndarray:
        return np.array([v.normalized_state for v in self.vectors])

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    # ── Temperatura homeostatica ─────────────────────────────────────────────

    @property
    def temperature(self) -> float:
        tau = self.TAU_BASE / (1.0 + self.ALPHA * self.L)
        return float(max(tau, self.TAU_MIN))

    # ── Deformacao metrica ───────────────────────────────────────────────────

    def metric_deformation_matrix(self, seq_len: int, seed: int = 99) -> np.ndarray:
        """G(H(t)) — deformacao metrica. W fixo (gray box, seed=99)."""
        cache_key = (seq_len, seed)
        if cache_key not in self._G_cache:
            rng = np.random.RandomState(seed=seed)
            self._G_cache[cache_key] = rng.randn(seq_len * seq_len, 5) * 0.25
        W = self._G_cache[cache_key]
        h_norm = self.normalized_states
        deformation_flat = W @ h_norm
        deformation = deformation_flat.reshape(seq_len, seq_len)
        deformation = (deformation + deformation.T) / 2.0
        return np.eye(seq_len) + 0.3 * np.tanh(deformation)

    # ── Atualizacao ──────────────────────────────────────────────────────────

    def update(self, representation: np.ndarray) -> None:
        for v in self.vectors:
            v.update(representation)
        self.step += 1
        self.history.append(self.state.copy())
        self.deviation_history.append(self.deviations.copy())
        self.temperature_history.append(self.temperature)

        # Calibracao ensemble apos burn-in
        if not self._calibrated and self.step >= self.BURN_IN_STEPS:
            self._run_ensemble_calibration()

    # ── Diagnostico a posteriori ─────────────────────────────────────────────

    def emotional_label(self) -> str:
        devs = self.deviations
        if devs[1] > 0.15:
            if devs[0] < 0.05:
                return "FEAR_SEEK_TRANSITION"
            return "FEAR_ANALOG"
        elif devs[2] > 0.12 and devs[4] > 0.10:
            return "RAGE_ANALOG"
        elif devs[0] > 0.10:
            return "PANIC_ANALOG"
        elif self.L < 0.005:
            return "SEEK_ANALOG"
        else:
            return "TRANSITIONAL"

    def summary(self) -> dict:
        return {
            "step":            self.step,
            "H":               self.state.tolist(),
            "set_points":      self.set_points.tolist(),
            "decay_rates":     self.decay_rates.tolist(),
            "deviations":      self.deviations.tolist(),
            "L_t":             round(self.L, 6),
            "temperature":     round(self.temperature, 4),
            "emotional_label": self.emotional_label(),
            "calibrated":      self._calibrated,
        }
