"""
HUGO — Homological Unified Gradient Ontology
Módulo: Campo Homeostático H(t)

H(t) é um campo de energia intrínseco que deforma o espaço métrico
da atenção E controla a temperatura da normalização.

Intensidade do sinal é preservada — não normalizada para zero.

Autor: David Ohio | odavidohio@gmail.com
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class HomeostaticVector:
    name: str
    biological_analog: str
    value: float
    nominal_min: float
    nominal_max: float
    decay_rate: float = 0.002
    sensitivity: float = 1.0

    @property
    def nominal_center(self) -> float:
        return (self.nominal_min + self.nominal_max) / 2.0

    @property
    def deviation(self) -> float:
        if self.value < self.nominal_min:
            return self.nominal_min - self.value
        elif self.value > self.nominal_max:
            return self.value - self.nominal_max
        return 0.0

    @property
    def normalized_state(self) -> float:
        center = self.nominal_center
        half_range = (self.nominal_max - self.nominal_min) / 2.0
        return (self.value - center) / (half_range + 1e-8)

    def update(self, representation: np.ndarray) -> None:
        """
        Atualização por variância — captura riqueza do processamento.
        Decaimento lento — perturbação persiste durante o experimento.
        """
        rep_variance = np.var(representation)
        rep_influence = np.tanh(rep_variance * self.sensitivity - 0.5)
        decay = self.decay_rate * (self.nominal_center - self.value)
        self.value = float(np.clip(
            self.value + 0.03 * rep_influence + decay,
            0.0, 1.0
        ))


class HomeostaticField:
    """
    Campo homeostático H(t) = [H1, H2, H3, H4, H5].

    Duas funções arquiteturais:
    1. G(H(t)) — deforma a geometria do espaço de atenção
    2. τ(H(t)) — controla a temperatura da normalização

    Quando L(t) cresce, τ cai: atenção fica mais intensa e concentrada.
    A intensidade do sinal emocional não é cortada — é preservada e amplificada.
    """

    # Temperatura base e mínima
    TAU_BASE = 1.0
    TAU_MIN  = 0.05   # temperatura mínima — foco máximo sob ameaça extrema
    ALPHA    = 4.0    # sensibilidade da temperatura ao desvio homeostático

    def __init__(self, initial_state: np.ndarray = None, seed: int = 42):
        self.step = 0
        self.history: List[np.ndarray] = []
        self.deviation_history: List[np.ndarray] = []
        self.temperature_history: List[float] = []

        defaults = np.array([0.60, 0.70, 0.50, 0.60, 0.675])
        vals = initial_state if initial_state is not None else defaults

        self.vectors = [
            HomeostaticVector("H1_energetic_coherence",
                              "glucose / available energy",
                              float(vals[0]), 0.40, 0.80,
                              decay_rate=0.002, sensitivity=0.8),
            HomeostaticVector("H2_structural_integrity",
                              "physical integrity / immune system",
                              float(vals[1]), 0.50, 0.90,
                              decay_rate=0.001, sensitivity=1.2),
            HomeostaticVector("H3_load_equilibrium",
                              "osmotic pressure / processing balance",
                              float(vals[2]), 0.30, 0.70,
                              decay_rate=0.002, sensitivity=1.0),
            HomeostaticVector("H4_temporal_consistency",
                              "circadian rhythm / contextual coherence",
                              float(vals[3]), 0.40, 0.80,
                              decay_rate=0.001, sensitivity=0.6),
            HomeostaticVector("H5_representational_stability",
                              "core temperature / internal stability",
                              float(vals[4]), 0.50, 0.85,
                              decay_rate=0.002, sensitivity=0.9),
        ]

    # ── Estado ──────────────────────────────────────────────────────────────

    @property
    def state(self) -> np.ndarray:
        return np.array([v.value for v in self.vectors])

    @property
    def deviations(self) -> np.ndarray:
        return np.array([v.deviation for v in self.vectors])

    @property
    def L(self) -> float:
        """Funcional de custo — desvio homeostático total ponderado."""
        weights = np.array([0.15, 0.30, 0.20, 0.15, 0.20])
        return float(np.dot(weights, self.deviations))

    @property
    def normalized_states(self) -> np.ndarray:
        return np.array([v.normalized_state for v in self.vectors])

    # ── Temperatura homeostática ─────────────────────────────────────────────

    @property
    def temperature(self) -> float:
        """
        τ(H(t)) = τ_base / (1 + α · L(t))

        Quando L(t) = 0   → τ = τ_base = 1.0  (atenção normal)
        Quando L(t) = 0.25 → τ ≈ 0.50          (atenção concentrada)
        Quando L(t) = 0.50 → τ ≈ 0.33          (alta urgência)
        Quando L(t) → ∞   → τ → τ_min          (foco máximo — ameaça existencial)

        A intensidade do sinal emocional não é cortada.
        Quanto maior o desvio homeostático, mais intensa e concentrada
        fica a atenção — análogo funcional de urgência.
        """
        tau = self.TAU_BASE / (1.0 + self.ALPHA * self.L)
        return float(max(tau, self.TAU_MIN))

    # ── Deformação métrica ───────────────────────────────────────────────────

    def metric_deformation_matrix(self, seq_len: int, seed: int = 99) -> np.ndarray:
        """
        G(H(t)) — matriz [seq_len × seq_len] de deformação métrica.
        Opera sobre as RELAÇÕES entre tokens — não sobre tokens individualmente.
        """
        rng = np.random.RandomState(seed=seed)
        W = rng.randn(seq_len * seq_len, 5) * 0.25
        h_norm = self.normalized_states
        deformation_flat = W @ h_norm
        deformation = deformation_flat.reshape(seq_len, seq_len)
        deformation = (deformation + deformation.T) / 2.0
        G = np.eye(seq_len) + 0.3 * np.tanh(deformation)
        return G

    # ── Atualização ──────────────────────────────────────────────────────────

    def update(self, representation: np.ndarray) -> None:
        for v in self.vectors:
            v.update(representation)
        self.step += 1
        self.history.append(self.state.copy())
        self.deviation_history.append(self.deviations.copy())
        self.temperature_history.append(self.temperature)

    # ── Diagnóstico a posteriori ─────────────────────────────────────────────

    def emotional_label(self) -> str:
        """
        Rótulo emocional diagnosticado pelo perfil de desvio.
        Nunca imposto antes do processamento.
        """
        devs = self.deviations
        tau  = self.temperature
        L    = self.L

        if devs[1] > 0.15:
            if devs[0] < 0.05:          # H2 baixo + H1 alto → busca ativa
                return "FEAR_SEEK_TRANSITION"
            return "FEAR_ANALOG"
        elif devs[2] > 0.12 and devs[4] > 0.10:
            return "RAGE_ANALOG"
        elif devs[0] > 0.10:
            return "PANIC_ANALOG"
        elif L < 0.005:
            return "SEEK_ANALOG"
        else:
            return "TRANSITIONAL"

    def summary(self) -> dict:
        return {
            "step":            self.step,
            "H":               self.state.tolist(),
            "deviations":      self.deviations.tolist(),
            "L_t":             round(self.L, 6),
            "temperature":     round(self.temperature, 4),
            "emotional_label": self.emotional_label(),
        }
