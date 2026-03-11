"""
HUGO — Homological Unified Gradient Ontology
Módulo: Pipeline Kappa em Tempo Real

Monitora dois espaços simultaneamente:
  1. A(t)      — fluxo pós-softmax  (direção da atenção)
  2. scores(t) — potencial pré-softmax (intensidade do sinal)

A temperatura τ(H(t)) afeta A(t) diretamente.
Os scores preservam as magnitudes que o softmax redistribui.
Medir ambos permite separar efeitos de direção e intensidade.

Autor: David Ohio | odavidohio@gmail.com
"""

import numpy as np
from typing import List, Dict, Optional
from ripser import ripser
from persim import wasserstein
import warnings
warnings.filterwarnings('ignore')


class KappaMonitor:

    def __init__(self, max_dim: int = 1, threshold: float = 1.5):
        """
        threshold maior (1.5) para matrizes 32×32 — escala correta
        para detectar estrutura topológica real.
        """
        self.max_dim   = max_dim
        self.threshold = threshold

        # Séries temporais — pós-softmax
        self.betti_history:   List[np.ndarray] = []
        self.entropy_history: List[float]      = []
        self.wass_history:    List[float]      = []

        # Séries temporais — pré-softmax (intensidade)
        self.score_betti_history:   List[np.ndarray] = []
        self.score_entropy_history: List[float]      = []
        self.score_wass_history:    List[float]      = []

        # Temperatura e L(t)
        self.tau_history: List[float] = []
        self.L_history:   List[float] = []

        self.diagram_history:       List[dict] = []
        self.score_diagram_history: List[dict] = []
        self.step = 0

    # ── Homologia persistente ──────────────────────────────────────────────

    def _compute_ph(self, point_cloud: np.ndarray) -> dict:
        if point_cloud.shape[0] < 4:
            empty = np.array([]).reshape(0, 2)
            return {"dgms": [np.array([[0, 1]]), empty]}
        try:
            return ripser(point_cloud, maxdim=self.max_dim, thresh=self.threshold)
        except Exception:
            empty = np.array([]).reshape(0, 2)
            return {"dgms": [np.array([[0, 1]]), empty]}

    def _betti_numbers(self, diagrams: list) -> np.ndarray:
        betti = np.zeros(self.max_dim + 1)
        min_pers = 0.02
        for dim, dgm in enumerate(diagrams):
            if dim > self.max_dim or len(dgm) == 0:
                continue
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) > 0:
                pers = finite[:, 1] - finite[:, 0]
                betti[dim] = int(np.sum(pers > min_pers))
        return betti

    def _persistence_entropy(self, diagrams: list) -> float:
        all_pers = []
        for dgm in diagrams:
            if len(dgm) == 0:
                continue
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite) > 0:
                pers = finite[:, 1] - finite[:, 0]
                all_pers.extend(pers[pers > 0])
        if not all_pers:
            return 0.0
        p = np.array(all_pers)
        p = p / (p.sum() + 1e-10)
        return float(-np.sum(p * np.log(p + 1e-10)))

    def _wasserstein_dist(self, dgm_curr: np.ndarray, dgm_prev: np.ndarray) -> float:
        try:
            c = dgm_curr[np.isfinite(dgm_curr[:, 1])] if len(dgm_curr) > 0 else np.array([]).reshape(0, 2)
            p = dgm_prev[np.isfinite(dgm_prev[:, 1])] if len(dgm_prev) > 0 else np.array([]).reshape(0, 2)
            if len(c) == 0 or len(p) == 0:
                return 0.0
            return float(wasserstein(c, p))
        except Exception:
            return 0.0

    # ── Processamento ──────────────────────────────────────────────────────

    def process(
        self,
        A: np.ndarray,
        scores: np.ndarray,
        r: np.ndarray,
        H_state: np.ndarray,
        L_t: float,
        tau: float,
    ) -> dict:
        """
        Processa um step — mede topologia de A(t) e de scores(t).

        A(t)      — distribuição de atenção (direção)
        scores(t) — potencial bruto (intensidade, preservada pelo τ)
        """

        # ── 1. Topologia de A(t) — pós-softmax ──
        ph_A    = self._compute_ph(A)
        dgms_A  = ph_A["dgms"]
        betti_A = self._betti_numbers(dgms_A)
        ent_A   = self._persistence_entropy(dgms_A)

        wass_A = 0.0
        if self.diagram_history:
            prev_dgm1 = self.diagram_history[-1]["dgms"][1] if len(self.diagram_history[-1]["dgms"]) > 1 else np.array([]).reshape(0,2)
            curr_dgm1 = dgms_A[1] if len(dgms_A) > 1 else np.array([]).reshape(0,2)
            wass_A = self._wasserstein_dist(curr_dgm1, prev_dgm1)

        # ── 2. Topologia de scores(t) — pré-softmax ──
        # Normalização por norma para comparabilidade — preserva estrutura relativa
        scores_norm = scores / (np.linalg.norm(scores, axis=-1, keepdims=True) + 1e-8)
        ph_S    = self._compute_ph(scores_norm)
        dgms_S  = ph_S["dgms"]
        betti_S = self._betti_numbers(dgms_S)
        ent_S   = self._persistence_entropy(dgms_S)

        wass_S = 0.0
        if self.score_diagram_history:
            prev_s1 = self.score_diagram_history[-1]["dgms"][1] if len(self.score_diagram_history[-1]["dgms"]) > 1 else np.array([]).reshape(0,2)
            curr_s1 = dgms_S[1] if len(dgms_S) > 1 else np.array([]).reshape(0,2)
            wass_S = self._wasserstein_dist(curr_s1, prev_s1)

        # ── 3. Registro ──
        self.betti_history.append(betti_A.copy())
        self.entropy_history.append(ent_A)
        self.wass_history.append(wass_A)
        self.diagram_history.append({"dgms": dgms_A, "step": self.step})

        self.score_betti_history.append(betti_S.copy())
        self.score_entropy_history.append(ent_S)
        self.score_wass_history.append(wass_S)
        self.score_diagram_history.append({"dgms": dgms_S, "step": self.step})

        self.tau_history.append(tau)
        self.L_history.append(L_t)
        self.step += 1

        return {
            "step": self.step,
            # pós-softmax
            "betti_0":    int(betti_A[0]),
            "betti_1":    int(betti_A[1]) if len(betti_A) > 1 else 0,
            "entropy_A":  round(ent_A, 6),
            "wass_A":     round(wass_A, 6),
            # pré-softmax
            "score_betti_0": int(betti_S[0]),
            "score_betti_1": int(betti_S[1]) if len(betti_S) > 1 else 0,
            "entropy_S":     round(ent_S, 6),
            "wass_S":        round(wass_S, 6),
            # homeostase
            "tau":   round(tau, 4),
            "L_t":   round(L_t, 6),
            "H_state": H_state.tolist(),
            "topological_signature": self._signature(betti_A, ent_A, tau),
        }

    def _signature(self, betti: np.ndarray, entropy: float, tau: float) -> str:
        b1 = int(betti[1]) if len(betti) > 1 else 0
        if tau < 0.2 and entropy < 1.5:
            return "HIGH_URGENCY_CONTRACTED"
        elif tau < 0.4:
            return "MODERATE_URGENCY"
        elif b1 >= 3 and entropy > 2.5:
            return "EXPANDED_EXPLORATORY"
        elif entropy < 1.0:
            return "CONTRACTED"
        else:
            return "NOMINAL"

    def get_time_series(self) -> dict:
        return {
            # pós-softmax
            "betti_0":    [int(b[0]) for b in self.betti_history],
            "betti_1":    [int(b[1]) if len(b) > 1 else 0 for b in self.betti_history],
            "entropy_A":  self.entropy_history.copy(),
            "wass_A":     self.wass_history.copy(),
            # pré-softmax
            "score_betti_1":  [int(b[1]) if len(b) > 1 else 0 for b in self.score_betti_history],
            "entropy_S":      self.score_entropy_history.copy(),
            "wass_S":         self.score_wass_history.copy(),
            # temperatura e custo
            "tau":  self.tau_history.copy(),
            "L_t":  self.L_history.copy(),
            "n_steps": self.step,
        }

    def reset(self) -> None:
        self.__init__(max_dim=self.max_dim, threshold=self.threshold)
