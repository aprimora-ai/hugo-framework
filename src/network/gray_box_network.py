"""
HUGO — Homological Unified Gradient Ontology
Módulo: Rede Neural Gray Box — Atenção com Temperatura Homeostática

Arquitetura central:
    A(t) = softmax( G(H(t)) · scores / τ(H(t)) )

Onde:
    G(H(t)) — deforma a geometria do espaço de atenção
    τ(H(t)) — temperatura derivada do desvio homeostático L(t)

Quando L(t) cresce, τ cai.
A intensidade do sinal não é normalizada para zero —
é preservada e amplificada pelo próprio estado homeostático.

Autor: David Ohio | odavidohio@gmail.com
"""

import numpy as np
from typing import List, Tuple, Optional
from ..homeostasis.homeostatic_field import HomeostaticField


class StructuralAttentionNetwork:
    """
    Rede de atenção estrutural com temperatura homeostática intrínseca.

    Pesos Q, K, V fixos e aleatórios — sem treinamento, sem objetivo.
    O comportamento emergente é consequência da geometria e da temperatura
    imposta por H(t) — não de otimização.

    seq_len=32 garante matrizes A(t) ricas para análise topológica.
    """

    def __init__(
        self,
        input_dim: int = 64,
        seq_len: int = 32,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        initial_homeostasis: np.ndarray = None,
        seed: int = 42
    ):
        self.input_dim  = input_dim
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim
        self.n_heads    = n_heads
        self.n_layers   = n_layers
        self.head_dim   = hidden_dim // n_heads
        self.step       = 0

        self.attention_history:     List[np.ndarray] = []
        self.representation_history:List[np.ndarray] = []
        self.temperature_history:   List[float]      = []
        self.raw_score_history:     List[np.ndarray] = []   # scores antes do softmax

        self.homeostatic_field = HomeostaticField(
            initial_state=initial_homeostasis,
            seed=seed
        )

        rng = np.random.RandomState(seed=seed)
        self._init_fixed_weights(rng)

    def _init_fixed_weights(self, rng: np.random.RandomState) -> None:
        self.W_input = rng.randn(self.input_dim, self.hidden_dim) * 0.1
        self.W_Q, self.W_K, self.W_V, self.W_O = [], [], [], []
        for _ in range(self.n_layers):
            self.W_Q.append(rng.randn(self.hidden_dim, self.hidden_dim) * 0.1)
            self.W_K.append(rng.randn(self.hidden_dim, self.hidden_dim) * 0.1)
            self.W_V.append(rng.randn(self.hidden_dim, self.hidden_dim) * 0.1)
            self.W_O.append(rng.randn(self.hidden_dim, self.hidden_dim) * 0.1)

    # ── Softmax com temperatura ─────────────────────────────────────────────

    def _temperature_softmax(self, scores: np.ndarray, tau: float) -> np.ndarray:
        """
        softmax(scores / τ)

        τ pequeno → distribuição mais concentrada (alta urgência)
        τ → 0     → distribuição colapsa num único token (foco máximo)
        τ = 1.0   → softmax padrão

        A temperatura não é um hiperparâmetro fixo —
        é derivada do estado homeostático a cada step.
        """
        scaled = scores / (tau + 1e-8)
        shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
        ex = np.exp(shifted)
        return ex / (np.sum(ex, axis=-1, keepdims=True) + 1e-10)

    # ── Camada de atenção ───────────────────────────────────────────────────

    def _attention_layer(
        self, x: np.ndarray, layer_idx: int, tau: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Camada de atenção com deformação métrica G(H(t)) e temperatura τ(H(t)).

        Retorna:
            output      — representação após atenção
            A_mean      — matriz de atenção média entre cabeças [seq × seq]
            scores_mean — scores brutos médios [seq × seq] (antes do softmax)
        """
        seq_len = x.shape[0]
        scale   = np.sqrt(self.head_dim)

        Q = x @ self.W_Q[layer_idx]
        K = x @ self.W_K[layer_idx]
        V = x @ self.W_V[layer_idx]

        # Matriz de deformação métrica — altera relações entre tokens
        G = self.homeostatic_field.metric_deformation_matrix(seq_len)

        all_output = []
        all_attn   = []
        all_scores = []

        for h in range(self.n_heads):
            s = h * self.head_dim
            e = s + self.head_dim

            Qh = Q[:, s:e]
            Kh = K[:, s:e]
            Vh = V[:, s:e]

            # Scores no espaço deformado por G(H(t))
            raw = (Qh @ Kh.T) / scale
            deformed = G * raw                      # Hadamard — deformação métrica

            # Softmax com temperatura homeostática
            attn = self._temperature_softmax(deformed, tau)
            out  = attn @ Vh

            all_output.append(out)
            all_attn.append(attn)
            all_scores.append(deformed)

        concat = np.concatenate(all_output, axis=-1)
        output = concat @ self.W_O[layer_idx]

        A_mean      = np.mean(all_attn,   axis=0)   # [seq × seq]
        scores_mean = np.mean(all_scores, axis=0)   # [seq × seq]

        return output, A_mean, scores_mean

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x_input: np.ndarray) -> dict:
        """
        Passo de processamento completo.

        x_input: (seq_len, input_dim) — input unificado, idêntico entre condições.
        A diferença de resposta emerge exclusivamente de H(t).
        """
        if x_input.ndim == 1:
            x_input = x_input.reshape(1, -1)

        # Ajusta seq_len
        if x_input.shape[0] != self.seq_len:
            reps = int(np.ceil(self.seq_len / x_input.shape[0]))
            x_input = np.tile(x_input, (reps, 1))[:self.seq_len]

        x = np.tanh(x_input @ self.W_input)

        # Temperatura atual — derivada do estado homeostático
        tau = self.homeostatic_field.temperature

        attention_layers = []
        raw_score_layers = []

        for layer_idx in range(self.n_layers):
            x, A, scores = self._attention_layer(x, layer_idx, tau)
            x = np.tanh(x)
            attention_layers.append(A)
            raw_score_layers.append(scores)

        r_final = np.mean(x, axis=0)

        # Atualiza H(t) pela variância da representação
        self.homeostatic_field.update(r_final)

        A_final      = attention_layers[-1]
        scores_final = raw_score_layers[-1]

        self.attention_history.append(A_final.copy())
        self.representation_history.append(r_final.copy())
        self.temperature_history.append(tau)
        self.raw_score_history.append(scores_final.copy())
        self.step += 1

        return {
            "A_final":         A_final,          # [seq × seq] — pós-softmax
            "scores_final":    scores_final,      # [seq × seq] — pré-softmax
            "r_final":         r_final,           # [hidden_dim]
            "tau":             tau,
            "H_state":         self.homeostatic_field.state.copy(),
            "deviations":      self.homeostatic_field.deviations.copy(),
            "L_t":             self.homeostatic_field.L,
            "emotional_label": self.homeostatic_field.emotional_label(),
            "step":            self.step,
        }
