"""
HUGO — Expansão 3: Figuras de Trajetórias Topológicas

Fig 1 — Trajetória média de τ ao longo de 80 steps por configuração
Fig 2 — Trajetória média de L(t) ao longo de 80 steps
Fig 3 — Espaço de fase τ × L(t): trajetórias sobrepostas
Fig 4 — Convergência por configuração (boxplot)
Fig 5 — Deriva por configuração (boxplot)
Fig 6 — Entropia de trajetória por configuração
Fig 7 — Painel unificado para publicação

Autor: David Ohio | odavidohio@gmail.com
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_PATH   = ROOT / "results" / "exp_expansion_3" / "exp_expansion_3_results.json"
RADIANTE_PATH  = ROOT / "results" / "exp_expansion_3" / "radiante_trajectories.json"
FIGURES_DIR    = ROOT / "results" / "exp_expansion_3" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Paleta ────────────────────────────────────────────────────────────────

COLORS = {
    "BASELINE":       "#4C72B0",
    "FEAR":           "#DD2222",
    "CARE":           "#22AA66",
    "EXHAUSTION":     "#8B5CF6",
    "CONFLICT":       "#E8A020",
    "FEAR_SEEK_TRANS":"#009999",
}

LABELS = {
    "BASELINE":       "BASELINE (SEEK)",
    "FEAR":           "FEAR",
    "CARE":           "CARE",
    "EXHAUSTION":     "EXHAUSTION",
    "CONFLICT":       "CONFLICT",
    "FEAR_SEEK_TRANS":"FEAR-SEEK",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
})

# ── Carrega dados ─────────────────────────────────────────────────────────

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(RADIANTE_PATH, "r", encoding="utf-8") as f:
    radiante = json.load(f)

configs = [k for k in data.keys() if k != "statistical_analysis"]

def get_mean_trajectory(cfg):
    """Retorna trajetória média (N_STEPS, 5) do arquivo do Radiante."""
    traj = np.array(radiante[cfg]["mean_trajectory"])
    return traj  # colunas: [tau, L_t, ent_A, ent_S, wass_S]

def smooth(x, w=5):
    return uniform_filter1d(x, size=w)

def get_traj_metric_trials(cfg, metric):
    return [t[metric] for t in data[cfg]["trial_traj_metrics"]]

steps = np.arange(80)

# ── Fig 1 — Trajetória de τ ao longo do tempo ────────────────────────────

def fig1_tau_trajectory():
    fig, ax = plt.subplots(figsize=(12, 5))

    for cfg in configs:
        traj = get_mean_trajectory(cfg)
        tau  = smooth(traj[:, 0], w=5)
        ax.plot(steps, tau, color=COLORS[cfg], lw=2.0,
                label=LABELS[cfg], alpha=0.9)

    ax.set_xlabel("Step de processamento", fontsize=10)
    ax.set_ylabel("τ(H(t)) — temperatura homeostática", fontsize=10)
    ax.set_title("Fig 1 — Trajetória temporal de τ: como a temperatura evolui\n"
                 "durante 80 steps de processamento por configuração",
                 fontsize=11, pad=10)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.axhline(1.0, ls="--", lw=0.7, color="gray", alpha=0.4, label="τ máximo")

    plt.tight_layout()
    out = FIGURES_DIR / "fig1_tau_trajectory_exp3.png"
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {out.name}")

# ── Fig 2 — Trajetória de L(t) ───────────────────────────────────────────

def fig2_Lt_trajectory():
    fig, ax = plt.subplots(figsize=(12, 5))

    for cfg in configs:
        traj = get_mean_trajectory(cfg)
        Lt   = smooth(traj[:, 1], w=5)
        ax.plot(steps, Lt, color=COLORS[cfg], lw=2.0,
                label=LABELS[cfg], alpha=0.9)

    ax.set_xlabel("Step de processamento", fontsize=10)
    ax.set_ylabel("L(t) — desvio homeostático total", fontsize=10)
    ax.set_title("Fig 2 — Trajetória temporal de L(t): urgência ao longo do processamento\n"
                 "EXHAUSTION mantém L(t) mais alto — depleção distribuída persiste",
                 fontsize=11, pad=10)
    ax.legend(fontsize=8.5, loc="upper right")

    plt.tight_layout()
    out = FIGURES_DIR / "fig2_Lt_trajectory_exp3.png"
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {out.name}")

# ── Fig 3 — Espaço de fase τ × L(t) ─────────────────────────────────────

def fig3_phase_space():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Esquerda: trajetória completa
    ax = axes[0]
    for cfg in configs:
        traj = get_mean_trajectory(cfg)
        tau  = smooth(traj[:, 0], w=3)
        Lt   = smooth(traj[:, 1], w=3)
        ax.plot(Lt, tau, color=COLORS[cfg], lw=1.8, alpha=0.85,
                label=LABELS[cfg])
        # Ponto inicial
        ax.scatter(Lt[0], tau[0], color=COLORS[cfg], s=80, zorder=5,
                   marker="o", edgecolors="black", lw=0.7)
        # Ponto final
        ax.scatter(Lt[-1], tau[-1], color=COLORS[cfg], s=80, zorder=5,
                   marker="s", edgecolors="black", lw=0.7)

    ax.set_xlabel("L(t) — desvio homeostático", fontsize=9)
    ax.set_ylabel("τ(H(t)) — temperatura", fontsize=9)
    ax.set_title("Espaço de fase τ × L(t)\n(o=início, s=fim)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    # Direita: primeiros 20 vs últimos 20 steps (convergência)
    ax2 = axes[1]
    for cfg in configs:
        traj = get_mean_trajectory(cfg)
        tau  = traj[:, 0]
        Lt   = traj[:, 1]
        # Primeiros 20
        ax2.plot(Lt[:20], tau[:20], color=COLORS[cfg],
                 lw=1.2, alpha=0.5, ls="--")
        # Últimos 20
        ax2.plot(Lt[-20:], tau[-20:], color=COLORS[cfg],
                 lw=2.2, alpha=0.9, label=LABELS[cfg])

    ax2.set_xlabel("L(t)", fontsize=9)
    ax2.set_ylabel("τ(H(t))", fontsize=9)
    ax2.set_title("Primeiros 20 steps (tracejado)\nvs Últimos 20 steps (sólido)",
                  fontsize=10)
    ax2.legend(fontsize=7.5, loc="upper right")

    fig.suptitle("Fig 3 — Espaço de fase τ × L(t): trajetórias dinâmicas\n"
                 "HUGO — Expansão 3 | David Ohio", fontsize=11, y=1.01)

    plt.tight_layout()
    out = FIGURES_DIR / "fig3_phase_space_exp3.png"
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {out.name}")

# ── Fig 4 — Convergência (boxplot) ───────────────────────────────────────

def fig4_convergence():
    fig, ax = plt.subplots(figsize=(10, 5))

    vals   = [get_traj_metric_trials(c, "convergencia") for c in configs]
    colors = [COLORS[c] for c in configs]

    bp = ax.boxplot(vals, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    ax.axhline(1.0, ls="--", lw=1.0, color="black", alpha=0.5,
               label="limiar: < 1 converge, > 1 oscila/diverge")

    # Anotações
    interp = {
        "BASELINE":       "converge\n(atrator)",
        "FEAR":           "oscila\n(ativa)",
        "CARE":           "oscila\n(expande)",
        "EXHAUSTION":     "converge\n(estagna)",
        "CONFLICT":       "diverge\n(instável)",
        "FEAR_SEEK_TRANS":"oscila\n(transição)",
    }
    for i, cfg in enumerate(configs):
        m = np.mean(vals[i])
        ax.text(i+1, m + 0.05, interp[cfg],
                ha="center", va="bottom", fontsize=7.5,
                color=COLORS[cfg], style="italic")

    ax.set_xticks(range(1, len(configs)+1))
    ax.set_xticklabels([LABELS[c] for c in configs], fontsize=8.5)
    ax.set_ylabel("Convergência (var_final / var_inicial)", fontsize=10)
    ax.set_title("Fig 4 — Convergência de trajetória: atratores vs oscilações\n"
                 "BASELINE e EXHAUSTION convergem; CONFLICT diverge",
                 fontsize=11, pad=10)
    ax.legend(fontsize=8.5)

    plt.tight_layout()
    out = FIGURES_DIR / "fig4_convergence_exp3.png"
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {out.name}")

# ── Fig 5 — Deriva (boxplot) ─────────────────────────────────────────────

def fig5_drift():
    fig, ax = plt.subplots(figsize=(10, 5))

    vals   = [get_traj_metric_trials(c, "deriva") for c in configs]
    colors = [COLORS[c] for c in configs]

    bp = ax.boxplot(vals, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    interp = {
        "BASELINE":       "alta deriva\n(exploração)",
        "FEAR":           "baixa deriva\n(foco fixo)",
        "CARE":           "alta deriva\n(afeto móvel)",
        "EXHAUSTION":     "baixa deriva\n(colapso)",
        "CONFLICT":       "baixa deriva\n(travado)",
        "FEAR_SEEK_TRANS":"baixa deriva\n(busca tensa)",
    }
    for i, cfg in enumerate(configs):
        m = np.mean(vals[i])
        ax.text(i+1, m + 0.005, interp[cfg],
                ha="center", va="bottom", fontsize=7.5,
                color=COLORS[cfg], style="italic")

    ax.set_xticks(range(1, len(configs)+1))
    ax.set_xticklabels([LABELS[c] for c in configs], fontsize=8.5)
    ax.set_ylabel("Deriva (distância centróide inicial → final)", fontsize=10)
    ax.set_title("Fig 5 — Deriva de trajetória: quanto o sistema muda de região\n"
                 "BASELINE e CARE exploram mais; EXHAUSTION e CONFLICT permanecem fixos",
                 fontsize=11, pad=10)

    plt.tight_layout()
    out = FIGURES_DIR / "fig5_drift_exp3.png"
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {out.name}")

# ── Fig 6 — Entropia de trajetória ───────────────────────────────────────

def fig6_entropy_trajectory():
    fig, ax = plt.subplots(figsize=(10, 5))

    vals   = [get_traj_metric_trials(c, "entropia_traj") for c in configs]
    colors = [COLORS[c] for c in configs]

    bp = ax.boxplot(vals, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    ax.set_xticks(range(1, len(configs)+1))
    ax.set_xticklabels([LABELS[c] for c in configs], fontsize=8.5)
    ax.set_ylabel("Entropia da trajetória (variabilidade dos passos)", fontsize=10)
    ax.set_title("Fig 6 — Entropia de trajetória: riqueza dinâmica do processamento\n"
                 "FEAR_SEEK e CONFLICT têm maior variabilidade temporal",
                 fontsize=11, pad=10)

    plt.tight_layout()
    out = FIGURES_DIR / "fig6_entropy_trajectory_exp3.png"
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {out.name}")

# ── Fig 7 — Painel unificado ──────────────────────────────────────────────

def fig7_unified_panel():
    fig = plt.figure(figsize=(18, 14))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── τ trajectory (linha 0, col 0-1) ──
    ax1 = fig.add_subplot(gs[0, :2])
    for cfg in configs:
        traj = get_mean_trajectory(cfg)
        tau  = smooth(traj[:, 0], w=5)
        ax1.plot(steps, tau, color=COLORS[cfg], lw=2.0,
                 label=LABELS[cfg], alpha=0.9)
    ax1.set_xlabel("Step", fontsize=8)
    ax1.set_ylabel("τ(H(t))", fontsize=8)
    ax1.set_title("Trajetória de τ — 80 steps", fontsize=10)
    ax1.legend(fontsize=7.5, loc="upper right", ncol=2)

    # ── L(t) trajectory (linha 0, col 2) ──
    ax2 = fig.add_subplot(gs[0, 2])
    for cfg in configs:
        traj = get_mean_trajectory(cfg)
        Lt   = smooth(traj[:, 1], w=5)
        ax2.plot(steps, Lt, color=COLORS[cfg], lw=1.8, alpha=0.9)
    ax2.set_xlabel("Step", fontsize=8)
    ax2.set_ylabel("L(t)", fontsize=8)
    ax2.set_title("Trajetória de L(t)", fontsize=10)

    # ── Fase τ × L(t) (linha 1, col 0) ──
    ax3 = fig.add_subplot(gs[1, 0])
    for cfg in configs:
        traj = get_mean_trajectory(cfg)
        tau  = smooth(traj[:, 0], w=3)
        Lt   = smooth(traj[:, 1], w=3)
        ax3.plot(Lt, tau, color=COLORS[cfg], lw=1.6, alpha=0.85)
        ax3.scatter(Lt[0],  tau[0],  color=COLORS[cfg], s=50, marker="o",
                    edgecolors="black", lw=0.6, zorder=5)
        ax3.scatter(Lt[-1], tau[-1], color=COLORS[cfg], s=50, marker="s",
                    edgecolors="black", lw=0.6, zorder=5)
    ax3.set_xlabel("L(t)", fontsize=8)
    ax3.set_ylabel("τ", fontsize=8)
    ax3.set_title("Espaço de fase τ×L(t)", fontsize=10)

    # ── Convergência (linha 1, col 1) ──
    ax4 = fig.add_subplot(gs[1, 1])
    vals_conv = [get_traj_metric_trials(c, "convergencia") for c in configs]
    bp4 = ax4.boxplot(vals_conv, patch_artist=True, notch=False,
                      medianprops=dict(color="black", linewidth=1.5),
                      whiskerprops=dict(linewidth=0.9),
                      flierprops=dict(marker="o", markersize=2, alpha=0.4))
    for patch, cfg in zip(bp4["boxes"], configs):
        patch.set_facecolor(COLORS[cfg])
        patch.set_alpha(0.82)
    ax4.axhline(1.0, ls="--", lw=0.8, color="black", alpha=0.5)
    ax4.set_xticks(range(1, len(configs)+1))
    ax4.set_xticklabels([LABELS[c] for c in configs], fontsize=6.5, rotation=15)
    ax4.set_ylabel("Convergência", fontsize=8)
    ax4.set_title("Convergência de trajetória", fontsize=10)

    # ── Deriva (linha 1, col 2) ──
    ax5 = fig.add_subplot(gs[1, 2])
    vals_der = [get_traj_metric_trials(c, "deriva") for c in configs]
    bp5 = ax5.boxplot(vals_der, patch_artist=True, notch=False,
                      medianprops=dict(color="black", linewidth=1.5),
                      whiskerprops=dict(linewidth=0.9),
                      flierprops=dict(marker="o", markersize=2, alpha=0.4))
    for patch, cfg in zip(bp5["boxes"], configs):
        patch.set_facecolor(COLORS[cfg])
        patch.set_alpha(0.82)
    ax5.set_xticks(range(1, len(configs)+1))
    ax5.set_xticklabels([LABELS[c] for c in configs], fontsize=6.5, rotation=15)
    ax5.set_ylabel("Deriva", fontsize=8)
    ax5.set_title("Deriva de trajetória", fontsize=10)

    # ── Entropia S(t) ao longo do tempo (linha 2, col 0-1) ──
    ax6 = fig.add_subplot(gs[2, :2])
    for cfg in configs:
        traj   = get_mean_trajectory(cfg)
        ent_S  = smooth(traj[:, 3], w=5)
        ax6.plot(steps, ent_S, color=COLORS[cfg], lw=1.8,
                 label=LABELS[cfg], alpha=0.9)
    ax6.set_xlabel("Step", fontsize=8)
    ax6.set_ylabel("Entropia S(t) pré-softmax", fontsize=8)
    ax6.set_title("Evolução temporal da Entropia S(t)", fontsize=10)
    ax6.legend(fontsize=7.5, loc="upper right", ncol=2)

    # ── Entropia trajetória (linha 2, col 2) ──
    ax7 = fig.add_subplot(gs[2, 2])
    vals_et = [get_traj_metric_trials(c, "entropia_traj") for c in configs]
    bp7 = ax7.boxplot(vals_et, patch_artist=True, notch=False,
                      medianprops=dict(color="black", linewidth=1.5),
                      whiskerprops=dict(linewidth=0.9),
                      flierprops=dict(marker="o", markersize=2, alpha=0.4))
    for patch, cfg in zip(bp7["boxes"], configs):
        patch.set_facecolor(COLORS[cfg])
        patch.set_alpha(0.82)
    ax7.set_xticks(range(1, len(configs)+1))
    ax7.set_xticklabels([LABELS[c] for c in configs], fontsize=6.5, rotation=15)
    ax7.set_ylabel("Entropia de trajetória", fontsize=8)
    ax7.set_title("Riqueza dinâmica", fontsize=10)

    fig.suptitle(
        "HUGO — Expansão 3: Trajetórias Topológicas Temporais\n"
        "David Ohio | odavidohio@gmail.com",
        fontsize=13, y=1.01, fontweight="bold"
    )

    out = FIGURES_DIR / "fig7_unified_panel_exp3.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out.name}")

# ── Principal ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("HUGO — Expansão 3 — Gerando figuras")
    print("=" * 60)

    fig1_tau_trajectory()
    fig2_Lt_trajectory()
    fig3_phase_space()
    fig4_convergence()
    fig5_drift()
    fig6_entropy_trajectory()
    fig7_unified_panel()

    print("\nTodas as figuras salvas em:")
    print(f"  {FIGURES_DIR}")
