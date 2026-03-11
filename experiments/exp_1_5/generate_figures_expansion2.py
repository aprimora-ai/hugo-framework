"""
HUGO — Expansão 2
Geração de Figuras para Publicação

Figuras geradas:
  Fig 1 — Temperatura τ por configuração — todas as 8 (boxplot)
  Fig 2 — Mapa emocional: espaço τ × L(t) com anotações
  Fig 3 — Gradiente FEAR: monotonia das métricas vs H2 decrescente
  Fig 4 — Wasserstein S(t): perfis de volatilidade (CARE, CONFLICT, EXHAUSTION)
  Fig 5 — Entropia A(t) vs S(t): dissociação pós/pré-softmax
  Fig 6 — PCA 2D: separabilidade das 8 configurações no espaço de features
  Fig 7 — Painel unificado para publicação

Autor: David Ohio | odavidohio@gmail.com
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_PATH = ROOT / "results" / "exp_expansion_2" / "exp_expansion_2_results.json"
FIGURES_DIR  = ROOT / "results" / "exp_expansion_2" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Paleta e estilo ────────────────────────────────────────────────────────

# Gradiente FEAR: 4 tons de vermelho do claro ao escuro
COLORS = {
    "FEAR_GRAD_1":    "#FFAAAA",   # vermelho muito claro — PRE_FEAR
    "FEAR_GRAD_2":    "#EE6666",   # vermelho claro — FEAR_MILD
    "FEAR_GRAD_3":    "#DD2222",   # vermelho médio — FEAR_ANALOG
    "FEAR_GRAD_4":    "#880000",   # vermelho escuro — FEAR_EXTREME
    "CARE":           "#22AA66",   # verde — afiliativo
    "EXHAUSTION":     "#8B5CF6",   # roxo — depleção distribuída
    "CONFLICT":       "#E8A020",   # laranja — conflito motivacional
    "FEAR_SEEK_TRANS":"#4C72B0",   # azul — transição FEAR-SEEK
}

LABELS = {
    "FEAR_GRAD_1":    "FEAR_GRAD_1\n(PRE-FEAR)",
    "FEAR_GRAD_2":    "FEAR_GRAD_2\n(FEAR_MILD)",
    "FEAR_GRAD_3":    "FEAR_GRAD_3\n(FEAR)",
    "FEAR_GRAD_4":    "FEAR_GRAD_4\n(FEAR_EXTREME)",
    "CARE":           "CARE\n(afiliativo)",
    "EXHAUSTION":     "EXHAUSTION\n(depleção)",
    "CONFLICT":       "CONFLICT\n(ambivalência)",
    "FEAR_SEEK_TRANS":"FEAR_SEEK\n(transição)",
}

LABELS_SHORT = {
    "FEAR_GRAD_1":    "PRE-FEAR",
    "FEAR_GRAD_2":    "FEAR_MILD",
    "FEAR_GRAD_3":    "FEAR",
    "FEAR_GRAD_4":    "FEAR_EXT",
    "CARE":           "CARE",
    "EXHAUSTION":     "EXHAUST",
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

configs = [k for k in data.keys()
           if k not in ("statistical_analysis", "fear_gradient_analysis")]

def get_trials(cfg, metric):
    return [t[metric] for t in data[cfg]["trial_summaries"]]

def get_agg(cfg, metric):
    return data[cfg]["aggregate"][metric]

# ── Fig 1 — τ por configuração (boxplot) ──────────────────────────────────

def fig1_tau_boxplot():
    fig, ax = plt.subplots(figsize=(12, 5))

    vals   = [get_trials(c, "mean_tau") for c in configs]
    colors = [COLORS[c] for c in configs]
    xlabs  = [LABELS_SHORT[c] for c in configs]

    bp = ax.boxplot(vals, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Destaca gradiente FEAR com seta indicando direção
    ax.annotate("", xy=(4.5, 0.52), xytext=(1.5, 0.73),
                arrowprops=dict(arrowstyle="-|>", color="#880000",
                                lw=1.5, mutation_scale=14))
    ax.text(2.8, 0.62, "H2 decresce\n(ameaça aumenta)",
            fontsize=8, color="#880000", ha="center")

    ax.set_xticks(range(1, len(configs)+1))
    ax.set_xticklabels(xlabs, fontsize=8.5)
    ax.set_ylabel("Temperatura homeostática τ(H(t))", fontsize=10)
    ax.set_title("Fig 1 — Temperatura τ por configuração homeostática\n"
                 "HUGO — Expansão 2 | David Ohio", fontsize=11, pad=10)
    ax.axhline(1.0, ls="--", lw=0.8, color="gray", alpha=0.4, label="τ máximo (equilíbrio)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = FIGURES_DIR / "fig1_tau_boxplot_exp2.png"
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {out.name}")

# ── Fig 2 — Mapa emocional τ × L(t) ──────────────────────────────────────

def fig2_emotional_map():
    fig, ax = plt.subplots(figsize=(9, 7))

    for cfg in configs:
        tau_vals = get_trials(cfg, "mean_tau")
        L_vals   = get_trials(cfg, "mean_L_t")
        ax.scatter(L_vals, tau_vals,
                   color=COLORS[cfg], alpha=0.25, s=18)
        ax.scatter(np.mean(L_vals), np.mean(tau_vals),
                   color=COLORS[cfg], s=120, zorder=5,
                   edgecolors="black", linewidth=0.8)
        ax.annotate(LABELS_SHORT[cfg],
                    (np.mean(L_vals), np.mean(tau_vals)),
                    xytext=(6, 5), textcoords="offset points",
                    fontsize=8.5, color=COLORS[cfg], fontweight="bold")

    # Quadrantes
    ax.axvline(0.15, ls="--", lw=0.8, color="gray", alpha=0.4)
    ax.axhline(0.70, ls="--", lw=0.8, color="gray", alpha=0.4)
    ax.text(0.04, 0.82, "baixa urgência\nalta abertura",
            fontsize=7.5, color="gray", style="italic")
    ax.text(0.21, 0.55, "alta urgência\nbaixa abertura",
            fontsize=7.5, color="gray", style="italic")

    ax.set_xlabel("L(t) médio — desvio homeostático total", fontsize=10)
    ax.set_ylabel("τ(H(t)) médio — temperatura da atenção", fontsize=10)
    ax.set_title("Fig 2 — Mapa emocional: espaço τ × L(t)\n"
                 "Cada ponto = 1 trial | círculo grande = média da config",
                 fontsize=11, pad=10)

    plt.tight_layout()
    out = FIGURES_DIR / "fig2_emotional_map_exp2.png"
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {out.name}")

# ── Fig 3 — Gradiente FEAR: monotonia ────────────────────────────────────

def fig3_fear_gradient():
    fear_cfgs = ["FEAR_GRAD_1", "FEAR_GRAD_2", "FEAR_GRAD_3", "FEAR_GRAD_4"]
    h2_vals   = [0.55, 0.40, 0.25, 0.10]
    h2_labels = ["H2=0.55\n(PRE-FEAR)", "H2=0.40\n(MILD)", "H2=0.25\n(FEAR)", "H2=0.10\n(EXTREME)"]
    feat_colors = [COLORS[c] for c in fear_cfgs]

    metrics = [
        ("mean_tau",       "τ(H(t)) médio",          True),
        ("mean_L_t",       "L(t) médio",              False),
        ("mean_entropy_A", "Entropia A(t) pós-softmax", True),
        ("mean_wass_S",    "Wasserstein S(t) médio",  False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    fig.suptitle("Fig 3 — Gradiente FEAR: monotonia das métricas com H2 decrescente\n"
                 "Evidência de que emoção é campo contínuo, não categoria discreta",
                 fontsize=11, y=1.01)

    for idx, (metric, ylabel, decreasing) in enumerate(metrics):
        ax = axes[idx]
        means = [get_agg(c, metric) for c in fear_cfgs]
        stds  = [np.std(get_trials(c, metric)) for c in fear_cfgs]

        ax.errorbar(h2_vals, means, yerr=stds,
                    fmt="o-", color="#880000", lw=2, ms=8,
                    capsize=4, elinewidth=1.2, markeredgecolor="black")

        for h2, mean, color in zip(h2_vals, means, feat_colors):
            ax.scatter(h2, mean, color=color, s=100, zorder=5,
                       edgecolors="black", linewidth=0.8)

        # Spearman
        fg = data.get("fear_gradient_analysis", {}).get(metric, {})
        if fg:
            r = fg.get("spearman_r", None)
            p = fg.get("p_value", None)
            mono = fg.get("monotonic", False)
            symbol = "✓" if mono else "✗"
            label  = f"Spearman r={r:.2f}  p={p:.3f}  {symbol}"
            ax.text(0.97, 0.06, label,
                    transform=ax.transAxes, ha="right",
                    fontsize=8, color="#880000" if mono else "gray")

        ax.set_xlabel("H2 (integridade)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10)
        ax.invert_xaxis()  # H2 decresce da esquerda para direita

    plt.tight_layout()
    out = FIGURES_DIR / "fig3_fear_gradient_exp2.png"
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {out.name}")

# ── Fig 4 — Wasserstein S(t): perfis de volatilidade ────────────────────

def fig4_wass_profiles():
    focus = ["CARE", "CONFLICT", "EXHAUSTION", "FEAR_GRAD_3", "FEAR_SEEK_TRANS"]
    fig, ax = plt.subplots(figsize=(10, 5))

    vals   = [get_trials(c, "mean_wass_S") for c in focus]
    colors = [COLORS[c] for c in focus]
    xlabs  = [LABELS_SHORT[c] for c in focus]

    bp = ax.boxplot(vals, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # Anotações interpretativas
    means = [np.mean(v) for v in vals]
    for i, (cfg, mean) in enumerate(zip(focus, means)):
        interpretation = {
            "CARE":           "riqueza\nafiliativa",
            "CONFLICT":       "instabilidade\nconflitual",
            "EXHAUSTION":     "estagnação\ndifusa",
            "FEAR_GRAD_3":    "medo\nfocalizado",
            "FEAR_SEEK_TRANS":"busca\nativa",
        }.get(cfg, "")
        ax.text(i+1, mean + 0.005, interpretation,
                ha="center", va="bottom", fontsize=7.5,
                color=COLORS[cfg], style="italic")

    ax.set_xticks(range(1, len(focus)+1))
    ax.set_xticklabels(xlabs, fontsize=9)
    ax.set_ylabel("Wasserstein S(t) — volatilidade topológica", fontsize=10)
    ax.set_title("Fig 4 — Perfis de volatilidade topológica: CARE vs CONFLICT vs EXHAUSTION\n"
                 "Wasserstein S(t) pré-softmax revela padrões dinâmicos distintos",
                 fontsize=11, pad=10)

    plt.tight_layout()
    out = FIGURES_DIR / "fig4_wass_profiles_exp2.png"
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {out.name}")

# ── Fig 5 — Entropia A(t) vs S(t): dissociação pós/pré-softmax ──────────

def fig5_entropy_dissociation():
    fig, ax = plt.subplots(figsize=(9, 7))

    for cfg in configs:
        ent_A = get_agg(cfg, "mean_entropy_A")
        ent_S = get_agg(cfg, "mean_entropy_S")
        ax.scatter(ent_S, ent_A,
                   color=COLORS[cfg], s=140, zorder=5,
                   edgecolors="black", linewidth=0.9)
        ax.annotate(LABELS_SHORT[cfg],
                    (ent_S, ent_A),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=8.5, color=COLORS[cfg], fontweight="bold")

    # Linha diagonal de referência (dissociação zero)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    ]
    ax.plot(lims, lims, ls="--", lw=0.8, color="gray", alpha=0.5,
            label="dissociação zero")

    ax.set_xlabel("Entropia S(t) pré-softmax — intensidade do sinal", fontsize=10)
    ax.set_ylabel("Entropia A(t) pós-softmax — direção da atenção", fontsize=10)
    ax.set_title("Fig 5 — Dissociação pós/pré-softmax: direção vs intensidade\n"
                 "Pontos abaixo da diagonal: atenção concentrada com sinal intenso",
                 fontsize=11, pad=10)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = FIGURES_DIR / "fig5_entropy_dissociation_exp2.png"
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {out.name}")

# ── Fig 6 — PCA 2D ─────────────────────────────────────────────────────────

def fig6_pca():
    feature_matrix = []
    labels_list    = []

    for cfg in configs:
        for t in data[cfg]["trial_summaries"]:
            feature_matrix.append([
                t["mean_tau"],
                t["mean_L_t"],
                t["mean_entropy_A"],
                t["mean_entropy_S"],
                t["mean_wass_A"],
                t["mean_wass_S"],
            ])
            labels_list.append(cfg)

    X = np.array(feature_matrix)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))

    for cfg in configs:
        mask = [l == cfg for l in labels_list]
        pts  = X_pca[mask]
        ax.scatter(pts[:, 0], pts[:, 1],
                   color=COLORS[cfg], alpha=0.3, s=18)
        center = pts.mean(axis=0)
        ax.scatter(*center, color=COLORS[cfg], s=130, zorder=5,
                   edgecolors="black", linewidth=0.9)
        ax.annotate(LABELS_SHORT[cfg], center,
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=9, color=COLORS[cfg], fontweight="bold")

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variância explicada)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variância explicada)", fontsize=10)
    ax.set_title(f"Fig 6 — PCA 2D: separabilidade das 8 configurações no espaço de features\n"
                 f"Variância total explicada: {(var[0]+var[1])*100:.1f}% | 6 features topológicas",
                 fontsize=11, pad=10)

    plt.tight_layout()
    out = FIGURES_DIR / "fig6_pca_exp2.png"
    plt.savefig(out)
    plt.close()
    print(f"  Salvo: {out.name}")

# ── Fig 7 — Painel unificado ──────────────────────────────────────────────

def fig7_unified_panel():
    fig = plt.figure(figsize=(18, 14))
    gs  = GridSpec(3, 3, figure=fig,
                   hspace=0.42, wspace=0.35)

    # ── τ boxplot (linha 0, col 0-1) ──
    ax1 = fig.add_subplot(gs[0, :2])
    vals   = [get_trials(c, "mean_tau") for c in configs]
    colors_list = [COLORS[c] for c in configs]
    xlabs  = [LABELS_SHORT[c] for c in configs]
    bp = ax1.boxplot(vals, patch_artist=True, notch=False,
                     medianprops=dict(color="black", linewidth=1.8),
                     whiskerprops=dict(linewidth=1.0),
                     capprops=dict(linewidth=1.0),
                     flierprops=dict(marker="o", markersize=2.5, alpha=0.4))
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    ax1.set_xticks(range(1, len(configs)+1))
    ax1.set_xticklabels(xlabs, fontsize=7.5)
    ax1.set_ylabel("τ(H(t))", fontsize=9)
    ax1.set_title("Temperatura homeostática por configuração", fontsize=10)
    ax1.axhline(1.0, ls="--", lw=0.7, color="gray", alpha=0.4)

    # ── PCA (linha 0, col 2) ──
    ax2 = fig.add_subplot(gs[0, 2])
    feature_matrix = []
    labels_list    = []
    for cfg in configs:
        for t in data[cfg]["trial_summaries"]:
            feature_matrix.append([
                t["mean_tau"], t["mean_L_t"],
                t["mean_entropy_A"], t["mean_entropy_S"],
                t["mean_wass_A"], t["mean_wass_S"],
            ])
            labels_list.append(cfg)
    X = np.array(feature_matrix)
    X_scaled = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    for cfg in configs:
        mask = [l == cfg for l in labels_list]
        pts  = X_pca[mask]
        ax2.scatter(pts[:, 0], pts[:, 1], color=COLORS[cfg], alpha=0.25, s=12)
        center = pts.mean(axis=0)
        ax2.scatter(*center, color=COLORS[cfg], s=80, zorder=5,
                    edgecolors="black", linewidth=0.7)
        ax2.annotate(LABELS_SHORT[cfg], center,
                     xytext=(4, 3), textcoords="offset points",
                     fontsize=6.5, color=COLORS[cfg], fontweight="bold")
    ax2.set_xlabel("PC1", fontsize=8)
    ax2.set_ylabel("PC2", fontsize=8)
    ax2.set_title("PCA: 8 configurações", fontsize=10)

    # ── Gradiente FEAR — τ (linha 1, col 0) ──
    ax3 = fig.add_subplot(gs[1, 0])
    fear_cfgs = ["FEAR_GRAD_1", "FEAR_GRAD_2", "FEAR_GRAD_3", "FEAR_GRAD_4"]
    h2_vals   = [0.55, 0.40, 0.25, 0.10]
    means_tau = [get_agg(c, "mean_tau") for c in fear_cfgs]
    stds_tau  = [np.std(get_trials(c, "mean_tau")) for c in fear_cfgs]
    ax3.errorbar(h2_vals, means_tau, yerr=stds_tau,
                 fmt="o-", color="#880000", lw=2, ms=7,
                 capsize=3, elinewidth=1.0, markeredgecolor="black")
    for h2, m, c in zip(h2_vals, means_tau, [COLORS[x] for x in fear_cfgs]):
        ax3.scatter(h2, m, color=c, s=80, zorder=5, edgecolors="black", lw=0.7)
    ax3.set_xlabel("H2 (integridade)", fontsize=8)
    ax3.set_ylabel("τ médio", fontsize=8)
    ax3.set_title("Gradiente FEAR — τ", fontsize=10)
    ax3.invert_xaxis()

    # ── Gradiente FEAR — L(t) (linha 1, col 1) ──
    ax4 = fig.add_subplot(gs[1, 1])
    means_L = [get_agg(c, "mean_L_t") for c in fear_cfgs]
    stds_L  = [np.std(get_trials(c, "mean_L_t")) for c in fear_cfgs]
    ax4.errorbar(h2_vals, means_L, yerr=stds_L,
                 fmt="o-", color="#880000", lw=2, ms=7,
                 capsize=3, elinewidth=1.0, markeredgecolor="black")
    for h2, m, c in zip(h2_vals, means_L, [COLORS[x] for x in fear_cfgs]):
        ax4.scatter(h2, m, color=c, s=80, zorder=5, edgecolors="black", lw=0.7)
    ax4.set_xlabel("H2 (integridade)", fontsize=8)
    ax4.set_ylabel("L(t) médio", fontsize=8)
    ax4.set_title("Gradiente FEAR — L(t)", fontsize=10)
    ax4.invert_xaxis()

    # ── Wasserstein S(t) — volatilidade (linha 1, col 2) ──
    ax5 = fig.add_subplot(gs[1, 2])
    focus = ["CARE", "CONFLICT", "EXHAUSTION", "FEAR_GRAD_3", "FEAR_SEEK_TRANS"]
    vals5 = [get_trials(c, "mean_wass_S") for c in focus]
    bp5   = ax5.boxplot(vals5, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=1.5),
                        whiskerprops=dict(linewidth=0.9),
                        capprops=dict(linewidth=0.9),
                        flierprops=dict(marker="o", markersize=2, alpha=0.4))
    for patch, cfg in zip(bp5["boxes"], focus):
        patch.set_facecolor(COLORS[cfg])
        patch.set_alpha(0.82)
    ax5.set_xticks(range(1, len(focus)+1))
    ax5.set_xticklabels([LABELS_SHORT[c] for c in focus], fontsize=7.5)
    ax5.set_ylabel("Wass S(t)", fontsize=8)
    ax5.set_title("Volatilidade topológica", fontsize=10)

    # ── Mapa emocional τ × L(t) (linha 2, col 0-1) ──
    ax6 = fig.add_subplot(gs[2, :2])
    for cfg in configs:
        tau_vals = get_trials(cfg, "mean_tau")
        L_vals   = get_trials(cfg, "mean_L_t")
        ax6.scatter(L_vals, tau_vals, color=COLORS[cfg], alpha=0.2, s=14)
        ax6.scatter(np.mean(L_vals), np.mean(tau_vals),
                    color=COLORS[cfg], s=100, zorder=5,
                    edgecolors="black", linewidth=0.7)
        ax6.annotate(LABELS_SHORT[cfg],
                     (np.mean(L_vals), np.mean(tau_vals)),
                     xytext=(5, 4), textcoords="offset points",
                     fontsize=7.5, color=COLORS[cfg], fontweight="bold")
    ax6.axvline(0.15, ls="--", lw=0.7, color="gray", alpha=0.4)
    ax6.axhline(0.70, ls="--", lw=0.7, color="gray", alpha=0.4)
    ax6.set_xlabel("L(t) médio", fontsize=8)
    ax6.set_ylabel("τ médio", fontsize=8)
    ax6.set_title("Mapa emocional: espaço τ × L(t)", fontsize=10)

    # ── Dissociação entropia (linha 2, col 2) ──
    ax7 = fig.add_subplot(gs[2, 2])
    for cfg in configs:
        ent_A = get_agg(cfg, "mean_entropy_A")
        ent_S = get_agg(cfg, "mean_entropy_S")
        ax7.scatter(ent_S, ent_A, color=COLORS[cfg], s=80, zorder=5,
                    edgecolors="black", linewidth=0.7)
        ax7.annotate(LABELS_SHORT[cfg], (ent_S, ent_A),
                     xytext=(4, 3), textcoords="offset points",
                     fontsize=6.5, color=COLORS[cfg], fontweight="bold")
    ax7.set_xlabel("Entropia S(t)", fontsize=8)
    ax7.set_ylabel("Entropia A(t)", fontsize=8)
    ax7.set_title("Dissociação pós/pré-softmax", fontsize=10)

    fig.suptitle(
        "HUGO — Expansão 2: Mapeamento Granular do Espaço Emocional Primário\n"
        "David Ohio | odavidohio@gmail.com",
        fontsize=13, y=1.01, fontweight="bold"
    )

    out = FIGURES_DIR / "fig7_unified_panel_exp2.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Salvo: {out.name}")

# ── Principal ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("HUGO — Expansão 2 — Gerando figuras")
    print("=" * 60)

    fig1_tau_boxplot()
    fig2_emotional_map()
    fig3_fear_gradient()
    fig4_wass_profiles()
    fig5_entropy_dissociation()
    fig6_pca()
    fig7_unified_panel()

    print("\nTodas as figuras salvas em:")
    print(f"  {FIGURES_DIR}")
