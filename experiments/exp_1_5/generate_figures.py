"""
HUGO — Experimento 1.5 v3
Geração de Figuras para Publicação

Figuras geradas:
  Fig 1 — Temperatura τ(H(t)) por configuração (boxplot)
  Fig 2 — Entropia de persistência A(t) vs S(t) (barras com IC)
  Fig 3 — Wasserstein temporal S(t) por configuração (boxplot)
  Fig 4 — Diagrama de persistência representativo por configuração (H1)
  Fig 5 — Trajetória de τ ao longo dos steps (média ± desvio)
  Fig 6 — Mapa 2D das configurações (PCA sobre vetor de features)

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

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_PATH = ROOT / "results" / "exp_1_5" / "exp_1_5_v3_results.json"
FIGURES_DIR  = ROOT / "results" / "exp_1_5" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Paleta e estilo ────────────────────────────────────────────────────────

COLORS = {
    "BASELINE":       "#4C72B0",   # azul
    "H2_LOW":         "#DD4444",   # vermelho — FEAR
    "H3H5_HIGH":      "#E8A020",   # laranja — RAGE
    "H1_LOW":         "#8B5CF6",   # roxo — PANIC
    "H2_LOW_H1_HIGH": "#22AA66",   # verde — transição FEAR-SEEK
}

LABELS = {
    "BASELINE":       "BASELINE\n(equilíbrio)",
    "H2_LOW":         "H2_LOW\n(FEAR)",
    "H3H5_HIGH":      "H3H5_HIGH\n(RAGE)",
    "H1_LOW":         "H1_LOW\n(PANIC)",
    "H2_LOW_H1_HIGH": "H2_LOW\nH1_HIGH\n(FEAR→SEEK)",
}

FONT = {"family": "DejaVu Sans", "size": 10}
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# ── Carrega dados ──────────────────────────────────────────────────────────

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

CONFIGS = [k for k in data.keys() if k != "statistical_analysis"]

def get_trial_metric(config, metric):
    return [t[metric] for t in data[config]["trial_summaries"]]


# ══════════════════════════════════════════════════════════════════════════
# FIG 1 — Temperatura τ por configuração
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 4.5))

tau_data = [get_trial_metric(c, "mean_tau") for c in CONFIGS]
bp = ax.boxplot(
    tau_data,
    patch_artist=True,
    widths=0.55,
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=3, alpha=0.4),
)

for patch, cfg in zip(bp["boxes"], CONFIGS):
    patch.set_facecolor(COLORS[cfg])
    patch.set_alpha(0.85)
for flier, cfg in zip(bp["fliers"], CONFIGS):
    flier.set(markerfacecolor=COLORS[cfg], markeredgecolor=COLORS[cfg])

ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="τ = 1.0 (baseline teórico)")
ax.set_xticks(range(1, len(CONFIGS)+1))
ax.set_xticklabels([LABELS[c] for c in CONFIGS], fontsize=9)
ax.set_ylabel("Temperatura τ(H(t))")
ax.set_title("Fig 1 — Temperatura Homeostática por Configuração\n"
             "Quanto maior o desvio L(t), menor τ — atenção mais concentrada e urgente",
             pad=10)
ax.legend(fontsize=8)

# Anotações de τ médio
for i, (cfg, vals) in enumerate(zip(CONFIGS, tau_data), 1):
    ax.text(i, np.median(vals) + 0.005, f"τ={np.mean(vals):.3f}",
            ha="center", va="bottom", fontsize=7.5, color=COLORS[cfg], fontweight="bold")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig1_temperature_by_config.png")
plt.close(fig)
print("Fig 1 salva.")


# ══════════════════════════════════════════════════════════════════════════
# FIG 2 — Entropia A(t) vs S(t) — dois espaços topológicos
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

for ax, metric, title, ylabel in zip(
    axes,
    ["mean_entropy_A", "mean_entropy_S"],
    ["Entropia de Persistência — A(t)\n(pós-softmax, direção do fluxo)",
     "Entropia de Persistência — S(t)\n(pré-softmax, intensidade do sinal)"],
    ["Entropia de Persistência", ""],
):
    means = [np.mean(get_trial_metric(c, metric)) for c in CONFIGS]
    sems  = [stats.sem(get_trial_metric(c, metric)) for c in CONFIGS]
    x     = np.arange(len(CONFIGS))

    bars = ax.bar(x, means, yerr=sems, capsize=4,
                  color=[COLORS[c] for c in CONFIGS],
                  alpha=0.85, width=0.6,
                  error_kw=dict(elinewidth=1.2, ecolor="#333"))

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in CONFIGS], fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)

    ymin = min(means) - 0.02
    ymax = max(means) + 0.02
    ax.set_ylim(ymin, ymax)

    for bar, mean, sem in zip(bars, means, sems):
        ax.text(bar.get_x() + bar.get_width()/2, mean + sem + 0.002,
                f"{mean:.4f}", ha="center", va="bottom", fontsize=7.5)

fig.suptitle("Fig 2 — Entropia Topológica: Dissociação Pós/Pré-Softmax\n"
             "Configurações homeostáticas distintas produzem assinaturas topológicas distintas\n"
             "(Kruskal-Wallis p < 10⁻⁸ em ambas as métricas)", y=1.02, fontsize=10)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig2_entropy_comparison.png")
plt.close(fig)
print("Fig 2 salva.")


# ══════════════════════════════════════════════════════════════════════════
# FIG 3 — Wasserstein S(t) — volatilidade temporal do sinal
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 4.5))

wass_data = [get_trial_metric(c, "mean_wass_S") for c in CONFIGS]
bp = ax.boxplot(
    wass_data,
    patch_artist=True,
    widths=0.55,
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=3, alpha=0.4),
)

for patch, cfg in zip(bp["boxes"], CONFIGS):
    patch.set_facecolor(COLORS[cfg])
    patch.set_alpha(0.85)
for flier, cfg in zip(bp["fliers"], CONFIGS):
    flier.set(markerfacecolor=COLORS[cfg], markeredgecolor=COLORS[cfg])

ax.set_xticks(range(1, len(CONFIGS)+1))
ax.set_xticklabels([LABELS[c] for c in CONFIGS], fontsize=9)
ax.set_ylabel("Distância de Wasserstein S(t) — step a step")
ax.set_title("Fig 3 — Volatilidade Temporal do Fluxo Informacional (pré-softmax)\n"
             "Sobrecarga (RAGE) = alta volatilidade | Exaustão (PANIC) = estagnação", pad=10)

for i, (cfg, vals) in enumerate(zip(CONFIGS, wass_data), 1):
    ax.text(i, np.median(vals) + 0.001, f"{np.mean(vals):.3f}",
            ha="center", va="bottom", fontsize=7.5, color=COLORS[cfg], fontweight="bold")

fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig3_wasserstein_temporal.png")
plt.close(fig)
print("Fig 3 salva.")


# ══════════════════════════════════════════════════════════════════════════
# FIG 4 — Diagramas de persistência sintéticos por configuração
# Gerados a partir de A(t) de um trial representativo
# ══════════════════════════════════════════════════════════════════════════

from src.network.gray_box_network import StructuralAttentionNetwork
from src.kappa.kappa_monitor import KappaMonitor
from ripser import ripser

HOMEOSTATIC_CONFIGS_VALS = {
    "BASELINE":       np.array([0.60, 0.70, 0.50, 0.60, 0.675]),
    "H2_LOW":         np.array([0.60, 0.25, 0.50, 0.60, 0.675]),
    "H3H5_HIGH":      np.array([0.60, 0.70, 0.85, 0.60, 0.95]),
    "H1_LOW":         np.array([0.15, 0.70, 0.50, 0.60, 0.675]),
    "H2_LOW_H1_HIGH": np.array([0.78, 0.25, 0.50, 0.60, 0.675]),
}

fig, axes = plt.subplots(1, 5, figsize=(16, 4))

for ax, cfg in zip(axes, CONFIGS):
    rng = np.random.RandomState(seed=42)

    net = StructuralAttentionNetwork(
        input_dim=64, seq_len=32,
        hidden_dim=128, n_heads=4, n_layers=3,
        initial_homeostasis=HOMEOSTATIC_CONFIGS_VALS[cfg].copy(),
        seed=42
    )
    # 20 steps de aquecimento
    for _ in range(20):
        x = rng.randn(32, 64)
        result = net.forward(x)

    A = result["A_final"]
    tau = result["tau"]

    # Homologia persistente de A(t)
    ph = ripser(A, maxdim=1, thresh=1.5)
    dgms = ph["dgms"]

    color = COLORS[cfg]

    # H0 — componentes
    dgm0 = dgms[0]
    finite0 = dgm0[np.isfinite(dgm0[:, 1])]
    if len(finite0) > 0:
        ax.scatter(finite0[:, 0], finite0[:, 1],
                   c=color, alpha=0.7, s=20, marker="o", label="H₀", zorder=3)
        # ponto no infinito — linha de referência
        inf_pts = dgm0[~np.isfinite(dgm0[:, 1])]
        if len(inf_pts) > 0:
            ax.scatter(inf_pts[:, 0], [ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.6] * len(inf_pts),
                       c=color, alpha=0.4, s=30, marker="^", zorder=3)

    # H1 — loops
    dgm1 = dgms[1]
    finite1 = dgm1[np.isfinite(dgm1[:, 1])]
    if len(finite1) > 0:
        ax.scatter(finite1[:, 0], finite1[:, 1],
                   c=color, alpha=0.9, s=35, marker="s", label="H₁", zorder=4,
                   edgecolors="white", linewidths=0.5)

    # Diagonal
    lim = 1.6
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.6, alpha=0.4)
    ax.set_xlim(-0.05, lim)
    ax.set_ylim(-0.05, lim)
    ax.set_xlabel("Nascimento", fontsize=8)
    if cfg == CONFIGS[0]:
        ax.set_ylabel("Morte", fontsize=8)
    ax.set_title(f"{LABELS[cfg]}\nτ={tau:.3f}", fontsize=8.5, color=color, pad=6)
    ax.tick_params(labelsize=7)

    # Legenda interna
    h0_patch = mpatches.Patch(color=color, alpha=0.7, label="H₀ (componentes)")
    h1_patch = mpatches.Patch(color=color, alpha=0.9, label="H₁ (loops)")
    ax.legend(handles=[h0_patch, h1_patch], fontsize=6.5, loc="upper left")

fig.suptitle("Fig 4 — Diagramas de Persistência por Configuração Homeostática\n"
             "Cada ponto = feature topológica em A(t) | Distância à diagonal = persistência",
             y=1.04, fontsize=10)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig4_persistence_diagrams.png")
plt.close(fig)
print("Fig 4 salva.")


# ══════════════════════════════════════════════════════════════════════════
# FIG 5 — Trajetória de τ ao longo dos steps
# Média ± desvio padrão entre trials
# ══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 4.5))

# Reconstrói trajetórias de τ a partir de trials representativos
for cfg, init_H in HOMEOSTATIC_CONFIGS_VALS.items():
    tau_trajectories = []

    for trial in range(10):   # 10 trials para estimativa
        rng_t = np.random.RandomState(seed=trial * 1000)
        net = StructuralAttentionNetwork(
            input_dim=64, seq_len=32,
            hidden_dim=128, n_heads=4, n_layers=3,
            initial_homeostasis=init_H.copy(),
            seed=42 + trial
        )
        taus = []
        for step in range(40):
            x = rng_t.randn(32, 64)
            r = net.forward(x)
            taus.append(r["tau"])
        tau_trajectories.append(taus)

    tau_arr = np.array(tau_trajectories)   # [10, 40]
    mean_tau = tau_arr.mean(axis=0)
    std_tau  = tau_arr.std(axis=0)
    steps    = np.arange(40)

    ax.plot(steps, mean_tau, color=COLORS[cfg], linewidth=2,
            label=LABELS[cfg].replace("\n", " "))
    ax.fill_between(steps, mean_tau - std_tau, mean_tau + std_tau,
                    color=COLORS[cfg], alpha=0.15)

ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
ax.set_xlabel("Step de processamento")
ax.set_ylabel("Temperatura τ(H(t))")
ax.set_title("Fig 5 — Evolução Temporal de τ por Configuração Homeostática\n"
             "Média ± desvio padrão (10 trials) | τ reflete urgência dinâmica", pad=10)
ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig5_tau_trajectory.png")
plt.close(fig)
print("Fig 5 salva.")


# ══════════════════════════════════════════════════════════════════════════
# FIG 6 — Mapa 2D das configurações (PCA sobre features)
# ══════════════════════════════════════════════════════════════════════════

from sklearn.decomposition import PCA

fig, ax = plt.subplots(figsize=(7, 6))

feature_matrix = []
labels_for_pca = []

for cfg in CONFIGS:
    summaries = data[cfg]["trial_summaries"]
    for s in summaries:
        feat = [
            s["mean_tau"],
            s["mean_L_t"],
            s["mean_entropy_A"],
            s["mean_entropy_S"],
            s["mean_wass_A"],
            s["mean_wass_S"],
        ]
        feature_matrix.append(feat)
        labels_for_pca.append(cfg)

X = np.array(feature_matrix)
X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_std)

for cfg in CONFIGS:
    idx = [i for i, l in enumerate(labels_for_pca) if l == cfg]
    pts = X_2d[idx]
    ax.scatter(pts[:, 0], pts[:, 1],
               c=COLORS[cfg], alpha=0.5, s=18, label=LABELS[cfg].replace("\n", " "))
    # Centroide
    cx, cy = pts.mean(axis=0)
    ax.scatter(cx, cy, c=COLORS[cfg], s=120, marker="*",
               edgecolors="white", linewidths=0.8, zorder=5)
    ax.annotate(LABELS[cfg].split("\n")[0],
                (cx, cy), textcoords="offset points", xytext=(6, 4),
                fontsize=8.5, color=COLORS[cfg], fontweight="bold")

var_exp = pca.explained_variance_ratio_
ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variância)")
ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variância)")
ax.set_title("Fig 6 — Espaço Topológico das Configurações Homeostáticas (PCA)\n"
             "Features: τ, L(t), entropia A/S, Wasserstein A/S\n"
             "Separação no espaço de features = distinguibilidade real", pad=10)
ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig6_pca_config_map.png")
plt.close(fig)
print("Fig 6 salva.")


# ══════════════════════════════════════════════════════════════════════════
# FIG 7 — Painel unificado (publicação)
# ══════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 10))
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── τ boxplot ──
ax1 = fig.add_subplot(gs[0, 0])
tau_data = [get_trial_metric(c, "mean_tau") for c in CONFIGS]
bp = ax1.boxplot(tau_data, patch_artist=True, widths=0.55,
                 medianprops=dict(color="white", linewidth=2),
                 flierprops=dict(marker="o", markersize=2.5, alpha=0.3))
for patch, cfg in zip(bp["boxes"], CONFIGS):
    patch.set_facecolor(COLORS[cfg]); patch.set_alpha(0.85)
ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
ax1.set_xticks(range(1, len(CONFIGS)+1))
ax1.set_xticklabels([c.replace("_", "\n") for c in CONFIGS], fontsize=7)
ax1.set_ylabel("τ(H(t))")
ax1.set_title("(A) Temperatura homeostática", fontsize=9, pad=6)

# ── Entropia A barras ──
ax2 = fig.add_subplot(gs[0, 1])
means_A = [np.mean(get_trial_metric(c, "mean_entropy_A")) for c in CONFIGS]
sems_A  = [stats.sem(get_trial_metric(c, "mean_entropy_A")) for c in CONFIGS]
x = np.arange(len(CONFIGS))
ax2.bar(x, means_A, yerr=sems_A, capsize=3,
        color=[COLORS[c] for c in CONFIGS], alpha=0.85, width=0.6,
        error_kw=dict(elinewidth=1))
ax2.set_xticks(x)
ax2.set_xticklabels([c.replace("_", "\n") for c in CONFIGS], fontsize=7)
ax2.set_ylabel("Entropia")
ax2.set_ylim(min(means_A)-0.02, max(means_A)+0.02)
ax2.set_title("(B) Entropia A(t) — pós-softmax", fontsize=9, pad=6)

# ── Wasserstein S barras ──
ax3 = fig.add_subplot(gs[0, 2])
means_S = [np.mean(get_trial_metric(c, "mean_wass_S")) for c in CONFIGS]
sems_S  = [stats.sem(get_trial_metric(c, "mean_wass_S")) for c in CONFIGS]
ax3.bar(x, means_S, yerr=sems_S, capsize=3,
        color=[COLORS[c] for c in CONFIGS], alpha=0.85, width=0.6,
        error_kw=dict(elinewidth=1))
ax3.set_xticks(x)
ax3.set_xticklabels([c.replace("_", "\n") for c in CONFIGS], fontsize=7)
ax3.set_ylabel("Wasserstein médio")
ax3.set_title("(C) Volatilidade S(t) — pré-softmax", fontsize=9, pad=6)

# ── Trajetória τ ──
ax4 = fig.add_subplot(gs[1, :2])
for cfg, init_H in HOMEOSTATIC_CONFIGS_VALS.items():
    tau_tr = []
    for trial in range(10):
        rng_t = np.random.RandomState(seed=trial*1000)
        net = StructuralAttentionNetwork(
            input_dim=64, seq_len=32, hidden_dim=128,
            n_heads=4, n_layers=3,
            initial_homeostasis=init_H.copy(), seed=42+trial)
        taus = []
        for step in range(40):
            x = rng_t.randn(32, 64); r = net.forward(x); taus.append(r["tau"])
        tau_tr.append(taus)
    arr = np.array(tau_tr); m = arr.mean(0); s = arr.std(0); st = np.arange(40)
    ax4.plot(st, m, color=COLORS[cfg], linewidth=1.8,
             label=cfg.replace("_", " "))
    ax4.fill_between(st, m-s, m+s, color=COLORS[cfg], alpha=0.12)
ax4.axhline(1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
ax4.set_xlabel("Step"); ax4.set_ylabel("τ(H(t))")
ax4.set_title("(D) Evolução temporal de τ — média ± dp (10 trials)", fontsize=9, pad=6)
ax4.legend(fontsize=7.5, ncol=2)

# ── PCA ──
ax5 = fig.add_subplot(gs[1, 2])
for cfg in CONFIGS:
    idx = [i for i, l in enumerate(labels_for_pca) if l == cfg]
    pts = X_2d[idx]
    ax5.scatter(pts[:, 0], pts[:, 1], c=COLORS[cfg], alpha=0.45, s=12)
    cx, cy = pts.mean(axis=0)
    ax5.scatter(cx, cy, c=COLORS[cfg], s=90, marker="*",
                edgecolors="white", linewidths=0.6, zorder=5)
    ax5.annotate(cfg.split("_")[0], (cx, cy),
                 textcoords="offset points", xytext=(4, 3),
                 fontsize=7.5, color=COLORS[cfg], fontweight="bold")
ax5.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", fontsize=8)
ax5.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)", fontsize=8)
ax5.set_title("(E) Mapa PCA das configurações", fontsize=9, pad=6)

fig.suptitle(
    "HUGO — Experimento 1.5\n"
    "Configurações Homeostáticas Produzem Assinaturas Topológicas Distinguíveis\n"
    "David Ohio | odavidohio@gmail.com",
    fontsize=11, y=1.01
)
fig.savefig(FIGURES_DIR / "fig7_unified_panel.png", bbox_inches="tight")
plt.close(fig)
print("Fig 7 (painel unificado) salva.")

print(f"\nTodas as figuras salvas em:\n  {FIGURES_DIR}")
print("\nFiguras geradas:")
for f in sorted(FIGURES_DIR.glob("*.png")):
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name}  ({size_kb} KB)")
