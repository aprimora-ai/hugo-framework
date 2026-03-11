"""
HUGO — Experimento 1.5 (v3 — Temperatura Homeostática)

Hipótese: configurações homeostáticas distintas, expostas ao mesmo estímulo,
produzem assinaturas topológicas distinguíveis via temperatura τ(H(t)).

A intensidade do sinal emocional é preservada — não normalizada.
O rótulo emocional é diagnóstico a posteriori.

Autor: David Ohio | odavidohio@gmail.com
"""

import sys
import numpy as np
import json
from pathlib import Path
from scipy import stats
from collections import Counter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.network.gray_box_network import StructuralAttentionNetwork
from src.kappa.kappa_monitor import KappaMonitor

# ── Configuração ──────────────────────────────────────────────────────────

N_TRIALS   = 50
N_STEPS    = 40
INPUT_DIM  = 64
SEQ_LEN    = 32
HIDDEN_DIM = 128
N_HEADS    = 4
N_LAYERS   = 3
SEED       = 42

RESULTS_DIR = ROOT / "results" / "exp_1_5"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Configurações homeostáticas ───────────────────────────────────────────
# H = [H1_energia, H2_integridade, H3_carga, H4_temporal, H5_estabilidade]
# τ esperado calculado: τ = 1.0 / (1 + 4 · L(t))

HOMEOSTATIC_CONFIGS = {
    "BASELINE": {
        "description": "Equilibrio nominal — L(t)~0, tau~1.0",
        "initial_H": np.array([0.60, 0.70, 0.50, 0.60, 0.675]),
    },
    "H2_LOW": {
        "description": "Integridade comprometida — L(t)~0.135, tau~0.65",
        "initial_H": np.array([0.60, 0.25, 0.50, 0.60, 0.675]),
    },
    "H3H5_HIGH": {
        "description": "Sobrecarga H3+H5 — L(t)~0.095, tau~0.72",
        "initial_H": np.array([0.60, 0.70, 0.85, 0.60, 0.95]),
    },
    "H1_LOW": {
        "description": "Deplecao energetica — L(t)~0.075, tau~0.77",
        "initial_H": np.array([0.15, 0.70, 0.50, 0.60, 0.675]),
    },
    "H2_LOW_H1_HIGH": {
        "description": "Ameaca com energia — L(t)~0.135, tau~0.65 (transicao FEAR-SEEK)",
        "initial_H": np.array([0.78, 0.25, 0.50, 0.60, 0.675]),
    },
}

# ── Input unificado ───────────────────────────────────────────────────────

def generate_unified_input(seq_len, input_dim, trial_seed):
    """Input gaussiano padrão — idêntico entre configurações para o mesmo trial."""
    rng = np.random.RandomState(seed=trial_seed)
    return rng.randn(seq_len, input_dim)

# ── Trial ─────────────────────────────────────────────────────────────────

def run_trial(config_name, config, trial_idx):
    net = StructuralAttentionNetwork(
        input_dim=INPUT_DIM, seq_len=SEQ_LEN,
        hidden_dim=HIDDEN_DIM, n_heads=N_HEADS, n_layers=N_LAYERS,
        initial_homeostasis=config["initial_H"].copy(),
        seed=SEED + trial_idx
    )
    kappa = KappaMonitor(max_dim=1, threshold=1.5)

    trial_data = []
    r_history  = []
    labels_obs = []

    for step in range(N_STEPS):
        x = generate_unified_input(SEQ_LEN, INPUT_DIM,
                                   trial_seed=trial_idx * 1000 + step)
        result = net.forward(x)
        r_history.append(result["r_final"])

        kr = kappa.process(
            A=result["A_final"],
            scores=result["scores_final"],
            r=result["r_final"],
            H_state=result["H_state"],
            L_t=result["L_t"],
            tau=result["tau"],
        )

        labels_obs.append(result["emotional_label"])
        trial_data.append({
            "step":              step,
            # pós-softmax
            "betti_0":           kr["betti_0"],
            "betti_1":           kr["betti_1"],
            "entropy_A":         kr["entropy_A"],
            "wass_A":            kr["wass_A"],
            # pré-softmax
            "score_betti_1":     kr["score_betti_1"],
            "entropy_S":         kr["entropy_S"],
            "wass_S":            kr["wass_S"],
            # homeostase
            "tau":               kr["tau"],
            "L_t":               kr["L_t"],
            "H_state":           kr["H_state"],
            "emotional_label":   result["emotional_label"],
            "topo_signature":    kr["topological_signature"],
        })

    ts = kappa.get_time_series()
    dominant = Counter(labels_obs).most_common(1)[0][0]

    return {
        "config": config_name, "trial": trial_idx,
        "dominant_label": dominant,
        "label_dist": dict(Counter(labels_obs)),
        "steps": trial_data,
        "summary": {
            # pós-softmax
            "mean_betti_1":  float(np.mean(ts["betti_1"])),
            "mean_entropy_A":float(np.mean(ts["entropy_A"])),
            "std_entropy_A": float(np.std(ts["entropy_A"])),
            "mean_wass_A":   float(np.mean(ts["wass_A"])),
            # pré-softmax
            "mean_sb1":      float(np.mean(ts["score_betti_1"])),
            "mean_entropy_S":float(np.mean(ts["entropy_S"])),
            "mean_wass_S":   float(np.mean(ts["wass_S"])),
            # temperatura
            "mean_tau":      float(np.mean(ts["tau"])),
            "min_tau":       float(np.min(ts["tau"])),
            "mean_L_t":      float(np.mean(ts["L_t"])),
            "final_H":       trial_data[-1]["H_state"],
        }
    }

# ── Análise estatística ───────────────────────────────────────────────────

def statistical_analysis(all_results):
    metrics = [
        "mean_entropy_A", "mean_entropy_S",
        "mean_tau", "mean_L_t",
        "mean_wass_A", "mean_wass_S",
    ]
    config_names = list(all_results.keys())
    analysis = {}
    for metric in metrics:
        groups = [[t["summary"][metric] for t in all_results[n]["trials"]]
                  for n in config_names]
        # Verifica se há variância
        all_vals = [v for g in groups for v in g]
        if np.std(all_vals) < 1e-10:
            analysis[metric] = {"error": "all_identical", "significant": False}
            continue
        stat, pval = stats.kruskal(*groups)
        analysis[metric] = {
            "kruskal_stat": round(float(stat), 4),
            "p_value":      round(float(pval), 8),
            "significant":  bool(pval < 0.05),
            "group_means":  {n: round(float(np.mean(g)), 5)
                             for n, g in zip(config_names, groups)},
        }
    return analysis

# ── Principal ─────────────────────────────────────────────────────────────

def run_experiment():
    print("=" * 68)
    print("HUGO - Experimento 1.5 v3 - Temperatura Homeostática")
    print("Input unificado | Rótulo a posteriori | Dois espaços topológicos")
    print("=" * 68)

    all_results = {}

    for config_name, config in HOMEOSTATIC_CONFIGS.items():
        # Calcula tau esperado
        from src.homeostasis.homeostatic_field import HomeostaticField
        hf = HomeostaticField(initial_state=config["initial_H"].copy())
        tau_init = hf.temperature
        L_init   = hf.L

        print(f"\n[{config_name}]")
        print(f"  {config['description']}")
        print(f"  H={np.round(config['initial_H'],2).tolist()}")
        print(f"  L(t) inicial={L_init:.4f}  tau inicial={tau_init:.4f}")

        trials    = []
        summaries = []

        for trial in range(N_TRIALS):
            result = run_trial(config_name, config, trial)
            trials.append(result)
            summaries.append(result["summary"])

            if (trial + 1) % 10 == 0:
                me  = np.mean([s["mean_entropy_A"] for s in summaries])
                mes = np.mean([s["mean_entropy_S"] for s in summaries])
                mt  = np.mean([s["mean_tau"]       for s in summaries])
                mL  = np.mean([s["mean_L_t"]       for s in summaries])
                print(f"  Trial {trial+1:3d}/{N_TRIALS} | "
                      f"ent_A={me:.4f} ent_S={mes:.4f} "
                      f"tau={mt:.3f} L(t)={mL:.4f}")

        all_results[config_name] = {
            "config":    config["description"],
            "initial_H": config["initial_H"].tolist(),
            "tau_initial": tau_init,
            "L_initial":   L_init,
            "n_trials":  N_TRIALS,
            "trials":    trials,
            "aggregate": {k: float(np.mean([s[k] for s in summaries]))
                          for k in summaries[0].keys()
                          if k != "final_H"},
            "dominant_labels": dict(Counter([t["dominant_label"] for t in trials])),
        }

        agg = all_results[config_name]["aggregate"]
        print(f"\n  RESULTADO [{config_name}]:")
        print(f"    tau medio:      {agg['mean_tau']:.4f}  (min: {agg['min_tau']:.4f})")
        print(f"    L(t) medio:     {agg['mean_L_t']:.4f}")
        print(f"    entropia A(t):  {agg['mean_entropy_A']:.4f} +/- {agg['std_entropy_A']:.4f}")
        print(f"    entropia S(t):  {agg['mean_entropy_S']:.4f}")
        print(f"    wasserstein A:  {agg['mean_wass_A']:.5f}")
        print(f"    wasserstein S:  {agg['mean_wass_S']:.5f}")
        print(f"    rotulos:        {all_results[config_name]['dominant_labels']}")

    # ── Tabela comparativa ──
    print("\n" + "=" * 68)
    print("COMPARACAO — CINCO CONFIGURACOES HOMEOSTÁTICAS")
    print("=" * 68)
    print(f"{'Config':<20} {'tau':>6} {'ent_A':>8} {'ent_S':>8} {'wass_A':>9} {'L(t)':>8}")
    print("-" * 68)
    for name, data in all_results.items():
        a = data["aggregate"]
        print(f"{name:<20} {a['mean_tau']:>6.3f} {a['mean_entropy_A']:>8.4f} "
              f"{a['mean_entropy_S']:>8.4f} {a['mean_wass_A']:>9.5f} {a['mean_L_t']:>8.4f}")

    # ── Análise estatística ──
    print("\n" + "=" * 68)
    print("DISTINGUIBILIDADE ESTATÍSTICA (Kruskal-Wallis)")
    print("=" * 68)
    stat_results = statistical_analysis(all_results)
    for metric, res in stat_results.items():
        if "error" in res:
            print(f"\n  [{metric}] — variância nula, impossível testar")
            continue
        sig = "*** SIGNIFICATIVO ***" if res["significant"] else "nao significativo"
        print(f"\n  [{metric}]  H={res['kruskal_stat']}  p={res['p_value']:.8f}  {sig}")
        for cname, mean in res["group_means"].items():
            print(f"      {cname:<22}: {mean:.5f}")

    # ── Salva ──
    def clean(obj):
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict):        return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):        return [clean(i) for i in obj]
        return obj

    out = {}
    for name, data in all_results.items():
        out[name] = {
            "config":          data["config"],
            "initial_H":       data["initial_H"],
            "tau_initial":     data["tau_initial"],
            "L_initial":       data["L_initial"],
            "n_trials":        data["n_trials"],
            "aggregate":       data["aggregate"],
            "dominant_labels": data["dominant_labels"],
            "trial_summaries": [t["summary"] for t in data["trials"]],
        }
    out["statistical_analysis"] = stat_results

    output_path = RESULTS_DIR / "exp_1_5_v3_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean(out), f, indent=2)

    print(f"\nResultados salvos em: {output_path}")
    return all_results


if __name__ == "__main__":
    run_experiment()
