"""
HUGO â€” ExpansÃ£o 2 do Experimento 1.5
Mapeamento granular do espaÃ§o emocional primÃ¡rio

Novas configuraÃ§Ãµes homeostÃ¡ticas:
  1. FEAR_GRADIENT_1..4 â€” H2 decaindo em quatro passos (0.55 â†’ 0.10)
     Demonstra que a fronteira entre estados nÃ£o Ã© discreta â€”
     Ã© uma regiÃ£o de transiÃ§Ã£o com topologia prÃ³pria.

  2. CARE â€” H4 elevado, H1/H2 estÃ¡veis
     Panksepp: CARE como sistema primÃ¡rio distinto.
     Primeira captura de topologia de estado afiliativo.

  3. EXHAUSTION â€” todos os vetores levemente abaixo do nominal
     DepleÃ§Ã£o distribuÃ­da vs colapso especÃ­fico do PANIC (H1).
     Fadiga generalizada tem assinatura topolÃ³gica diferente.

  4. CONFLICT â€” H2 baixo + H3 alto simultaneamente
     AnÃ¡logo ao conflito motivacional â€” sistema puxado em
     direÃ§Ãµes incompatÃ­veis. Topologia de ambivalÃªncia.

Total: 9 configuraÃ§Ãµes originais + 8 novas = 9 configuraÃ§Ãµes
(4 gradientes FEAR + CARE + EXHAUSTION + CONFLICT + FEAR_SEEK original)

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
from src.homeostasis.homeostatic_field import HomeostaticField

# â”€â”€ ConfiguraÃ§Ã£o â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

N_TRIALS   = 50
N_STEPS    = 40
INPUT_DIM  = 64
SEQ_LEN    = 32
HIDDEN_DIM = 128
N_HEADS    = 4
N_LAYERS   = 3
SEED       = 42

RESULTS_DIR = ROOT / "results" / "exp_expansion_2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# â”€â”€ Novas configuraÃ§Ãµes homeostÃ¡ticas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# H = [H1_energia, H2_integridade, H3_carga, H4_temporal, H5_estabilidade]
# Nominais: [0.60, 0.70, 0.50, 0.60, 0.675]

HOMEOSTATIC_CONFIGS = {

    # â”€â”€ Gradiente FEAR â€” H2 decaindo em 4 passos â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Demonstra transiÃ§Ã£o contÃ­nua: ameaÃ§a leve â†’ ameaÃ§a severa
    # H1 fixo em 0.60 (energia neutra â€” sem transiÃ§Ã£o FEARâ†’SEEK)

    "FEAR_GRAD_1": {
        "description": "Ameaca leve (H2=0.55) â€” perturbacao minima de integridade",
        "initial_H": np.array([0.60, 0.55, 0.50, 0.60, 0.675]),
        "emotional_hypothesis": "PRE_FEAR / VIGILANCE",
    },
    "FEAR_GRAD_2": {
        "description": "Ameaca moderada (H2=0.40) â€” integridade comprometida",
        "initial_H": np.array([0.60, 0.40, 0.50, 0.60, 0.675]),
        "emotional_hypothesis": "FEAR_MILD",
    },
    "FEAR_GRAD_3": {
        "description": "Ameaca alta (H2=0.25) â€” replicacao H2_LOW original",
        "initial_H": np.array([0.60, 0.25, 0.50, 0.60, 0.675]),
        "emotional_hypothesis": "FEAR_ANALOG",
    },
    "FEAR_GRAD_4": {
        "description": "Ameaca extrema (H2=0.10) â€” colapso de integridade",
        "initial_H": np.array([0.60, 0.10, 0.50, 0.60, 0.675]),
        "emotional_hypothesis": "FEAR_EXTREME / FREEZE",
    },

    # â”€â”€ CARE / NURTURE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # H4 (consistÃªncia temporal) elevado â€” estado afiliativo estÃ¡vel
    # H1 e H2 nominais â€” sem ameaÃ§a, sem escassez de energia
    # HipÃ³tese: topologia de expansÃ£o suave, Ï„ alto, baixa volatilidade

    "CARE": {
        "description": "Estado afiliativo (H4 elevado) â€” CARE/NURTURE primario",
        "initial_H": np.array([0.65, 0.72, 0.48, 0.88, 0.68]),
        "emotional_hypothesis": "CARE_ANALOG",
    },

    # â”€â”€ EXHAUSTION â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Todos os vetores levemente abaixo do nominal â€” depleÃ§Ã£o distribuÃ­da
    # Diferente do PANIC: nÃ£o hÃ¡ colapso especÃ­fico, hÃ¡ declÃ­nio uniforme
    # HipÃ³tese: Ï„ mÃ©dio-baixo, baixa volatilidade Wasserstein (estagnaÃ§Ã£o difusa)

    "EXHAUSTION": {
        "description": "Deplecao distribuida â€” todos os vetores abaixo do nominal",
        "initial_H": np.array([0.38, 0.48, 0.32, 0.40, 0.42]),
        "emotional_hypothesis": "EXHAUSTION_ANALOG",
    },

    # â”€â”€ CONFLICT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # H2 baixo (ameaÃ§a) + H3 alto (sobrecarga) simultaneamente
    # AnÃ¡logo ao conflito motivacional â€” sistema puxado em direÃ§Ãµes opostas
    # H1 mÃ©dio â€” energia disponÃ­vel mas dividida
    # HipÃ³tese: topologia altamente instÃ¡vel, mÃ¡xima volatilidade Wasserstein

    "CONFLICT": {
        "description": "Conflito motivacional (H2 baixo + H3 alto) â€” ambivalencia",
        "initial_H": np.array([0.60, 0.22, 0.88, 0.60, 0.675]),
        "emotional_hypothesis": "CONFLICT_ANALOG",
    },

    # â”€â”€ FEAR_SEEK_TRANSITION (replicaÃ§Ã£o para comparaÃ§Ã£o) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Mantida do experimento original â€” H2 baixo + H1 alto
    # Garante comparabilidade entre os dois experimentos

    "FEAR_SEEK_TRANS": {
        "description": "Ameaca com energia alta (H2=0.25, H1=0.78) â€” transicao FEARâ†’SEEK",
        "initial_H": np.array([0.78, 0.25, 0.50, 0.60, 0.675]),
        "emotional_hypothesis": "FEAR_SEEK_TRANSITION",
    },
}

# â”€â”€ Input unificado â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def generate_unified_input(seq_len, input_dim, trial_seed):
    """Input gaussiano padrÃ£o â€” idÃªntico entre configuraÃ§Ãµes para o mesmo trial."""
    rng = np.random.RandomState(seed=trial_seed)
    return rng.randn(seq_len, input_dim)

# â”€â”€ Trial â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_trial(config_name, config, trial_idx):
    net = StructuralAttentionNetwork(
        input_dim=INPUT_DIM, seq_len=SEQ_LEN,
        hidden_dim=HIDDEN_DIM, n_heads=N_HEADS, n_layers=N_LAYERS,
        initial_homeostasis=config["initial_H"].copy(),
        seed=SEED + trial_idx
    )
    kappa = KappaMonitor(max_dim=1, threshold=1.5)

    trial_data = []
    labels_obs = []

    for step in range(N_STEPS):
        x = generate_unified_input(SEQ_LEN, INPUT_DIM,
                                   trial_seed=trial_idx * 1000 + step)
        result = net.forward(x)

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
            "step":            step,
            "betti_0":         kr["betti_0"],
            "betti_1":         kr["betti_1"],
            "entropy_A":       kr["entropy_A"],
            "wass_A":          kr["wass_A"],
            "score_betti_1":   kr["score_betti_1"],
            "entropy_S":       kr["entropy_S"],
            "wass_S":          kr["wass_S"],
            "tau":             kr["tau"],
            "L_t":             kr["L_t"],
            "H_state":         kr["H_state"],
            "emotional_label": result["emotional_label"],
            "topo_signature":  kr["topological_signature"],
        })

    ts = kappa.get_time_series()
    dominant = Counter(labels_obs).most_common(1)[0][0]

    return {
        "config":        config_name,
        "trial":         trial_idx,
        "dominant_label":dominant,
        "label_dist":    dict(Counter(labels_obs)),
        "steps":         trial_data,
        "summary": {
            "mean_betti_1":   float(np.mean(ts["betti_1"])),
            "mean_entropy_A": float(np.mean(ts["entropy_A"])),
            "std_entropy_A":  float(np.std(ts["entropy_A"])),
            "mean_wass_A":    float(np.mean(ts["wass_A"])),
            "mean_sb1":       float(np.mean(ts["score_betti_1"])),
            "mean_entropy_S": float(np.mean(ts["entropy_S"])),
            "mean_wass_S":    float(np.mean(ts["wass_S"])),
            "mean_tau":       float(np.mean(ts["tau"])),
            "min_tau":        float(np.min(ts["tau"])),
            "mean_L_t":       float(np.mean(ts["L_t"])),
            "final_H":        trial_data[-1]["H_state"],
        }
    }

# â”€â”€ AnÃ¡lise estatÃ­stica â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        all_vals = [v for g in groups for v in g]
        if np.std(all_vals) < 1e-10:
            analysis[metric] = {"error": "all_identical", "significant": False}
            continue
        stat, pval = stats.kruskal(*groups)
        analysis[metric] = {
            "kruskal_stat": round(float(stat), 4),
            "p_value":      round(float(pval), 10),
            "significant":  bool(pval < 0.05),
            "group_means":  {n: round(float(np.mean(g)), 5)
                             for n, g in zip(config_names, groups)},
        }
    return analysis

# â”€â”€ AnÃ¡lise do gradiente FEAR â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fear_gradient_analysis(all_results):
    """
    Analisa a monotonia do gradiente FEAR.
    Verifica se as mÃ©tricas variam monotonamente com H2 decrescente â€”
    evidÃªncia de que a fronteira emocional Ã© contÃ­nua, nÃ£o discreta.
    """
    fear_configs = ["FEAR_GRAD_1", "FEAR_GRAD_2", "FEAR_GRAD_3", "FEAR_GRAD_4"]
    h2_values = [0.55, 0.40, 0.25, 0.10]

    analysis = {}
    for metric in ["mean_tau", "mean_entropy_A", "mean_wass_S", "mean_L_t"]:
        means = []
        for cfg in fear_configs:
            if cfg in all_results:
                means.append(all_results[cfg]["aggregate"][metric])

        if len(means) == 4:
            # CorrelaÃ§Ã£o de Spearman com H2 â€” deve ser monÃ³tona
            corr, pval = stats.spearmanr(h2_values, means)
            analysis[metric] = {
                "h2_values":      h2_values,
                "metric_values":  [round(m, 5) for m in means],
                "spearman_r":     round(float(corr), 4),
                "p_value":        round(float(pval), 6),
                "monotonic":      bool(pval < 0.05),
            }
    return analysis

# â”€â”€ Principal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_expansion_2():
    print("=" * 70)
    print("HUGO - ExpansÃ£o 2 - Mapeamento Granular do EspaÃ§o Emocional")
    print("Gradiente FEAR | CARE | EXHAUSTION | CONFLICT")
    print("=" * 70)

    all_results = {}

    for config_name, config in HOMEOSTATIC_CONFIGS.items():
        hf      = HomeostaticField(initial_state=config["initial_H"].copy())
        tau_init = hf.temperature
        L_init   = hf.L

        print(f"\n[{config_name}]  hipÃ³tese: {config['emotional_hypothesis']}")
        print(f"  {config['description']}")
        print(f"  H={np.round(config['initial_H'], 2).tolist()}")
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
                mt  = np.mean([s["mean_tau"]        for s in summaries])
                mL  = np.mean([s["mean_L_t"]        for s in summaries])
                print(f"  Trial {trial+1:3d}/{N_TRIALS} | "
                      f"ent_A={me:.4f} ent_S={mes:.4f} "
                      f"tau={mt:.3f} L(t)={mL:.4f}")

        all_results[config_name] = {
            "config":              config["description"],
            "emotional_hypothesis":config["emotional_hypothesis"],
            "initial_H":           config["initial_H"].tolist(),
            "tau_initial":         tau_init,
            "L_initial":           L_init,
            "n_trials":            N_TRIALS,
            "trials":              trials,
            "aggregate":           {k: float(np.mean([s[k] for s in summaries]))
                                    for k in summaries[0].keys()
                                    if k != "final_H"},
            "dominant_labels":     dict(Counter([t["dominant_label"] for t in trials])),
        }

        agg = all_results[config_name]["aggregate"]
        print(f"\n  RESULTADO [{config_name}]:")
        print(f"    tau medio:      {agg['mean_tau']:.4f}  (min: {agg['min_tau']:.4f})")
        print(f"    L(t) medio:     {agg['mean_L_t']:.4f}")
        print(f"    entropia A(t):  {agg['mean_entropy_A']:.4f} +/- {agg['std_entropy_A']:.4f}")
        print(f"    entropia S(t):  {agg['mean_entropy_S']:.4f}")
        print(f"    wasserstein A:  {agg['mean_wass_A']:.5f}")
        print(f"    wasserstein S:  {agg['mean_wass_S']:.5f}")
        print(f"    rÃ³tulos:        {all_results[config_name]['dominant_labels']}")

    # â”€â”€ Tabela comparativa â”€â”€
    print("\n" + "=" * 70)
    print("COMPARAÃ‡ÃƒO â€” OITO CONFIGURAÃ‡Ã•ES")
    print(f"{'Config':<20} {'tau':>6} {'ent_A':>8} {'ent_S':>8} {'wass_S':>8} {'L(t)':>8}  hipÃ³tese")
    print("-" * 70)
    for name, data in all_results.items():
        a = data["aggregate"]
        hyp = data["emotional_hypothesis"]
        print(f"{name:<20} {a['mean_tau']:>6.3f} {a['mean_entropy_A']:>8.4f} "
              f"{a['mean_entropy_S']:>8.4f} {a['mean_wass_S']:>8.5f} "
              f"{a['mean_L_t']:>8.4f}  {hyp}")

    # â”€â”€ AnÃ¡lise estatÃ­stica global â”€â”€
    print("\n" + "=" * 70)
    print("DISTINGUIBILIDADE ESTATÃSTICA (Kruskal-Wallis â€” todas configs)")
    print("=" * 70)
    stat_results = statistical_analysis(all_results)
    for metric, res in stat_results.items():
        if "error" in res:
            print(f"\n  [{metric}] â€” variÃ¢ncia nula")
            continue
        sig = "*** SIGNIFICATIVO ***" if res["significant"] else "nao significativo"
        print(f"\n  [{metric}]  H={res['kruskal_stat']}  p={res['p_value']:.10f}  {sig}")
        for cname, mean in res["group_means"].items():
            print(f"      {cname:<22}: {mean:.5f}")

    # â”€â”€ AnÃ¡lise do gradiente FEAR â”€â”€
    print("\n" + "=" * 70)
    print("ANÃLISE DO GRADIENTE FEAR â€” Monotonia de H2")
    print("Verifica se emoÃ§Ã£o Ã© contÃ­nua (monÃ³tona) ou discreta (por categorias)")
    print("=" * 70)
    fg_analysis = fear_gradient_analysis(all_results)
    for metric, res in fg_analysis.items():
        mono = "MONÃ“TONA âœ“" if res["monotonic"] else "nao monÃ³tona"
        print(f"\n  [{metric}]  Spearman r={res['spearman_r']}  "
              f"p={res['p_value']}  â†’ {mono}")
        for h2, val in zip(res["h2_values"], res["metric_values"]):
            print(f"      H2={h2:.2f}: {val:.5f}")

    # â”€â”€ Salva â”€â”€
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
            "config":               data["config"],
            "emotional_hypothesis": data["emotional_hypothesis"],
            "initial_H":            data["initial_H"],
            "tau_initial":          data["tau_initial"],
            "L_initial":            data["L_initial"],
            "n_trials":             data["n_trials"],
            "aggregate":            data["aggregate"],
            "dominant_labels":      data["dominant_labels"],
            "trial_summaries":      [t["summary"] for t in data["trials"]],
        }
    out["statistical_analysis"]    = stat_results
    out["fear_gradient_analysis"]  = fg_analysis

    output_path = RESULTS_DIR / "exp_expansion_2_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean(out), f, indent=2)

    print(f"\nResultados salvos em: {output_path}")
    return all_results


if __name__ == "__main__":
    run_expansion_2()
