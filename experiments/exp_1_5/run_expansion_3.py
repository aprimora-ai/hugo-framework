"""
HUGO — Expansão 3: Trajetória Topológica Temporal
Captura step-a-step da evolução dos invariantes topológicos

Enquanto o Exp. 1.5 e Expansão 2 mediram FOTOS (médias sobre 40 steps),
a Expansão 3 captura o FILME — como a topologia evolui durante o processamento.

Para cada trial, cada step produz um vetor de 5 invariantes:
  [tau, L_t, entropy_A, entropy_S, wass_S]

Esse vetor no tempo forma uma TRAJETÓRIA no espaço R^5.
A topologia dessa trajetória é a assinatura dinâmica do estado emocional.

Hipóteses:
  - Configurações distintas produzem trajetórias com formas distintas
  - FEAR converge rapidamente a um atrator (trajetória estável)
  - CONFLICT oscila sem convergir (trajetória instável)
  - CARE expande progressivamente (trajetória aberta)
  - EXHAUSTION colapsa e estagna (trajetória de declínio)

O vetor de trajetória também é o input natural para o Radiante Estrutural:
cada step é um ponto em 5 dimensões, a trajetória completa é o objeto.

Autor: David Ohio | odavidohio@gmail.com
"""

import sys
import numpy as np
import json
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist
from collections import Counter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.network.gray_box_network import StructuralAttentionNetwork
from src.kappa.kappa_monitor import KappaMonitor
from src.homeostasis.homeostatic_field import HomeostaticField

# ── Configuração ──────────────────────────────────────────────────────────

N_TRIALS   = 30       # menos trials — mais steps por trial
N_STEPS    = 80       # mais steps — trajetória mais rica
INPUT_DIM  = 64
SEQ_LEN    = 32
HIDDEN_DIM = 128
N_HEADS    = 4
N_LAYERS   = 3
SEED       = 42

RESULTS_DIR = ROOT / "results" / "exp_expansion_3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Configurações selecionadas ────────────────────────────────────────────
# Subconjunto das mais contrastantes para análise de trajetória

HOMEOSTATIC_CONFIGS = {
    "BASELINE": {
        "description": "Equilibrio nominal",
        "initial_H": np.array([0.60, 0.70, 0.50, 0.60, 0.675]),
        "emotional_hypothesis": "SEEK_ANALOG",
        "color": "#4C72B0",
    },
    "FEAR": {
        "description": "Ameaca alta — H2=0.25",
        "initial_H": np.array([0.60, 0.25, 0.50, 0.60, 0.675]),
        "emotional_hypothesis": "FEAR_ANALOG",
        "color": "#DD2222",
    },
    "CARE": {
        "description": "Estado afiliativo — H4 elevado",
        "initial_H": np.array([0.65, 0.72, 0.48, 0.88, 0.68]),
        "emotional_hypothesis": "CARE_ANALOG",
        "color": "#22AA66",
    },
    "EXHAUSTION": {
        "description": "Deplecao distribuida",
        "initial_H": np.array([0.38, 0.48, 0.32, 0.40, 0.42]),
        "emotional_hypothesis": "EXHAUSTION_ANALOG",
        "color": "#8B5CF6",
    },
    "CONFLICT": {
        "description": "Conflito motivacional",
        "initial_H": np.array([0.60, 0.22, 0.88, 0.60, 0.675]),
        "emotional_hypothesis": "CONFLICT_ANALOG",
        "color": "#E8A020",
    },
    "FEAR_SEEK_TRANS": {
        "description": "Transicao FEAR-SEEK",
        "initial_H": np.array([0.78, 0.25, 0.50, 0.60, 0.675]),
        "emotional_hypothesis": "FEAR_SEEK_TRANSITION",
        "color": "#009999",
    },
}

# ── Métricas de trajetória ────────────────────────────────────────────────

def trajectory_metrics(traj: np.ndarray) -> dict:
    """
    Calcula métricas de forma da trajetória no espaço R^5.

    traj: array (N_STEPS, 5) — cada linha é [tau, L_t, ent_A, ent_S, wass_S]

    Métricas:
      - convergencia:   variância dos últimos 20 steps vs primeiros 20 steps
                        < 1.0: trajetória converge (atrator)
                        > 1.0: trajetória diverge ou oscila

      - velocidade_ini: distância média entre steps consecutivos nos primeiros 10 steps
      - velocidade_fin: distância média entre steps consecutivos nos últimos 10 steps
      - razao_vel:      velocidade_ini / velocidade_fin
                        > 1.0: sistema desacelera (convergindo)
                        < 1.0: sistema acelera (divergindo)

      - comprimento:    comprimento total da trajetória (soma das distâncias)

      - amplitude:      max - min por dimensão (range de exploração)

      - estabilidade:   1 / (variância da trajetória inteira)
                        alta: trajetória estável (FEAR, CARE)
                        baixa: trajetória errática (CONFLICT)

      - deriva:         distância entre centróide dos primeiros 20 e últimos 20 steps
                        alta: sistema muda de região durante o processamento
                        baixa: sistema permanece na mesma região

      - entropia_traj:  entropia de Shannon sobre distribuição de distâncias entre steps
                        alta: passos variados (exploração)
                        baixa: passos uniformes (convergido)
    """
    T = len(traj)
    mid = T // 2

    # Convergência: var final / var inicial
    var_ini = np.var(traj[:20], axis=0).mean()
    var_fin = np.var(traj[-20:], axis=0).mean()
    convergencia = float(var_fin / (var_ini + 1e-10))

    # Velocidade
    dists = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    vel_ini = float(dists[:10].mean())
    vel_fin = float(dists[-10:].mean())
    razao_vel = float(vel_ini / (vel_fin + 1e-10))

    # Comprimento total
    comprimento = float(dists.sum())

    # Amplitude por dimensão
    amplitude = (traj.max(axis=0) - traj.min(axis=0)).tolist()

    # Estabilidade
    estabilidade = float(1.0 / (np.var(traj) + 1e-10))

    # Deriva: centróide primeiros 20 → últimos 20
    centro_ini = traj[:20].mean(axis=0)
    centro_fin = traj[-20:].mean(axis=0)
    deriva = float(np.linalg.norm(centro_fin - centro_ini))

    # Entropia da trajetória
    hist, _ = np.histogram(dists, bins=20, density=True)
    hist = hist + 1e-10
    hist /= hist.sum()
    entropia_traj = float(-np.sum(hist * np.log(hist + 1e-10)))

    return {
        "convergencia":  round(convergencia, 5),
        "vel_ini":       round(vel_ini,       6),
        "vel_fin":       round(vel_fin,       6),
        "razao_vel":     round(razao_vel,     4),
        "comprimento":   round(comprimento,   5),
        "amplitude":     [round(a, 5) for a in amplitude],
        "estabilidade":  round(estabilidade,  3),
        "deriva":        round(deriva,        5),
        "entropia_traj": round(entropia_traj, 5),
    }

# ── Input unificado ───────────────────────────────────────────────────────

def generate_unified_input(seq_len, input_dim, trial_seed):
    rng = np.random.RandomState(seed=trial_seed)
    return rng.randn(seq_len, input_dim)

# ── Trial com trajetória completa ─────────────────────────────────────────

def run_trial_trajectory(config_name, config, trial_idx):
    net = StructuralAttentionNetwork(
        input_dim=INPUT_DIM, seq_len=SEQ_LEN,
        hidden_dim=HIDDEN_DIM, n_heads=N_HEADS, n_layers=N_LAYERS,
        initial_homeostasis=config["initial_H"].copy(),
        seed=SEED + trial_idx
    )
    kappa = KappaMonitor(max_dim=1, threshold=1.5)

    # Trajetória: (N_STEPS, 5) — cada step é um ponto no espaço R^5
    trajectory    = np.zeros((N_STEPS, 5))
    step_data     = []
    labels_obs    = []

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

        # Vetor de trajetória: [tau, L_t, entropy_A, entropy_S, wass_S]
        trajectory[step] = [
            kr["tau"],
            kr["L_t"],
            kr["entropy_A"],
            kr["entropy_S"],
            kr["wass_S"],
        ]

        labels_obs.append(result["emotional_label"])
        step_data.append({
            "step":   step,
            "tau":    kr["tau"],
            "L_t":    kr["L_t"],
            "ent_A":  kr["entropy_A"],
            "ent_S":  kr["entropy_S"],
            "wass_S": kr["wass_S"],
            "label":  result["emotional_label"],
        })

    # Métricas de forma da trajetória
    traj_metrics = trajectory_metrics(trajectory)
    dominant = Counter(labels_obs).most_common(1)[0][0]

    return {
        "config":         config_name,
        "trial":          trial_idx,
        "dominant_label": dominant,
        "label_dist":     dict(Counter(labels_obs)),
        "trajectory":     trajectory.tolist(),   # (N_STEPS, 5) — para o Radiante
        "step_data":      step_data,
        "traj_metrics":   traj_metrics,
    }

# ── Análise estatística das métricas de trajetória ───────────────────────

def statistical_analysis(all_results):
    traj_metric_names = [
        "convergencia", "razao_vel", "comprimento",
        "estabilidade", "deriva", "entropia_traj",
    ]
    config_names = list(all_results.keys())
    analysis = {}

    for metric in traj_metric_names:
        groups = [
            [t["traj_metrics"][metric] for t in all_results[n]["trials"]]
            for n in config_names
        ]
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

# ── Principal ─────────────────────────────────────────────────────────────

def run_expansion_3():
    print("=" * 70)
    print("HUGO - Expansão 3 - Trajetórias Topológicas Temporais")
    print(f"N_TRIALS={N_TRIALS} N_STEPS={N_STEPS} | 5 dimensões | 6 configs")
    print("=" * 70)

    all_results = {}

    for config_name, config in HOMEOSTATIC_CONFIGS.items():
        hf       = HomeostaticField(initial_state=config["initial_H"].copy())
        tau_init = hf.temperature
        L_init   = hf.L

        print(f"\n[{config_name}]  {config['emotional_hypothesis']}")
        print(f"  H={np.round(config['initial_H'], 2).tolist()}")
        print(f"  tau_0={tau_init:.4f}  L_0={L_init:.4f}")

        trials = []

        for trial in range(N_TRIALS):
            result = run_trial_trajectory(config_name, config, trial)
            trials.append(result)

            if (trial + 1) % 10 == 0:
                # Médias das métricas de trajetória até agora
                conv = np.mean([t["traj_metrics"]["convergencia"] for t in trials])
                rvel = np.mean([t["traj_metrics"]["razao_vel"]    for t in trials])
                est  = np.mean([t["traj_metrics"]["estabilidade"] for t in trials])
                der  = np.mean([t["traj_metrics"]["deriva"]       for t in trials])
                print(f"  Trial {trial+1:2d}/{N_TRIALS} | "
                      f"conv={conv:.3f}  razao_vel={rvel:.3f}  "
                      f"estab={est:.1f}  deriva={der:.4f}")

        # Agrega métricas de trajetória
        traj_agg = {}
        for key in trials[0]["traj_metrics"].keys():
            if key == "amplitude":
                traj_agg[key] = [
                    round(float(np.mean([t["traj_metrics"][key][i]
                                         for t in trials])), 5)
                    for i in range(5)
                ]
            else:
                traj_agg[key] = round(float(
                    np.mean([t["traj_metrics"][key] for t in trials])
                ), 5)

        all_results[config_name] = {
            "config":               config["description"],
            "emotional_hypothesis": config["emotional_hypothesis"],
            "initial_H":            config["initial_H"].tolist(),
            "tau_initial":          tau_init,
            "L_initial":            L_init,
            "n_trials":             N_TRIALS,
            "n_steps":              N_STEPS,
            "trials":               trials,
            "traj_agg":             traj_agg,
            "dominant_labels":      dict(Counter([t["dominant_label"]
                                                  for t in trials])),
        }

        a = traj_agg
        print(f"\n  TRAJETÓRIA [{config_name}]:")
        print(f"    convergencia:   {a['convergencia']:.5f}  "
              f"({'converge' if a['convergencia'] < 0.8 else 'oscila/diverge'})")
        print(f"    razao_vel:      {a['razao_vel']:.4f}  "
              f"({'desacelera' if a['razao_vel'] > 1.0 else 'acelera'})")
        print(f"    comprimento:    {a['comprimento']:.5f}")
        print(f"    estabilidade:   {a['estabilidade']:.3f}")
        print(f"    deriva:         {a['deriva']:.5f}")
        print(f"    entropia_traj:  {a['entropia_traj']:.5f}")

    # ── Tabela comparativa ──
    print("\n" + "=" * 70)
    print("COMPARAÇÃO — MÉTRICAS DE TRAJETÓRIA")
    print(f"{'Config':<20} {'conv':>7} {'r_vel':>7} {'comp':>8} "
          f"{'estab':>8} {'deriva':>8} {'ent_tj':>8}  hipótese")
    print("-" * 70)
    for name, data in all_results.items():
        a = data["traj_agg"]
        hyp = data["emotional_hypothesis"]
        print(f"{name:<20} {a['convergencia']:>7.4f} {a['razao_vel']:>7.4f} "
              f"{a['comprimento']:>8.5f} {a['estabilidade']:>8.2f} "
              f"{a['deriva']:>8.5f} {a['entropia_traj']:>8.5f}  {hyp}")

    # ── Análise estatística ──
    print("\n" + "=" * 70)
    print("DISTINGUIBILIDADE (Kruskal-Wallis — métricas de trajetória)")
    print("=" * 70)
    stat_results = statistical_analysis(all_results)
    for metric, res in stat_results.items():
        if "error" in res:
            print(f"\n  [{metric}] — variância nula")
            continue
        sig = "*** SIGNIFICATIVO ***" if res["significant"] else "nao significativo"
        print(f"\n  [{metric}]  H={res['kruskal_stat']}  "
              f"p={res['p_value']:.8f}  {sig}")
        for cname, mean in res["group_means"].items():
            print(f"      {cname:<22}: {mean:.5f}")

    # ── Exporta trajetórias para o Radiante ──
    radiante_export = {}
    for name, data in all_results.items():
        # Para o Radiante: trajetória média (média das trajetórias dos trials)
        all_traj = np.array([t["trajectory"] for t in data["trials"]])
        mean_traj = all_traj.mean(axis=0).tolist()  # (N_STEPS, 5)

        radiante_export[name] = {
            "emotional_hypothesis": data["emotional_hypothesis"],
            "initial_H":            data["initial_H"],
            "color":                HOMEOSTATIC_CONFIGS[name]["color"],
            "mean_trajectory":      mean_traj,     # (N_STEPS, 5)
            "traj_agg":             data["traj_agg"],
            "n_steps":              N_STEPS,
            "dimensions": [
                "tau",
                "L_t",
                "entropy_A",
                "entropy_S",
                "wass_S",
            ],
        }

    # ── Salva ──
    def clean(obj):
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict):        return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):        return [clean(i) for i in obj]
        return obj

    # Resultados completos (sem trajetórias step-a-step para não explodir o JSON)
    out = {}
    for name, data in all_results.items():
        out[name] = {
            "config":               data["config"],
            "emotional_hypothesis": data["emotional_hypothesis"],
            "initial_H":            data["initial_H"],
            "tau_initial":          data["tau_initial"],
            "L_initial":            data["L_initial"],
            "n_trials":             data["n_trials"],
            "n_steps":              data["n_steps"],
            "traj_agg":             data["traj_agg"],
            "dominant_labels":      data["dominant_labels"],
            # Guarda só as métricas de trajetória por trial (não os steps)
            "trial_traj_metrics": [t["traj_metrics"] for t in data["trials"]],
        }
    out["statistical_analysis"] = stat_results

    path_results = RESULTS_DIR / "exp_expansion_3_results.json"
    with open(path_results, "w", encoding="utf-8") as f:
        json.dump(clean(out), f, indent=2)
    print(f"\nResultados salvos em: {path_results}")

    # Export para o Radiante (trajetórias médias)
    path_radiante = RESULTS_DIR / "radiante_trajectories.json"
    with open(path_radiante, "w", encoding="utf-8") as f:
        json.dump(clean(radiante_export), f, indent=2)
    print(f"Trajetórias para Radiante: {path_radiante}")

    return all_results


if __name__ == "__main__":
    run_expansion_3()
