# HUGO — Homological Unified Gradient Ontology

**HUGO AGI Framework — Paper I**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://zenodo.org/badge/1178433785.svg)](https://doi.org/10.5281/zenodo.18947852)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![TDA](https://img.shields.io/badge/TDA-Persistent%20Homology-purple.svg)](https://gudhi.inria.fr/)
[![HUGO Series](https://img.shields.io/badge/HUGO%20Series-Paper%20I-blueviolet.svg)](https://github.com/aprimora-ai/hugo-framework)
[![Kappa Method](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18883639-blue.svg)](https://doi.org/10.5281/zenodo.18883639)

**Author:** David Ohio | odavidohio@gmail.com
**Version:** 2.0.0 — March 2026
**Repository:** https://github.com/aprimora-ai/hugo-framework

> *This repository accompanies Paper I of the HUGO AGI Framework series.*

---

## Overview

HUGO is a mathematical framework proposing a unified topological description of affective regulation and cognitive organization in information-processing systems.

The **Kappa Method** is the central mathematical instrument — persistent homology applied to attention distributions.

This repository accompanies the paper:

> Ohio, D. (2026). *Homeostatic Temperature and Primary Emotions as Topological Signatures in Neural Information Flow*. HUGO Series Paper I v2. Zenodo. https://doi.org/10.5281/zenodo.18947852

---

## Version 2.0.0 — Changes from v1

**HomeostaticField v3.0** — Three architectural changes fix the saturation instability found in v1:

1. **Set-point model.** Each vector oscillates around its initial condition (H_i*), not the nominal center. Identity encoded in set points; experience in oscillations.
2. **Orthogonal projections.** Each of 5 channels receives a fixed orthogonal projection (Gram-Schmidt), creating 5 independent feedback pathways H→τ→A→r→H.
3. **Ensemble-calibrated decay.** Decay rate derived from perturbation statistics during 30-step burn-in (CALM analog). Not a hyperparameter — derived from data.

**Paper I v2** — All experimental results re-run with v3.0 dynamics. Key change: convergence metric is NOT significant in v2 (genuine negative), whereas in v1 it was spuriously significant due to saturation.

**33 tests passing** — Including set_points_preserved, ensemble_calibration_runs, differentiation_persists_200_steps.

---

## Repository Structure

```
HUGO/
├── src/
│   ├── network/                # Gray-box structural attention network
│   │   └── gray_box_network.py
│   ├── homeostasis/            # Homeostatic field H(t) v3.0
│   │   └── homeostatic_field.py
│   ├── kappa/                  # Kappa monitor (persistent homology via ripser)
│   │   └── kappa_monitor.py
│   └── echo/                   # [Paper II — see ECHO repository]
├── experiments/
│   └── exp_1_5/                # All three experimental series
├── results/
│   ├── exp_1_5/
│   ├── exp_expansion_2/
│   └── exp_expansion_3/
├── paper/                      # Paper I v2 (Markdown + PDF)
├── tests/                      # 33 unit tests
│   └── test_hugo.py
├── LICENSE
└── README.md
```

---

## Experiments

### Experiment 1.5 — Baseline
5 homeostatic configurations, 50 trials, 40 steps.
Result: p = 0 across six independent metrics (Kruskal-Wallis H = 107–239).
tau spread = 0.233, 4 distinct emotional labels.

### Expansion 2 — Continuous Gradient and Novel States
8 configurations. FEAR as a continuous topological field (Spearman r = ±1.0 on 4 metrics).
States CARE, EXHAUSTION, CONFLICT, FEAR-SEEK-TRANSITION discriminated with p = 0.

### Expansion 3 — Temporal Trajectories
6 configurations, 30 trials, 80 steps.
4/6 trajectory metrics significant. Convergence and velocity ratio NOT significant (genuine negative — universal dissipative dynamic). Trajectory geometry carries emotional identity.

---

## Architecture

**Homeostatic field H(t) ∈ ℝ⁵ (v3.0):**

| Vector | Biological Analog | Decay |
|--------|-------------------|-------|
| H1 — Energy Coherence | Glucose | Ensemble-calibrated |
| H2 — Structural Integrity | Physical integrity | Ensemble-calibrated |
| H3 — Load Balance | Osmotic pressure | Ensemble-calibrated |
| H4 — Temporal Consistency | Circadian cycle | Ensemble-calibrated |
| H5 — Representational Stability | Internal temperature | Ensemble-calibrated |

Calibrated decay rates (typical): 0.003–0.01 per channel, derived from burn-in statistics.

**Homeostatic temperature:**
```
τ(H(t)) = τ_base / (1 + 4·L(t))
where L(t) = Σ wᵢ·δHᵢ(t)
```

**Set-point model:**
```
restoring_force_i = -d_i · (H_i(t) - H_i*)
where H_i* = initial condition (set point)
      d_i  = ensemble-calibrated decay rate
```

**Pre-Softmax Affective Intensity Principle:**
Softmax normalization suppresses affective intensity. HUGO monitors both S(t) (pre-softmax) and A(t) (post-softmax) to preserve the structural magnitude of homeostatic pressure before probabilistic compression.

---

## Requirements

```
Python 3.13
numpy, scipy
ripser, persim (persistent homology)
matplotlib, seaborn (visualization)
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run tests:
```bash
python tests/test_hugo.py
```

---

## Citation

```bibtex
@misc{ohio2026hugo,
  author    = {Ohio, David},
  title     = {{HUGO}: Homological Unified Gradient Ontology ---
               Homeostatic Temperature and Primary Emotions as
               Topological Signatures in Neural Information Flow},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18947852},
  url       = {https://doi.org/10.5281/zenodo.18947852}
}
```

---

## HUGO AGI Framework — Module Series

| Module | Paper | Function | Status |
|--------|-------|----------|--------|
| **HUGO** | Paper I | Primary affective states via homeostatic field | ✅ Published |
| **ECHO** | Paper II | Language-mediated empathic coupling | ✅ [Published](https://doi.org/10.5281/zenodo.19043115) |
| **REMIND** | Paper III | Memory and structural accumulation | ✅ [Published](https://doi.org/10.5281/zenodo.19052403) |
| RHEO | Paper IV | Temporal flow regulation | In development |
| ANIMA | Paper V | Phenomenological animation | In development |
| SELF | Paper VI | Reflexive self-modeling | In development |

---

## Related Work

- **ECHO Paper II:** [DOI: 10.5281/zenodo.19043115](https://doi.org/10.5281/zenodo.19043115)
- **ECHO Repository:** [github.com/aprimora-ai/hugo-framework-echo](https://github.com/aprimora-ai/hugo-framework-echo)
- **REMIND Paper III:** [DOI: 10.5281/zenodo.19052403](https://doi.org/10.5281/zenodo.19052403)
- **REMIND Repository:** [github.com/aprimora-ai/hugo-framework-remind](https://github.com/aprimora-ai/hugo-framework-remind)
- **Kappa Method:** [DOI: 10.5281/zenodo.18883639](https://doi.org/10.5281/zenodo.18883639)
- **Kappa-Radiante:** [DOI: 10.5281/zenodo.18940478](https://doi.org/10.5281/zenodo.18940478)
- **Kappa-LLM / HEIMDALL:** [DOI: 10.5281/zenodo.18883790](https://doi.org/10.5281/zenodo.18883790)

---

**David Ohio** | Independent Researcher | odavidohio@gmail.com
