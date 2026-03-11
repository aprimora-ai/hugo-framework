# HUGO — Homological Unified Gradient Ontology

**HUGO Series — Paper I**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://zenodo.org/badge/1178433785.svg)](https://doi.org/10.5281/zenodo.18947852)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-ee4c2c.svg)](https://pytorch.org/)
[![TDA](https://img.shields.io/badge/TDA-Persistent%20Homology-purple.svg)](https://gudhi.inria.fr/)
[![HUGO Series](https://img.shields.io/badge/HUGO%20Series-Paper%20I-blueviolet.svg)](https://github.com/aprimora-ai/hugo-framework)
[![Kappa Method](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18883639-blue.svg)](https://doi.org/10.5281/zenodo.18883639)

**Author:** David Ohio | odavidohio@gmail.com
**Version:** 1.0.0 — March 2026
**Repository:** https://github.com/aprimora-ai/hugo-framework

> *This repository accompanies Paper I of the HUGO series.*
> Paper II — Language-Mediated Empathic Coupling (ECHO) | Paper III — Persistent Topological Memory

---

## Overview

HUGO is a mathematical framework proposing a unified topological description of affective regulation and cognitive organization in information-processing systems.

The **Kappa Method** is the central mathematical instrument — persistent homology applied to attention distributions.

This repository accompanies the paper:

> Ohio, D. (2026). *Homeostatic Temperature and Primary Emotions as Topological Signatures in Neural Information Flow*. HUGO Series Paper I. Zenodo. https://doi.org/10.5281/zenodo.18947852

---

## Repository Structure

```
HUGO/
├── src/                        # Core source code
│   ├── network/                # Gray-box structural attention network
│   ├── homeostasis/            # Homeostatic field H(t)
│   └── kappa/                  # Kappa monitor (persistent homology)
├── experiments/
│   └── exp_1_5/                # All three experimental series
│       ├── run_experiment.py           # Experiment 1.5 (baseline)
│       ├── run_expansion_2.py          # Expansion 2 (8 configurations)
│       ├── run_expansion_3.py          # Expansion 3 (temporal trajectories)
│       ├── generate_figures.py
│       ├── generate_figures_expansion2.py
│       └── generate_figures_expansion3.py
├── results/
│   ├── exp_1_5/                # Results + figures Exp. 1.5
│   ├── exp_expansion_2/        # Results + figures Expansion 2
│   └── exp_expansion_3/        # Results + figures Expansion 3
├── paper/                      # LaTeX source + compiled PDF
├── data/
├── notebooks/
├── LICENSE
└── README.md
```

---

## Experiments

### Experiment 1.5 — Baseline
5 homeostatic configurations, 50 trials, 40 steps.
Result: p < 10⁻⁸ across six independent metrics (Kruskal-Wallis).

### Expansion 2 — Continuous Gradient and Novel States
8 configurations. FEAR as a continuous topological field (Spearman r = ±1.0).
States CARE, EXHAUSTION, CONFLICT, FEAR-SEEK-TRANSITION discriminated with p = 0.

### Expansion 3 — Temporal Trajectories
6 configurations, 30 trials, 80 steps.
Distinct trajectory topologies: attractors (BASELINE, EXHAUSTION), divergence (CONFLICT), exploration (CARE).

---

## Architecture

**Homeostatic field H(t) ∈ ℝ⁵:**

| Vector | Biological Analog | decay_rate |
|--------|-------------------|------------|
| H1 — Energy Coherence | Glucose | 0.002 |
| H2 — Structural Integrity | Physical integrity | 0.001 |
| H3 — Load Balance | Osmotic pressure | 0.002 |
| H4 — Temporal Consistency | Circadian cycle | 0.001 |
| H5 — Representational Stability | Internal temperature | 0.002 |

**Homeostatic temperature:**
```
τ(H(t)) = τ_base / (1 + 4·L(t))
where L(t) = Σ wᵢ·δHᵢ(t)
```

**Pre-Softmax Affective Intensity Principle:**
Softmax normalization suppresses affective intensity. HUGO monitors both S(t) (pre-softmax) and A(t) (post-softmax) to preserve the structural magnitude of homeostatic pressure before probabilistic compression.

---

## Requirements

```
Python 3.13
PyTorch 2.7.1+cu118
GUDHI 3.11
Ripser 0.6.14
Persim 0.3.8
```

Install dependencies:
```bash
pip install -r requirements.txt
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

## HUGO Series

| Paper | Title | Status |
|-------|-------|--------|
| **Paper I** | Internal Affective Geometry (this repository) | ✅ Published |
| Paper II | Language-Mediated Empathic Coupling (ECHO) | In development |
| Paper III | Persistent Topological Memory | In development |

---

## Related Work

- **Kappa Method:** https://doi.org/10.5281/zenodo.18883639
- **Kappa-FIN:** https://doi.org/10.5281/zenodo.18883585
