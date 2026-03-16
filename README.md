# HUGO — Homological Unified Gradient Ontology

**HUGO AGI Framework — Paper I**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://zenodo.org/badge/1178433785.svg)](https://doi.org/10.5281/zenodo.18947852)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![TDA](https://img.shields.io/badge/TDA-Persistent%20Homology-purple.svg)](https://gudhi.inria.fr/)
[![HUGO Series](https://img.shields.io/badge/HUGO%20Series-Paper%20I-blueviolet.svg)](https://github.com/aprimora-ai/hugo-framework)
[![Kappa Method](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18883639-blue.svg)](https://doi.org/10.5281/zenodo.18883639)

**Author:** David Ohio | odavidohio@gmail.com
**Version:** 1.0.1 — March 2026
**Repository:** https://github.com/aprimora-ai/hugo-framework

> *This repository accompanies Paper I of the HUGO AGI Framework series.*

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
├── src/
│   ├── network/                # Gray-box structural attention network
│   │   └── gray_box_network.py
│   ├── homeostasis/            # Homeostatic field H(t)
│   │   └── homeostatic_field.py
│   ├── kappa/                  # Kappa monitor (persistent homology via ripser)
│   │   └── kappa_monitor.py
│   └── echo/                   # [Paper II — in development]
├── experiments/
│   └── exp_1_5/                # All three experimental series
├── results/
│   ├── exp_1_5/
│   ├── exp_expansion_2/
│   └── exp_expansion_3/
├── paper/                      # LaTeX source + compiled PDF
├── tests/                      # Unit tests
│   └── test_hugo.py
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
| **ECHO** | Paper II | Temporal resonance and empathic coupling | In development |
| **REMIND** | Paper III | Episodic memory anchoring | In development |
| **RHEO** | Paper IV | Temporal flow regulation and processing intensity | In development |
| **ANIMA** | Paper V | Phenomenological animation of internal states | In development |
| **SELF** | Paper VI | Reflexive self-modeling and identity dynamics | [Repository](https://github.com/aprimora-ai/self-framework) |

---

## Related Work

- **Kappa Method:** https://doi.org/10.5281/zenodo.18883639
- **Kappa-Radiante:** https://doi.org/10.5281/zenodo.18940478
- **Kappa-LLM / HEIMDALL:** https://doi.org/10.5281/zenodo.18883790
- **Kappa-FIN:** https://doi.org/10.5281/zenodo.18883585
