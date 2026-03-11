# HUGO — Homological Unified Gradient Ontology

**HUGO Series — Paper I**

**Autor:** David Ohio | odavidohio@gmail.com
**Versão:** 1.0.0 — Março 2026
**Licença:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
**Repositório:** https://github.com/aprimora-ai/hugo-framework

> *This repository is Part I of the HUGO series.*
> Paper II — Language-Mediated Empathic Coupling (ECHO) | Paper III — Persistent Topological Memory

---

## Visão Geral

HUGO é um framework matemático que propõe uma descrição topológica unificada para regulação afetiva e organização cognitiva em sistemas de processamento de informação.

O **Kappa Method** é o instrumento matemático central — homologia persistente aplicada a distribuições de atenção.

Este repositório acompanha o paper:

> Ohio, D. (2026). *Homeostatic Temperature and Primary Emotions as Topological Signatures in Neural Information Flow*. HUGO Series Paper I. Zenodo.

---

## Estrutura do Projeto

```
HUGO/
├── src/                        # Código-fonte principal
│   ├── network/                # Rede neural gray-box
│   ├── homeostasis/            # Campo homeostático H(t)
│   └── kappa/                  # Monitor Kappa (homologia persistente)
├── experiments/
│   └── exp_1_5/                # Scripts dos três experimentos
│       ├── run_experiment.py           # Experimento 1.5 (baseline)
│       ├── run_expansion_2.py          # Expansão 2 (8 configurações)
│       ├── run_expansion_3.py          # Expansão 3 (trajetórias temporais)
│       ├── generate_figures.py
│       ├── generate_figures_expansion2.py
│       └── generate_figures_expansion3.py
├── results/
│   ├── exp_1_5/                # Resultados + figuras Exp. 1.5
│   ├── exp_expansion_2/        # Resultados + figuras Expansão 2
│   └── exp_expansion_3/        # Resultados + figuras Expansão 3
├── paper/                      # Fonte LaTeX do paper
├── data/
├── notebooks/
├── LICENSE
└── README.md
```

---

## Experimentos

### Experimento 1.5 — Baseline
5 configurações homeostáticas, 50 trials, 40 steps.
Resultado: p < 10⁻⁸ em seis métricas independentes (Kruskal-Wallis).

### Expansão 2 — Gradiente Contínuo e Estados Novos
8 configurações. FEAR como campo contínuo (Spearman r = ±1.0).
Estados CARE, EXHAUSTION, CONFLICT, FEAR-SEEK-TRANSITION discriminados com p = 0.

### Expansão 3 — Trajetórias Temporais
6 configurações, 30 trials, 80 steps.
Topologias de trajetória distintas: atratores (BASELINE, EXHAUSTION), divergência (CONFLICT), exploração (CARE).

---

## Arquitetura

**Campo homeostático H(t) ∈ ℝ⁵:**

| Vetor | Análogo Biológico | decay_rate |
|-------|-------------------|------------|
| H1 — Coerência Energética | Glicose | 0.002 |
| H2 — Integridade Estrutural | Integridade física | 0.001 |
| H3 — Equilíbrio de Carga | Pressão osmótica | 0.002 |
| H4 — Consistência Temporal | Ciclo circadiano | 0.001 |
| H5 — Estabilidade de Representação | Temperatura interna | 0.002 |

**Temperatura homeostática:**
τ(H(t)) = τ_base / (1 + 4·L(t))
onde L(t) = Σ wᵢ·δHᵢ(t)

**Pre-Softmax Affective Intensity Principle:**
Softmax normalization suppresses affective intensity. HUGO monitors both S(t) (pre-softmax) and A(t) (post-softmax) to preserve the structural magnitude of homeostatic pressure before probabilistic compression.

---

## Ambiente

```
Python 3.13
PyTorch 2.7.1+cu118
GUDHI 3.11
Ripser 0.6.14
Persim 0.3.8
```

Instalar dependências:
```bash
pip install -r requirements.txt
```

---

## Como Citar

```bibtex
@misc{ohio2026hugo,
  author    = {Ohio, David},
  title     = {{HUGO}: Homological Unified Gradient Ontology ---
               Homeostatic Temperature and Primary Emotions as
               Topological Signatures in Neural Information Flow},
  year      = {2026},
  publisher = {Zenodo},
  url       = {https://github.com/aprimora-ai/hugo-framework}
}
```

---

## Série HUGO

| Paper | Título | Status |
|-------|--------|--------|
| **Paper I** | Internal Affective Geometry (este repositório) | ✅ Publicado |
| Paper II | Language-Mediated Empathic Coupling (ECHO) | Em desenvolvimento |
| Paper III | Persistent Topological Memory | Em desenvolvimento |

---

## Trabalhos Relacionados

- **Kappa Method:** https://doi.org/10.5281/zenodo.18883639
- **Kappa-FIN:** https://doi.org/10.5281/zenodo.18883585
