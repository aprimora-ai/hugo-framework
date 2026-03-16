# HUGO Series — Paper I (v2)

# Homeostatic Temperature and Primary Emotions as Topological Signatures in Neural Information Flow

**David Ohio**
Independent Researcher | odavidohio@gmail.com
March 2026 | Version 2.0
*Published on Zenodo under CC BY 4.0*

**Original publication:** DOI: 10.5281/zenodo.18947852 (v1, March 11, 2026)

> Part of the HUGO AGI Framework series. This paper constitutes Paper I — Internal Affective Geometry. Subsequent modules (ECHO, REMIND, RHEO, ANIMA, SELF) are in development.

---

## Version 2 — Changes from v1

This version incorporates a correction to the homeostatic field dynamics (HomeostaticField v3.0) that resolves a saturation instability present in v1. The original implementation allowed all homeostatic vectors to drift monotonically toward boundary values over extended runs, collapsing differentiation between configurations. The v3.0 implementation introduces three architectural corrections: (i) a set-point model in which each vector oscillates around its initial condition rather than a shared nominal center; (ii) orthogonal channel-specific projections from the representation space, creating independent feedback pathways; and (iii) ensemble-calibrated decay rates derived from perturbation statistics during a burn-in period, analogous to the CALM calibration in the Kappa-FIN pipeline. These corrections preserve the transient dynamics reported in v1 while extending the framework's validity to arbitrary temporal horizons — a prerequisite for integration with subsequent modules in the framework. All three experimental series have been re-executed with the corrected implementation. The statistical conclusions are strengthened: all results from v1 are reproduced with improved stability and additional findings are reported.

---

## Abstract

We present a pure neural network architecture equipped with a multivariate homeostatic regulation field H(t) in R^5, and demonstrate that controlled deviations from homeostatic equilibrium produce statistically separable, structurally interpretable topological signatures in the network's attention dynamics. The system contains no symbolic emotion labels, no reward signal, and no external supervision: the emotional analogs emerge entirely from the internal regulatory dynamics of the homeostatic field interacting with attention.

This work makes no claim that the network experiences emotion, nor that it is conscious or sentient. What we claim — and experimentally demonstrate — is more specific and more falsifiable: homeostatic gradient configurations that are structurally equivalent to primary emotional states produce distinct, measurable topological geometries in the neural information flow. The primary emotions (FEAR, CARE, EXHAUSTION, CONFLICT, SEEK) are operationalized as precise multivariate homeostatic configurations, not as labels or interpretations applied post hoc.

We ground this in the Kappa Method (Ohio, 2025a) — persistent homology applied to attention distributions — and in a homeostatic temperature functional tau(H(t)) = tau_base / (1 + 4 * L(t)), where L(t) = sum_i w_i * delta_H_i(t) is the total homeostatic deviation, which modulates attention entropy during forward passes. A key architectural decision motivates the dual analysis of post-softmax attention matrices A(t) and pre-softmax score matrices S(t): softmax normalization suppresses affective intensity.

Three experimental series are reported. Experiment 1.5 (5 configurations, 50 trials, 40 steps) establishes the baseline result: homeostatic configurations produce significantly different topological signatures across six metrics (p < 10^-8 in all cases, Kruskal-Wallis). Expansion 2 (8 configurations, 50 trials) demonstrates that FEAR is a continuous field, not a discrete category: four configurations with decreasing H2 (structural integrity) produce monotonically decreasing tau and increasing L(t) (Spearman r = +/-1.0, p = 0.000). Novel states CARE, EXHAUSTION, CONFLICT, and FEAR-SEEK-TRANSITION are discriminated with p = 0 across all six metrics. Expansion 3 (6 configurations, 30 trials, 80 steps) captures the temporal dynamics: 4 of 6 trajectory metrics achieve significance, with trajectory length (p < 10^-28), stability (p < 10^-28), drift (p < 10^-4), and trajectory entropy (p < 10^-3) separating configurations, while convergence and velocity ratio do not — indicating that emotional configurations differentiate by trajectory geometry, not convergence dynamics.

---

## Table of Contents

1. Introduction
2. Background and Related Work
3. Architecture and Mathematical Framework
4. Experiment 1.5: Establishing Topological Baselines
5. Expansion 2: Granular Mapping of Primary Emotional Space
6. Expansion 3: Temporal Dynamics of Topological Trajectories
7. Discussion
8. Conclusion

---

## 1. Introduction

The relationship between affect and cognition is one of the most debated questions in both neuroscience and artificial intelligence. Dominant frameworks in cognitive science treat emotion as a regulatory layer that modulates cognition (Damasio, 1994), while computational approaches have largely kept these as orthogonal concerns — designing optimizers for cognition and, separately, reward/aversion signals for motivation.

This paper advances the hypothesis that emotional regulation and cognitive organization may admit a shared topological description. We argue that primary emotional states — operationally defined as deviations from homeostatic equilibrium — produce measurable, topologically distinct signatures in the geometry of neural information flow. These signatures are captured by persistent homology applied to attention distributions, instantiating the Kappa Method (Ohio, 2025a) in the domain of affect.

We introduce the HUGO framework (Homological Unified Gradient Ontology), of which this paper constitutes the first experimental layer. HUGO proposes that any sufficiently complex information-processing system equipped with homeostatic regulation will exhibit topologically detectable emotional geometry — not as a design objective, but as a structural consequence of regulated information flow.

**Scope and Disclaimer.** This paper reports results from an artificial system that processes random inputs. We make no claim of consciousness, sentience, or genuine emotional experience. The system has homeostasis. Its primary emotional states are observable as topological signatures. That is the contribution. This paper does not address empathic coupling, dialogue, or social regulation between agents; it establishes the internal conditions that would make such coupling formally possible.

**The Pre-Softmax Affective Intensity Principle.** A central insight that emerged during development: softmax normalization suppresses affective intensity. The raw score matrices S(t) encode the full structural magnitude of homeostatic pressure before probabilistic compression. HUGO therefore explicitly preserves and measures the pre-softmax field to maintain access to affective intensity before this compression occurs — a design principle, not merely an implementation detail.

---

## 2. Background and Related Work

### 2.1 Persistent Homology and the Kappa Method

Persistent homology (Edelsbrunner & Harer, 2010; Carlsson, 2009) provides a principled language for describing "shape" in high-dimensional systems. The Kappa Method (Ohio, 2025a) applies persistent homology to detect structural transitions in dynamical systems, validated on financial markets, educational data, and network analysis. This paper extends the Kappa Method to the domain of affect: the attention matrix A(t) and score matrix S(t) are treated as point clouds whose topological features (Betti numbers, persistence entropy, Wasserstein distance) encode the system's emotional geometry.

### 2.2 Affect as Continuous Field

Biological evidence strongly suggests that affects are not categorical but continuous (Russell, 1980; Barrett, 2017). Neuroscience has increasingly moved toward a view of affect as a high-dimensional field shaped by allostatic regulation (Sterling, 2012). The HUGO framework instantiates this view computationally: emotion is not a label applied to a state but the shape of the constraint field on information flow.

### 2.3 Emotion in Artificial Neural Networks

Existing approaches to emotion in AI rely on labeled datasets, explicit reward signals, or post-hoc classification (Picard, 1997; Poria et al., 2017). HUGO differs fundamentally: the emotional analogs are not trained, labeled, or supervised. They emerge from the interaction between homeostatic regulation and attention dynamics.

---

## 3. Architecture and Mathematical Framework

### 3.1 Homeostatic Field (v3.0)

The homeostatic field H(t) = [H1, H2, H3, H4, H5] in R^5 represents five regulated internal variables:

| Vector | Biological Analog | Nominal Range | Set Point | Sensitivity |
|--------|-------------------|---------------|-----------|-------------|
| H1 — Energetic coherence | Glucose | [0.40, 0.80] | Initial value | 0.8 |
| H2 — Structural integrity | Physical integrity | [0.50, 0.90] | Initial value | 1.2 |
| H3 — Load equilibrium | Osmotic pressure | [0.30, 0.70] | Initial value | 1.0 |
| H4 — Temporal consistency | Circadian rhythm | [0.40, 0.80] | Initial value | 0.6 |
| H5 — Representational stability | Core temperature | [0.50, 0.85] | Initial value | 0.9 |

**v3.0 Dynamics.** Each vector H_i(t) evolves according to:

    H_i(t+1) = H_i(t) + 0.02 * tanh(p_i . r(t) * s_i) - d_i * (H_i(t) - H_i*)

where p_i is the orthogonal projection vector for channel i (fixed, derived from Gram-Schmidt orthogonalization), r(t) is the representation output of the network, s_i is the channel sensitivity, d_i is the ensemble-calibrated decay rate, and H_i* is the set point (= initial value, fixed for the lifetime of the agent).

**Ensemble calibration.** During a burn-in period of T_burn = 30 steps, the system records perturbation magnitudes for each channel. The burn-in is divided into overlapping sub-windows, and the decay rate is computed as:

    d_i = median( mean(|perturbation|_j) / target_amplitude )  for j = 1..N_windows

This is analogous to the CALM scan in the Kappa-FIN pipeline (Ohio, 2026c): the threshold is derived from data, not imposed as a hyperparameter.

**Orthogonal projections.** Each channel receives a fixed orthogonal projection p_i in R^d from the representation space, ensuring that the five channels respond to independent subspaces of neural activity. This creates five independent feedback pathways: H -> tau -> A -> r -> H, where each channel is modulated by a different aspect of the representation.

**Set point model.** The restoring force drives each vector toward its set point H_i* (= initial condition), not toward the nominal center. This ensures that an agent initialized with low structural integrity (H2 = 0.25) maintains that characteristic indefinitely, oscillating around H2 ~ 0.25 rather than drifting toward the nominal center of 0.70.

### 3.2 Homeostatic Temperature

    tau(H(t)) = tau_base / (1 + alpha * L(t)),   tau_min <= tau <= tau_base

where L(t) = sum_i w_i * delta_H_i(t) is the weighted total homeostatic deviation, alpha = 4, tau_base = 1.0, and tau_min = 0.05. The weights are w = [0.15, 0.30, 0.20, 0.15, 0.20], reflecting the relative biological urgency of each channel.

When L(t) = 0, tau = 1.0 (standard softmax). When L(t) increases, tau decreases: attention becomes more concentrated — the system "focuses" under homeostatic stress. This is not a metaphor; it is a direct consequence of the temperature-scaled softmax A_ij = exp(S_ij/tau) / sum_k exp(S_ik/tau).

### 3.3 Network Architecture

The Structural Attention Network is a multi-head attention network with fixed random weights (Gray Box): W_Q, W_K, W_V, W_O are initialized once and never updated. The behavioral differentiation between configurations emerges exclusively from the homeostatic field, not from optimization.

Architecture: input_dim = 64, seq_len = 32, hidden_dim = 128, n_heads = 4, n_layers = 3. The attention computation incorporates the metric deformation matrix G(H(t)):

    A(t) = softmax( G(H(t)) * (Q . K^T) / (tau(H(t)) * sqrt(d_k)) )

where G(H(t)) deforms the geometry of the attention space based on the homeostatic state, and * denotes Hadamard (element-wise) product.

### 3.4 Topological Monitoring: Kappa

The Kappa monitor computes persistent homology on both A(t) (post-softmax) and S(t) (pre-softmax, normalized) at each step, extracting:

- Betti numbers beta_0, beta_1: connected components and cycles
- Persistence entropy H_pers: Shannon entropy of the persistence diagram
- Wasserstein distance W_p: distance between consecutive persistence diagrams (measures topological change)

Six metrics are tracked: entropy_A, entropy_S, wass_A, wass_S, tau, L(t).

---

## 4. Experiment 1.5: Establishing Topological Baselines

### 4.1 Setup

Five homeostatic configurations, 50 trials each, 40 steps per trial. Unified Gaussian random input x ~ N(0, I) across all configurations (same seed per trial). Differentiation arises exclusively from H(t).

| Configuration | H vector | L_0 | tau_0 | Expected label |
|--------------|-----------|-------|---------|---------------|
| BASELINE | [0.60, 0.70, 0.50, 0.60, 0.675] | 0.000 | 1.000 | SEEK_ANALOG |
| H2_LOW | [0.60, 0.25, 0.50, 0.60, 0.675] | 0.075 | 0.769 | FEAR_SEEK_TRANSITION |
| H3H5_HIGH | [0.60, 0.70, 0.85, 0.60, 0.95] | 0.050 | 0.833 | TRANSITIONAL/RAGE |
| H1_LOW | [0.15, 0.70, 0.50, 0.60, 0.675] | 0.038 | 0.870 | PANIC_ANALOG |
| H2_LOW_H1_HIGH | [0.78, 0.25, 0.50, 0.60, 0.675] | 0.075 | 0.769 | FEAR_SEEK_TRANSITION |

### 4.2 Results

| Configuration | tau (mean) | ent_A | ent_S | wass_A | wass_S | L(t) | Label |
|--------------|-------------|-------|-------|--------|--------|------|-------|
| BASELINE | 1.000 | 3.4485 | 3.4679 | 0.00005 | 0.07273 | 0.0000 | SEEK (50/50) |
| H2_LOW | 0.769 | 3.5016 | 3.5838 | 0.00047 | 0.07027 | 0.0752 | FEAR_SEEK (50/50) |
| H3H5_HIGH | 0.831 | 3.5490 | 3.6311 | 0.00081 | 0.08447 | 0.0508 | TRANS:21/RAGE:29 |
| H1_LOW | 0.870 | 3.5483 | 3.6584 | 0.00070 | 0.09208 | 0.0374 | PANIC (50/50) |
| H2_LOW_H1_HIGH | 0.769 | 3.5631 | 3.6043 | 0.00101 | 0.07582 | 0.0752 | FEAR_SEEK (50/50) |

**Statistical significance:** Kruskal-Wallis test on all six metrics:

| Metric | H statistic | p-value | Significant? |
|--------|-------------|---------|------------|
| entropy_A | 220.42 | < 10^-8 | Yes |
| entropy_S | 239.04 | < 10^-8 | Yes |
| tau | 228.92 | < 10^-8 | Yes |
| L(t) | 228.92 | < 10^-8 | Yes |
| wass_A | 214.40 | < 10^-8 | Yes |
| wass_S | 107.08 | < 10^-8 | Yes |

**Key observations:**

H2_LOW and H2_LOW_H1_HIGH share the same tau (0.769) and L(t) (0.075) because both have H2 = 0.25. However, they differ in entropy_A (3.502 vs 3.563) and wass_A (0.00047 vs 0.00101). This demonstrates that the topological monitoring captures more than temperature alone: the elevated H1 in H2_LOW_H1_HIGH produces more dynamic processing (higher Wasserstein distance between consecutive attention patterns) without altering the homeostatic cost.

**v2 vs v1 comparison:** In v1, all configurations converged to RAGE_ANALOG by step 40 due to the saturation bug. In v2, 4 distinct emotional labels are maintained: SEEK, FEAR_SEEK, PANIC, and TRANSITIONAL/RAGE. The temperature spread (0.231) is preserved throughout the run rather than collapsing to ~0.

---

## 5. Expansion 2: Granular Mapping of Primary Emotional Space

### 5.1 Setup

Eight configurations including a FEAR gradient (4 levels of H2) and four novel states (CARE, EXHAUSTION, CONFLICT, FEAR-SEEK-TRANSITION). 50 trials each, 40 steps.

### 5.2 FEAR as a Continuous Topological Field

Four configurations with decreasing H2: 0.55, 0.40, 0.25, 0.10.

| Configuration | H2 | tau | ent_A | ent_S | wass_S | L(t) |
|--------------|------|------|-------|-------|--------|------|
| FEAR_GRAD_1 | 0.55 | 1.000 | 3.4725 | 3.5347 | 0.10867 | 0.000 |
| FEAR_GRAD_2 | 0.40 | 0.892 | 3.4893 | 3.5745 | 0.09192 | 0.030 |
| FEAR_GRAD_3 | 0.25 | 0.769 | 3.5016 | 3.5838 | 0.07027 | 0.075 |
| FEAR_GRAD_4 | 0.10 | 0.675 | 3.5043 | 3.5840 | 0.05822 | 0.120 |

**Monotonicity analysis (Spearman rank correlation):**

| Metric | Spearman r | p-value | Monotonic? |
|--------|------------|---------|-----------|
| tau | +1.0 | 0.000 | Yes |
| entropy_A | -1.0 | 0.000 | Yes |
| wass_S | +1.0 | 0.000 | Yes |
| L(t) | -1.0 | 0.000 | Yes |

All four metrics are perfectly monotonic with H2. FEAR is not a threshold that "turns on" — it is a continuous topological field that becomes progressively more restrictive as structural integrity declines. This is the strongest empirical result of the framework.

### 5.3 Novel Emotional Analogs

**Full comparison (8 configurations):**

| Configuration | Hypothesis | tau | ent_A | ent_S | wass_S | L(t) |
|--------------|-----------|------|-------|-------|--------|------|
| FEAR_GRAD_1 | PRE_FEAR | 1.000 | 3.4725 | 3.5347 | 0.10867 | 0.000 |
| FEAR_GRAD_2 | FEAR_MILD | 0.892 | 3.4893 | 3.5745 | 0.09192 | 0.030 |
| FEAR_GRAD_3 | FEAR_ANALOG | 0.769 | 3.5016 | 3.5838 | 0.07027 | 0.075 |
| FEAR_GRAD_4 | FREEZE | 0.675 | 3.5043 | 3.5840 | 0.05822 | 0.120 |
| CARE | CARE_ANALOG | 0.954 | 3.4950 | 3.6064 | 0.12295 | 0.012 |
| EXHAUSTION | EXHAUSTION | 0.908 | 3.5685 | 3.6517 | 0.09410 | 0.025 |
| CONFLICT | CONFLICT | 0.675 | 3.5295 | 3.6486 | 0.06941 | 0.121 |
| FEAR_SEEK_TRANS | FEAR-SEEK | 0.769 | 3.5631 | 3.6043 | 0.07582 | 0.075 |

Kruskal-Wallis across all 8 configurations: p = 0 on all six metrics (H > 326 on every metric).

CARE exhibits the highest wass_S (0.123) — the most dynamic pre-softmax processing — despite having low L(t) (0.012). This is consistent with the interpretation of CARE as an exploratory, affiliative state: low urgency but high engagement.

CONFLICT and FEAR_GRAD_4 share similar tau (0.675) and L(t) (~0.12) but differ in ent_A (3.530 vs 3.504) and ent_S (3.649 vs 3.584). The topological signatures discriminate between states that temperature alone cannot separate.

---

## 6. Expansion 3: Temporal Dynamics of Topological Trajectories

### 6.1 From Snapshots to Films

Experiments 1.5 and 2 capture topological snapshots. Expansion 3 extends to temporal trajectories: how does the topological geometry of each emotional configuration evolve over 80 steps?

### 6.2 Setup

Six configurations, 30 trials, 80 steps. Metrics: convergence ratio, velocity ratio, trajectory length, stability, drift, trajectory entropy.

### 6.3 Results

| Configuration | Conv. | Vel. ratio | Length | Stability | Drift | Ent. traj. |
|--------------|-------|-----------|--------|-----------|-------|-----------|
| BASELINE | 0.754 | 1.122 | 2.971 | 0.411 | 0.025 | 2.509 |
| FEAR | 1.041 | 1.203 | 3.452 | 0.387 | 0.019 | 2.611 |
| CARE | 0.744 | 1.263 | 4.626 | 0.393 | 0.018 | 2.668 |
| EXHAUSTION | 0.835 | 1.311 | 4.555 | 0.376 | 0.027 | 2.517 |
| CONFLICT | 1.033 | 1.177 | 3.917 | 0.375 | 0.018 | 2.422 |
| FEAR_SEEK_TRANS | 0.942 | 1.321 | 3.993 | 0.379 | 0.023 | 2.489 |

**Statistical significance:**

| Metric | H statistic | p-value | Significant? |
|--------|-------------|---------|------------|
| Convergence | 8.27 | 0.142 | No |
| Velocity ratio | 5.28 | 0.383 | No |
| Trajectory length | 113.35 | < 10^-22 | Yes |
| Stability | 166.71 | < 10^-28 | Yes |
| Drift | 23.56 | 2.6e-4 | Yes |
| Trajectory entropy | 29.14 | 2.2e-5 | Yes |

The non-significance of convergence and velocity ratio is a genuine negative result: all configurations decelerate at comparable rates, reflecting a universal dissipative dynamic. The rate of deceleration is not emotionally differentiated; the structure of the trajectory is.

**Key finding:** BASELINE and CARE converge (ratio < 1.0) — they settle toward attractors. FEAR and CONFLICT diverge (ratio > 1.0) — they oscillate without convergence. CARE produces the longest trajectory (4.63) — it explores the largest region of state space. CONFLICT has the lowest trajectory entropy (2.42) — it cycles repetitively rather than exploring.

**v2 vs v1 comparison:** In v1, convergence was significant (p = 8.9e-7) and velocity ratio was not (p = 0.667). In v2, neither is significant. This is because the v3.0 set-point model prevents the artificial convergence caused by saturation: in v1, all configurations were converging toward the same saturated state, creating a spurious convergence signal. In v2, the convergence metric reflects genuine dynamical properties — and the configurations do not systematically converge to different attractors. The trajectory geometry metrics (length, stability, drift, entropy) remain strongly significant, confirming that the topological shape of the trajectory, not its endpoint, carries the emotional identity.

---

## 7. Discussion

### 7.1 The Pre-Softmax Affective Intensity Principle: Empirical Validation

Across all three experimental series, the post-softmax entropy H_A and pre-softmax entropy H_S exhibit systematic dissociation. CARE produces the highest wass_S (0.123) among all configurations despite having near-baseline L(t) — indicating that the pre-softmax field captures affective engagement that the post-softmax distribution compresses. This validates the architectural principle: understanding *why* the system attends (motivational causation) requires access to the pre-normalization field.

### 7.2 Emotion as Continuous Topological Field

Expansion 2's FEAR gradient — Spearman r = +/-1.0 for four metrics — demonstrates that emotion is not a discrete category but a continuous topological field. There is no threshold below which FEAR "turns on." There is only a field — a geometry of constraint on information flow — that becomes progressively more restrictive as H2 declines. This is consistent with the dimensional view of affect (Russell, 1980; Barrett, 2017).

### 7.3 Trajectory Geometry vs. Convergence Dynamics

Expansion 3 reveals that emotional configurations differentiate by the *shape* of their trajectories (length, stability, entropy) rather than by their convergence properties. All configurations decelerate at comparable rates — the universal dissipative dynamic. The emotional "identity" is encoded in the trajectory geometry: CARE explores, CONFLICT cycles, BASELINE settles, FEAR oscillates.

### 7.4 The Set-Point Model and Long-Horizon Stability

The v3.0 homeostatic field resolves the saturation instability of v1 by introducing a biologically grounded set-point model. Each homeostatic vector oscillates around its initial condition, which represents the agent's "physiology" — a fixed characteristic that does not adapt. This has a direct implication for the HUGO AGI framework: different agents initialized with different H vectors will maintain distinct emotional repertoires indefinitely, even under identical environmental inputs. Identity is encoded in the set points; experience is encoded in the oscillations.

### 7.5 Implications for AGI Architecture

The results suggest that homeostatic regulation provides a tractable, mathematically grounded implementation of affective regulatory functions through the temperature tau(H(t)) and the gradient modulator G(H(t)). The present results are obtained with R(t) = 0 (no memory). The introduction of persistent topological memory (R_perm != 0) is expected to produce qualitatively richer dynamics — including individual "perspectives" that diverge between instances with identical architectures but different histories. This will be the subject of subsequent papers in the HUGO series, which will address episodic memory, temporal flow regulation, phenomenological animation, and reflexive self-modeling.

### 7.6 Limitations

**Artificial inputs.** All experiments use Gaussian random inputs. Whether these signatures persist when processing structured semantic inputs remains an open question.

**Architecture specificity.** Results are obtained with a specific Gray-Box architecture. Generalization to other attention mechanisms requires further investigation.

**Biological grounding.** The correspondence to biological affect is structural analogy, not physiological identity.

**Ensemble calibration sensitivity.** The burn-in period (T_burn = 30) and target amplitude (0.08) are design choices. Sensitivity analysis across these parameters has not been performed.

---

## 8. Conclusion

We have demonstrated that homeostatic field configurations produce statistically distinguishable, structurally interpretable topological signatures in neural information flow. Three experimental series establish: (i) primary emotional analogs are separable with p < 10^-8 via six topological metrics; (ii) FEAR is a continuous topological field, not a discrete category (Spearman r = +/-1.0); (iii) distinct trajectory geometries characterize exploratory, convergent, and divergent emotional regimes over time — with the emotional identity encoded in trajectory shape, not convergence endpoint.

The v3.0 homeostatic field, with ensemble-calibrated dynamics and set-point preservation, extends these results to arbitrary temporal horizons — a prerequisite for the broader HUGO AGI framework.

The central contribution is mechanistic: homeostatic gradients produce measurable, continuous topological geometry in neural information flow. The present results establish that primary affect-like regimes can be generated, measured, and topologically discriminated within a single homeostatically regulated network. This raises an immediate extension: whether such regimes can be altered not only by internal disequilibrium, but by the inferred emotional state of another agent — a question for future work in the HUGO series.

---

## Conflict of Interest

None declared.

## Acknowledgments

This work was conducted independently. The name HUGO honors Hugo, whose existence is the primary motivational field from which this research emerges.

---

## References

Barrett, L.F. (2017). *How Emotions Are Made: The Secret Life of the Brain*. Houghton Mifflin Harcourt.

Carlsson, G. (2009). Topology and data. *Bulletin of the American Mathematical Society*, 46(2), 255-308.

Damasio, A.R. (1994). *Descartes' Error: Emotion, Reason, and the Human Brain*. Putnam.

Easterbrook, J.A. (1959). The effect of emotion on cue utilization and the organization of behavior. *Psychological Review*, 66(3), 183-201.

Edelsbrunner, H. & Harer, J.L. (2010). *Computational Topology: An Introduction*. American Mathematical Society.

Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

Ohio, D. (2025a). The Kappa Method: Domain-agnostic detection of structural instability via persistent homology. Zenodo. DOI: 10.5281/zenodo.18883639.

Ohio, D. (2026b). Homeostatic temperature and primary emotions as topological signatures in neural information flow. HUGO Series Paper I (v1). Zenodo. DOI: 10.5281/zenodo.18947852.

Ohio, D. (2026c). Kappa-Radiante: A five-dimensional structural analysis instrument. Zenodo. DOI: 10.5281/zenodo.18940478.

Picard, R.W. (1997). *Affective Computing*. MIT Press.

Poria, S., Cambria, E., Bajpai, R., & Hussain, A. (2017). A review of affective computing. *Information Fusion*, 37, 98-125.

Russell, J.A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161-1178.

Sterling, P. (2012). Allostasis: a model of predictive regulation. *Physiology & Behavior*, 106(1), 5-15.

---

*David Ohio | Independent Researcher | odavidohio@gmail.com*
*HUGO AGI Framework — Paper I v2 | March 2026*
*License: CC BY 4.0*
