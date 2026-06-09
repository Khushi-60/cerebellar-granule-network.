# cerebellar-granule-network.
A population-level spiking neural network simulation of a simplified cerebellar granule cell layer, built with Brian2 — the standard Python simulator for spiking neural networks in computational neuroscience.
# Sparse Spiking Network — Cerebellar Granule Cell Layer (Brian2)

A population-level spiking neural network simulation of a simplified cerebellar granule cell layer, built with [Brian2](https://brian2.readthedocs.io/) — the standard Python simulator for spiking neural networks in computational neuroscience.

Written as part of background preparation for PhD research in NeuroAI and Computational Neuroscience, exploring whether biological sparsity principles can be recovered in artificial networks performing real-time physical prediction.

---

## Biological Motivation

The cerebellar granule cell layer is the most densely packed neural structure in the brain — roughly 50 billion cells in humans. Yet despite their number, only **1–5% of granule cells are active at any moment** (Chadderton et al., 2004). This sparse population code underpins the cerebellum's ability to produce fast, accurate motor predictions while operating on roughly 20 W.

Two structural features produce this sparsity:

1. **Sparse connectivity** — each granule cell receives input from only 4–5 mossy fibres (Eccles et al., 1967), despite the mossy fibre population being much larger
2. **Feedforward architecture** — the cerebellum produces single-pass predictions rather than iterative refinement

This simulation reproduces both features and confirms that the resulting population activity lands in the biological range.

---

## Adaptation from the Brian2 Tutorial

The standard Brian2 introductory tutorial uses a homogeneous population with all-to-all connectivity. The key modification here is **biologically motivated sparse connectivity**: each granule cell independently selects exactly `n_inputs = 5` mossy fibres at random, rather than connecting to all of them.

This single change is sufficient to reproduce the 1–5% population sparsity observed in vivo — which is the central empirical observation motivating the broader PhD research question.

---

## Simulation Results

```
Active granule cells : ~10 / 400  (2.5%)
Biological target    : 1–5% active  (Chadderton et al., 2004)
Mean firing rate     : ~3.3 Hz  (active cells only)
```

### Output figure

Running the script produces a three-panel figure:

- **Raster plot** — each dot is one spike from one granule cell; sparsity is immediately visible
- **Population firing rate** — smoothed estimate of network activity over time
- **Sample voltage traces** — membrane potential of 5 individual granule cells, showing the integrate-and-fire dynamics

---

## Network Parameters

| Parameter | Value | Meaning |
|---|---|---|
| `N_mossy` | 100 | Mossy fibre inputs (heterogeneous rates: 5–60 Hz) |
| `N_granule` | 400 | Granule cell population |
| `n_inputs` | 5 | Mossy fibres per granule cell |
| `tau` | 20 ms | Membrane time constant |
| `V_thresh` | −50 mV | Spike threshold |
| `w` | 11 mV | Synaptic weight per spike |
| `tau_syn` | 5 ms | Synaptic decay time constant |

---

## Usage

```bash
# Install dependencies
pip install brian2 matplotlib numpy

# Run
python brian2_granule_network.py
```

Requires Python 3.8+. Brian2 also requires a C compiler for its code generation backend; on most systems this is already present. If you see a compiler warning, Brian2 will fall back to a pure Python backend automatically.

---

## References

- Chadderton, P., Margrie, T.W., & Häusser, M. (2004). Integration of quanta in cerebellar granule cells during sensory processing. *Nature*, 428, 856–860.
- Eccles, J.C., Ito, M., & Szentágothai, J. (1967). *The Cerebellum as a Neuronal Machine*. Springer.
- Marr, D. (1969). A theory of cerebellar cortex. *Journal of Physiology*, 202(2), 437–470.
- Zador, A.M. (2019). A critique of pure learning and what artificial neural networks can learn from animal brains. *Nature Communications*, 10, 3770. [doi.org/10.1038/s41467-019-11786-6](https://doi.org/10.1038/s41467-019-11786-6)<img width="1634" height="1181" alt="brian2_granule_network" src="https://github.com/user-attachments/assets/ada2c983-05e7-41dc-8d08-e2a52b86dcb9" />
