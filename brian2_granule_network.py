"""
Sparse Spiking Network: A Simplified Cerebellar Granule Cell Layer
==================================================================
Built with Brian2 — the standard simulator for spiking neural networks
in computational neuroscience.

This simulation models a key feature of the cerebellar cortex:
a large population of granule cells receiving sparse, random input
from mossy fibres, producing sparse population-level activity.

Biological grounding:
    - The cerebellar granule cell layer is the most densely packed
      neural structure in the brain (~50 billion cells in humans).
    - Despite their number, only 1-5% fire at any given moment
      (Chadderton et al., 2004) — the sparse code that motivates
      the doctoral research proposal.
    - Each granule cell receives input from only 4-5 mossy fibres
      (extremely sparse connectivity — Eccles et al., 1967).

Adaptation from the Brian2 tutorial:
    The standard introductory Brian2 tutorial uses homogeneous,
    all-to-all connectivity. Here the connectivity is explicitly
    sparse and random: each granule cell connects to exactly
    n_inputs mossy fibres chosen at random. This is the key
    biologically motivated modification.
"""

import numpy as np
import matplotlib.pyplot as plt
from brian2 import *

# ── Reproducibility ───────────────────────────────────────────────────────
np.random.seed(42)
start_scope()

# ── Network parameters ────────────────────────────────────────────────────
N_mossy   = 100   # mossy fibre inputs
N_granule = 400   # granule cell population
n_inputs  = 5     # each granule cell connects to exactly 5 mossy fibres

# ── LIF parameters ────────────────────────────────────────────────────────
tau      = 20  * ms
V_rest   = -70 * mV
V_thresh = -50 * mV
V_reset  = -75 * mV
t_ref    =  2  * ms
tau_syn  =  5  * ms
w        = 11  * mV   # synaptic weight per spike

# ── Neuron equations ──────────────────────────────────────────────────────
eqs = '''
    dv/dt     = (-(v - V_rest) + I_syn) / tau  : volt (unless refractory)
    dI_syn/dt = -I_syn / tau_syn               : volt
'''

# ── Mossy fibre input ─────────────────────────────────────────────────────
# Heterogeneous firing rates: some fibres active, most quiet
rates = np.random.uniform(5, 60, N_mossy) * Hz
mossy = PoissonGroup(N_mossy, rates=rates)

# ── Granule cell population ───────────────────────────────────────────────
granule = NeuronGroup(
    N_granule, eqs,
    threshold  = 'v > V_thresh',
    reset      = 'v = V_reset',
    refractory = t_ref,
    method     = 'euler'
)
granule.v     = V_rest
granule.I_syn = 0 * mV

# ── Sparse synaptic connectivity ──────────────────────────────────────────
# Each granule cell independently picks n_inputs mossy fibres
syn = Synapses(mossy, granule, on_pre='I_syn_post += w')
for j in range(N_granule):
    chosen = np.random.choice(N_mossy, n_inputs, replace=False)
    syn.connect(i=list(chosen), j=j)

# ── Monitors ──────────────────────────────────────────────────────────────
spike_mon = SpikeMonitor(granule)
rate_mon  = PopulationRateMonitor(granule)
state_mon = StateMonitor(granule, 'v', record=range(5))

# ── Run ───────────────────────────────────────────────────────────────────
T = 300 * ms
run(T)

# ── Results ───────────────────────────────────────────────────────────────
active_cells = len(set(spike_mon.i))
sparsity_pct = 100 * active_cells / N_granule
total_spikes = len(spike_mon.i)

print(f"\n── Simulation Results ──────────────────────────────────────")
print(f"Active granule cells : {active_cells} / {N_granule} ({sparsity_pct:.1f}%)")
print(f"Total spikes         : {total_spikes}")
if active_cells > 0:
    mean_rate = total_spikes / (active_cells * float(T))
    print(f"Mean rate (active)   : {mean_rate:.1f} Hz")
print(f"Biological target    : 1–5% active  (Chadderton et al., 2004)")
print(f"────────────────────────────────────────────────────────────")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(11, 8))
fig.suptitle(
    "Sparse Spiking Network — Simplified Cerebellar Granule Cell Layer\n"
    f"(n={N_granule} cells, each connected to {n_inputs} of {N_mossy} mossy fibres)",
    fontsize=12, fontweight='bold'
)

# Raster plot
axes[0].scatter(spike_mon.t / ms, spike_mon.i,
                s=1.2, color='steelblue', alpha=0.5)
axes[0].set_ylabel('Granule cell index')
axes[0].set_title(
    f'Spike raster  |  {active_cells}/{N_granule} cells active '
    f'({sparsity_pct:.1f}%)  |  Biological target: 1–5%',
    fontsize=9
)
axes[0].set_xlim(0, float(T/ms))

# Population firing rate
smooth = rate_mon.smooth_rate(window='flat', width=10*ms)
axes[1].plot(rate_mon.t / ms, smooth / Hz,
             color='darkorange', linewidth=1.1)
axes[1].set_ylabel('Population rate (Hz)')
axes[1].set_title('Smoothed population firing rate', fontsize=9)
axes[1].set_xlim(0, float(T/ms))

# Sample voltage traces
colours = plt.cm.viridis(np.linspace(0.2, 0.85, 5))
for idx in range(5):
    axes[2].plot(state_mon.t / ms, state_mon.v[idx] / mV,
                 color=colours[idx], linewidth=0.8, label=f'Cell {idx}')
axes[2].axhline(V_thresh / mV, color='crimson', linestyle='--',
                linewidth=0.7, label=f'Threshold ({int(V_thresh/mV)} mV)')
axes[2].set_ylabel('Membrane potential (mV)')
axes[2].set_xlabel('Time (ms)')
axes[2].set_title('Sample voltage traces (5 granule cells)', fontsize=9)
axes[2].legend(fontsize=7, loc='upper right', ncol=3)
axes[2].set_xlim(0, float(T/ms))

plt.tight_layout()
plt.savefig('brian2_granule_network.png', dpi=150, bbox_inches='tight')
plt.show()
