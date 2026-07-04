# STDP & Engram Formation

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![Brian2](https://img.shields.io/badge/Brian2-2.4%2B-orange.svg)](https://brian2.readthedocs.io/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19%2B-013243.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Brian2 simulation of **Spike-Timing-Dependent Plasticity (STDP)** — the "neurons that fire together, wire together" learning rule — showing how it sculpts synaptic weights from single spike pairs up to a network that wires itself into memory-like assemblies (engrams).

---

## What it does

- Implements the core STDP synapse model (potentiation/depression traces `apre`, `apost`)
- Maps out the full **STDP learning window**: how much a synapse strengthens or weakens as a function of spike timing (Δt = t_post − t_pre)
- Fits exponential curves to the potentiation (LTP) and depression (LTD) sides of the window
- Simulates a 10-neuron network with correlated Poisson input, split into two groups, and watches STDP reorganize the connections
- Shows that spike-timing correlation directly predicts final synaptic strength — the mechanistic seed of an **engram** (a group of neurons wired together by shared activity)

---

## Background

**STDP** is a biological learning rule: if a presynaptic neuron fires just *before* a postsynaptic one, the synapse strengthens (long-term potentiation, LTP); if it fires just *after*, the synapse weakens (long-term depression, LTD). This asymmetric, millisecond-precise timing dependence is one of the leading candidate mechanisms for how the brain encodes causal relationships and forms memories.

| Phenomenon | What it shows |
|---|---|
| **STDP window** | Near-simultaneous spikes with pre-before-post → strengthening; post-before-pre → weakening |
| **Engram formation** | Neurons that are co-active more often end up more strongly connected to each other than to unrelated neurons |
| **Correlation → weight link** | The more correlated two neurons' activity, the stronger STDP makes their connection |

This is a minimal, mechanistic demonstration of how a purely local, spike-timing-based rule can produce network-level structure resembling a memory trace.

---

## Installation

```bash
pip install brian2 numpy matplotlib scipy
```

## Usage

```bash
python stdp_engram.py
```

Runs all three parts in sequence and saves 3 figures to the script's directory (~10–20 seconds total, depending on machine).

### Core STDP synapse model

```python
on_pre = '''
v_post += w * 10*mV
apre += A_plus
w = clip(w - apost * A_minus, w_min, w_max)
'''
on_post = '''
apost += A_minus
w = clip(w + apre * A_plus, w_min, w_max)
'''
```

---

## Pipeline

1. **Define** the STDP synapse model (traces, weight update rules, bounds)
2. **Sweep spike timing** (Δt from −100 to +100 ms) and measure resulting weight change per interval
3. **Fit exponential decay curves** to the LTP and LTD halves of the window
4. **Build a 10-neuron network** (2 groups of 5) driven by correlated Poisson input, with all-to-all lateral STDP synapses
5. **Track weight evolution** over a 3-second simulation, separating within-group vs. between-group connections
6. **Correlate** spike-count correlation between neuron pairs against their final synaptic weight

---

## Outputs

| File | Shows |
|---|---|
| `stdp_learning_window.png` | Weight change vs. spike timing, with fitted LTP/LTD exponential curves and potentiation/depression zones shaded |
| `stdp_network_results.png` | Spike raster, final weight matrix, weight evolution over time (within- vs. between-group), and final weight distributions |
| `stdp_correlation_analysis.png` | Neuron-pair spike correlation matrix and its relationship to final synaptic weight |

---

## Example results

| Finding | Result |
|---|---|
| STDP window shape | Clear asymmetric potentiation (Δt>0) / depression (Δt<0) exponential decay |
| Within-group vs. between-group final weight | Within-group synapses end up substantially stronger |
| Correlation ↔ weight relationship | Strong positive Pearson correlation (p ≪ 0.05) |

**Takeaway:** a purely local, millisecond-timescale plasticity rule is enough to make co-active neurons wire together more strongly than uncorrelated ones — a minimal computational model of engram (memory trace) formation.

---

## Math, briefly

**STDP traces:** `dapre/dt = −apre/τ_pre`, `dapost/dt = −apost/τ_post`

**On presynaptic spike:** `apre += A₊`, `w ← clip(w − apost·A₋, w_min, w_max)`

**On postsynaptic spike:** `apost += A₋`, `w ← clip(w + apre·A₊, w_min, w_max)`

**Classic STDP window (fit form):**
```
Δw(Δt) =  A·exp(−Δt/τ)   for Δt > 0   (potentiation)
Δw(Δt) = −A·exp(Δt/τ)    for Δt < 0   (depression)
```

---

## Roadmap

- Triplet STDP rules (beyond pairwise spike timing)
- Homeostatic plasticity / synaptic scaling to stabilize learning
- Larger networks with structured (not just random) connectivity
- Explicit engram "reactivation" test — probe whether the trained assembly reactivates together on partial cues

---

## License

MIT — see [LICENSE](LICENSE).

## References

- Bi, G. Q., & Poo, M. M. (1998) — *Synaptic modifications in cultured hippocampal neurons*, Journal of Neuroscience
- Song, S., Miller, K. D., & Abbott, L. F. (2000) — *Competitive Hebbian learning through spike-timing-dependent synaptic plasticity*, Nature Neuroscience
- Caporale, N., & Dan, Y. (2008) — *Spike timing-dependent plasticity: a Hebbian learning rule*, Annual Review of Neuroscience
- Stimberg, M., Brette, R., & Goodman, D. F. (2019) — *Brian 2, an intuitive and efficient neural simulator*, eLife
- 
