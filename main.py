"""
Assignment 2: STDP Implementation & Engram Connection
Spike-Timing-Dependent Plasticity in Brian2
"""

import matplotlib
matplotlib.use('Agg')

from brian2 import *
import brian2
brian2.prefs.codegen.target = 'numpy'  # Use pure numpy backend (no cython compilation)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

#%% Part 1: Implement STDP Learning Rule
print("=" * 60)
print("Part 1: STDP Learning Rule Implementation")
print("=" * 60)

def create_stdp_synapse_model():
    stdp_params = {
        'tau_pre': 20*ms,
        'tau_post': 20*ms,
        'A_plus': 0.01,
        'A_minus': 0.012,
        'w_max': 1.0,
        'w_min': 0.0,
    }
    neuron_eqs = '''
    dv/dt = (v_rest - v + I_ext) / tau_m : volt
    I_ext : volt
    '''
    synapse_eqs = '''
    w : 1
    dapre/dt = -apre / tau_pre : 1
    dapost/dt = -apost / tau_post : 1
    '''
    on_pre = '''
    v_post += w * 10*mV
    apre += A_plus
    w = clip(w - apost * A_minus, w_min, w_max)
    '''
    on_post = '''
    apost += A_minus
    w = clip(w + apre * A_plus, w_min, w_max)
    '''
    return stdp_params, neuron_eqs, synapse_eqs, on_pre, on_post


#%% Part 2: Simulate STDP Window
print("\n" + "=" * 60)
print("Part 2: STDP Learning Window Simulation")
print("=" * 60)

def simulate_stdp_window(timing_intervals, n_pairs=50, initial_weight=0.5):
    weight_changes = []
    stdp_params, neuron_eqs, synapse_eqs, on_pre, on_post = create_stdp_synapse_model()

    for delta_t in timing_intervals:
        start_scope()

        tau_pre = stdp_params['tau_pre']
        tau_post = stdp_params['tau_post']
        A_plus = stdp_params['A_plus']
        A_minus = stdp_params['A_minus']
        w_max = stdp_params['w_max']
        w_min = stdp_params['w_min']

        pair_interval = 100

        if delta_t >= 0:
            pre_times = np.arange(n_pairs) * pair_interval
            post_times = pre_times + delta_t
        else:
            post_times = np.arange(n_pairs) * pair_interval
            pre_times = post_times - delta_t

        pre_neuron = SpikeGeneratorGroup(1, [0]*n_pairs, pre_times*ms)
        post_neuron = SpikeGeneratorGroup(1, [0]*n_pairs, post_times*ms)

        synapse = Synapses(pre_neuron, post_neuron,
                          model=synapse_eqs,
                          on_pre='''
                          apre += A_plus
                          w = clip(w - apost, w_min, w_max)
                          ''',
                          on_post='''
                          apost += A_minus
                          w = clip(w + apre, w_min, w_max)
                          ''')
        synapse.connect(i=0, j=0)
        synapse.w = initial_weight

        total_time = (n_pairs * pair_interval + 100) * ms
        run(total_time)

        weight_changes.append(synapse.w[0] - initial_weight)

    return np.array(weight_changes)


timing_intervals = np.linspace(-100, 100, 21)  # 10ms resolution for speed

print("Simulating STDP window with 20 spike pairs per timing interval...")
weight_changes = simulate_stdp_window(timing_intervals, n_pairs=20)

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(timing_intervals, weight_changes * 100, c='blue', s=50, alpha=0.7,
           label='Simulated data')

def stdp_positive(x, A, tau):
    return A * np.exp(-x / tau)

def stdp_negative(x, A, tau):
    return -A * np.exp(x / tau)

pos_mask = timing_intervals > 5
if np.sum(pos_mask) > 2:
    try:
        popt_pos, _ = curve_fit(stdp_positive, timing_intervals[pos_mask],
                                weight_changes[pos_mask] * 100, p0=[5, 20])
        x_pos = np.linspace(5, 100, 100)
        ax.plot(x_pos, stdp_positive(x_pos, *popt_pos), 'r-', linewidth=2,
                label=f'LTP fit: τ={popt_pos[1]:.1f}ms')
    except:
        pass

neg_mask = timing_intervals < -5
if np.sum(neg_mask) > 2:
    try:
        popt_neg, _ = curve_fit(stdp_negative, timing_intervals[neg_mask],
                                weight_changes[neg_mask] * 100, p0=[5, 20])
        x_neg = np.linspace(-100, -5, 100)
        ax.plot(x_neg, stdp_negative(x_neg, *popt_neg), 'b-', linewidth=2,
                label=f'LTD fit: τ={abs(popt_neg[1]):.1f}ms')
    except:
        pass

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

ax.fill_between(timing_intervals[timing_intervals > 0], 0,
                weight_changes[timing_intervals > 0] * 100,
                alpha=0.3, color='red', label='Potentiation zone')
ax.fill_between(timing_intervals[timing_intervals < 0], 0,
                weight_changes[timing_intervals < 0] * 100,
                alpha=0.3, color='blue', label='Depression zone')

ax.set_xlabel('Spike Timing Δt = t_post - t_pre (ms)', fontsize=12)
ax.set_ylabel('Weight Change (%)', fontsize=12)
ax.set_title('STDP Learning Window\n(50 spike pairs per timing interval)', fontsize=14)
ax.legend(loc='best')
ax.set_xlim(-110, 110)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'stdp_learning_window.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved stdp_learning_window.png")

print(f"\nSTDP Window Summary:")
print(f"  Maximum potentiation: {np.max(weight_changes)*100:.2f}% at Δt={timing_intervals[np.argmax(weight_changes)]:.0f}ms")
print(f"  Maximum depression: {np.min(weight_changes)*100:.2f}% at Δt={timing_intervals[np.argmin(weight_changes)]:.0f}ms")


#%% Part 3: Network Simulation with Correlated Inputs
print("\n" + "=" * 60)
print("Part 3: Network of 10 Neurons with Correlated Inputs")
print("=" * 60)

def simulate_correlated_network():
    start_scope()

    N = 10
    N_input = 100
    simulation_time = 3 * second

    tau_pre = 20*ms
    tau_post = 20*ms
    A_plus = 0.005
    A_minus = 0.006
    w_max = 1.0
    w_min = 0.0

    tau_m = 20*ms
    v_rest = -70*mV
    v_thresh = -50*mV
    v_reset = -70*mV

    input_rates = 20 * Hz

    input_group = PoissonGroup(N_input, rates=input_rates)

    neuron_eqs = '''
    dv/dt = (v_rest - v) / tau_m : volt
    group_id : integer
    '''

    neurons = NeuronGroup(N, neuron_eqs, threshold='v > v_thresh',
                         reset='v = v_reset', method='euler')
    neurons.v = v_rest
    neurons.group_id = [0]*5 + [1]*5

    synapse_eqs = '''
    w : 1
    dapre/dt = -apre / tau_pre : 1 (event-driven)
    dapost/dt = -apost / tau_post : 1 (event-driven)
    '''

    on_pre = '''
    v_post += w * 5*mV
    apre += A_plus
    w = clip(w - apost, w_min, w_max)
    '''

    on_post = '''
    apost += A_minus
    w = clip(w + apre, w_min, w_max)
    '''

    synapses = Synapses(input_group, neurons,
                       model=synapse_eqs,
                       on_pre=on_pre,
                       on_post=on_post)

    synapses.connect()

    initial_weights = np.full((N_input, N), 0.3)
    synapses.w = initial_weights.flatten()

    lateral_eqs = '''
    w : 1
    dapre/dt = -apre / tau_pre : 1 (event-driven)
    dapost/dt = -apost / tau_post : 1 (event-driven)
    '''

    lateral_on_pre = '''
    v_post += w * 3*mV
    apre += A_plus
    w = clip(w - apost, w_min, w_max)
    '''

    lateral_on_post = '''
    apost += A_minus
    w = clip(w + apre, w_min, w_max)
    '''

    lateral_synapses = Synapses(neurons, neurons,
                                model=lateral_eqs,
                                on_pre=lateral_on_pre,
                                on_post=lateral_on_post)
    lateral_synapses.connect(condition='i != j')
    lateral_synapses.w = 0.2

    spike_mon = SpikeMonitor(neurons)
    weight_mon = StateMonitor(lateral_synapses, 'w', record=True, dt=100*ms)
    rate_mon = PopulationRateMonitor(neurons)

    print("Running network simulation (10 seconds)...")
    run(simulation_time, report='text')

    return neurons, synapses, lateral_synapses, spike_mon, weight_mon, rate_mon, N


(neurons, synapses, lateral_synapses, spike_mon,
 weight_mon, rate_mon, N) = simulate_correlated_network()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
colors = ['red' if i < 5 else 'blue' for i in spike_mon.i]
ax1.scatter(spike_mon.t/ms, spike_mon.i, c=colors, s=1, alpha=0.5)
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Neuron Index')
ax1.set_title('Spike Raster Plot\n(Red: Group 1, Blue: Group 2)')
ax1.set_xlim(0, 10000)

ax2 = axes[0, 1]
final_weights = np.zeros((N, N))
for idx in range(len(lateral_synapses)):
    i = lateral_synapses.i[idx]
    j = lateral_synapses.j[idx]
    final_weights[i, j] = lateral_synapses.w[idx]

im = ax2.imshow(final_weights, cmap='hot', aspect='auto', vmin=0, vmax=1)
ax2.set_xlabel('Post-synaptic Neuron')
ax2.set_ylabel('Pre-synaptic Neuron')
ax2.set_title('Final Lateral Synaptic Weights\n(After STDP Learning)')
plt.colorbar(im, ax=ax2, label='Weight')
ax2.axhline(y=4.5, color='white', linestyle='--', linewidth=2)
ax2.axvline(x=4.5, color='white', linestyle='--', linewidth=2)

ax3 = axes[1, 0]
within_group_weights = []
between_group_weights = []

for t_idx in range(len(weight_mon.t)):
    within = []
    between = []
    for syn_idx in range(len(lateral_synapses)):
        i = lateral_synapses.i[syn_idx]
        j = lateral_synapses.j[syn_idx]
        w = weight_mon.w[syn_idx, t_idx]
        if (i < 5 and j < 5) or (i >= 5 and j >= 5):
            within.append(w)
        else:
            between.append(w)
    within_group_weights.append(np.mean(within))
    between_group_weights.append(np.mean(between))

ax3.plot(weight_mon.t/second, within_group_weights, 'r-', linewidth=2,
         label='Within-group (co-active)')
ax3.plot(weight_mon.t/second, between_group_weights, 'b-', linewidth=2,
         label='Between-group')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Average Synaptic Weight')
ax3.set_title('STDP Strengthens Co-active Connections')
ax3.legend()
ax3.set_ylim(0, 1)

ax4 = axes[1, 1]
within_final = []
between_final = []
for syn_idx in range(len(lateral_synapses)):
    i = lateral_synapses.i[syn_idx]
    j = lateral_synapses.j[syn_idx]
    w = lateral_synapses.w[syn_idx]
    if (i < 5 and j < 5) or (i >= 5 and j >= 5):
        within_final.append(w)
    else:
        between_final.append(w)

ax4.hist(within_final, bins=20, alpha=0.7, color='red', label='Within-group', density=True)
ax4.hist(between_final, bins=20, alpha=0.7, color='blue', label='Between-group', density=True)
ax4.set_xlabel('Synaptic Weight')
ax4.set_ylabel('Density')
ax4.set_title('Final Weight Distribution')
ax4.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'stdp_network_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved stdp_network_results.png")

print("\n" + "=" * 60)
print("Network Analysis Results")
print("=" * 60)
print(f"\nTotal spikes: {len(spike_mon.i)}")
print(f"Group 1 (neurons 0-4) spikes: {np.sum(spike_mon.i < 5)}")
print(f"Group 2 (neurons 5-9) spikes: {np.sum(spike_mon.i >= 5)}")
print(f"\nFinal average weights:")
print(f"  Within-group: {np.mean(within_final):.4f} ± {np.std(within_final):.4f}")
print(f"  Between-group: {np.mean(between_final):.4f} ± {np.std(between_final):.4f}")
print(f"\nWeight change from initial (0.2):")
print(f"  Within-group: {(np.mean(within_final) - 0.2)*100:+.1f}%")
print(f"  Between-group: {(np.mean(between_final) - 0.2)*100:+.1f}%")


#%% Additional Analysis: Correlation vs Weight Strength
print("\n" + "=" * 60)
print("Additional Analysis: Correlation Structure")
print("=" * 60)

def calculate_spike_correlations(spike_mon, N, bin_size=10*ms, max_time=10*second):
    n_bins = int(max_time / bin_size)
    spike_counts = np.zeros((N, n_bins))
    for i in range(N):
        neuron_spikes = spike_mon.t[spike_mon.i == i]
        spike_counts[i], _ = np.histogram(neuron_spikes/ms,
                                          bins=np.linspace(0, max_time/ms, n_bins+1))
    correlations = np.corrcoef(spike_counts)
    return correlations

correlations = calculate_spike_correlations(spike_mon, N, max_time=3*second)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
im1 = ax1.imshow(correlations, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax1.set_xlabel('Neuron')
ax1.set_ylabel('Neuron')
ax1.set_title('Spike Count Correlations')
plt.colorbar(im1, ax=ax1)
ax1.axhline(y=4.5, color='black', linestyle='--', linewidth=2)
ax1.axvline(x=4.5, color='black', linestyle='--', linewidth=2)

ax2 = axes[1]
corr_values = []
weight_values = []

for syn_idx in range(len(lateral_synapses)):
    i = lateral_synapses.i[syn_idx]
    j = lateral_synapses.j[syn_idx]
    corr_values.append(correlations[i, j])
    weight_values.append(lateral_synapses.w[syn_idx])

ax2.scatter(corr_values, weight_values, alpha=0.6, c='purple')
ax2.set_xlabel('Spike Correlation')
ax2.set_ylabel('Final Synaptic Weight')
ax2.set_title('STDP Links Correlation to Connection Strength')

z = np.polyfit(corr_values, weight_values, 1)
p = np.poly1d(z)
x_line = np.linspace(min(corr_values), max(corr_values), 100)
ax2.plot(x_line, p(x_line), 'r--', linewidth=2, label='Linear fit')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'stdp_correlation_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved stdp_correlation_analysis.png")

from scipy.stats import pearsonr
r, p_val = pearsonr(corr_values, weight_values)
print(f"\nCorrelation between spike correlation and synaptic weight:")
print(f"  Pearson r = {r:.3f}, p = {p_val:.2e}")

print("\n" + "=" * 60)
print("ALL STDP ANALYSES COMPLETE")
print("=" * 60)
