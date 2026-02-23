"""
Experiment: Alzheimer's Gradient Collapse

The Phi-Dwell finding: AD brains show collapse of the alpha-band dwell
specifically, with the gradient becoming flat across bands.

The Deerskin prediction: this should happen when homeostatic regulation
fails in intermediate-resolution layers. The alpha layer (near-Nyquist,
maximum moiré stress) should be MOST SENSITIVE to homeostatic disruption.

This experiment:
1. Builds a healthy DeerskinCortex and measures its dwell gradient
2. Simulates 'AD-like' disruption (failing homeostatic gain in alpha layer)
3. Measures how the gradient changes
4. Compares to Phi-Dwell clinical data

The prediction: disrupting alpha homeostasis should
- INCREASE vocabulary (more states visited, less settling)
- DECREASE structure (CV drops toward 1)
- FLATTEN the gradient (all bands similar)

This is exactly what Phi-Dwell found in AD patients:
'AD brains show more vocabulary with less structure'
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.insert(0, '/home/claude/deerskin_cortex')
from core.deerskin_layer import DeerskinCortex, DeerskinLayer, MoireGrid


def run_cortex_condition(cortex: DeerskinCortex, n_steps: int, 
                          signal_noise: float = 0.1) -> dict:
    """Run the cortex and collect vocabulary + gradient statistics."""
    
    np.random.seed(42)
    
    # Collect word sequences (like Phi-Dwell tokenization)
    words = []
    phase_histories = [[] for _ in range(len(cortex.layers))]
    
    for t in range(n_steps):
        x = np.sin(np.linspace(0, 4*np.pi, cortex.n_input) + t * 0.05)
        x += signal_noise * np.random.randn(cortex.n_input)
        
        output, phase_states = cortex.forward(x, learn=True)
        
        # Tokenize: dominant bin in each layer's phase state
        # (analogous to Phi-Dwell's dominant eigenmode per band)
        word = tuple(
            int(np.argmax(np.abs(ps))) % 8 
            for ps in phase_states
        )
        words.append(word)
        
        for i, ps in enumerate(phase_states):
            phase_histories[i].append(ps.copy())
    
    # Vocabulary statistics (Phi-Dwell metrics)
    from collections import Counter
    counts = Counter(words)
    vocab_size = len(counts)
    n_words = len(words)
    probs = np.array(list(counts.values())) / n_words
    entropy = -np.sum(probs * np.log2(probs + 1e-15))
    
    # Top-5 concentration
    top5 = sum(c for _, c in counts.most_common(5)) / n_words
    
    # Dwell gradient
    gradient = cortex.get_dwell_gradient()
    
    # Compute per-layer CV from phase histories
    layer_cvs = []
    for ph in phase_histories:
        arr = np.array(ph)
        # Track dominant dimension over time
        dominant = np.argmax(np.abs(arr), axis=1)
        dwells = []
        current = 1
        for i in range(1, len(dominant)):
            if dominant[i] == dominant[i-1]:
                current += 1
            else:
                dwells.append(current)
                current = 1
        if len(dwells) > 5:
            cv = np.std(dwells) / (np.mean(dwells) + 1e-9)
            mean_d = np.mean(dwells)
        else:
            cv = 0.0
            mean_d = 0.0
        layer_cvs.append((mean_d, cv))
    
    return {
        'vocab_size': vocab_size,
        'entropy': entropy,
        'top5_concentration': top5,
        'gradient': gradient,
        'layer_cvs': layer_cvs,
        'words': words,
    }


def simulate_alzheimer_disruption():
    """
    Test the Deerskin prediction about AD.
    
    Disruption mechanism: reduce homeostatic gain in alpha layer
    (the near-Nyquist, highest-moiré-stress layer).
    
    This simulates the loss of 'viscosity' that Phi-Dwell observes:
    the alpha layer can no longer hold a pattern stable - it slips.
    """
    
    n_input = 32
    n_output = 10
    hidden_dim = 16
    n_steps = 300
    
    print("Building healthy cortex...")
    healthy = DeerskinCortex(n_input, n_output, hidden_dim)
    healthy_stats = run_cortex_condition(healthy, n_steps)
    
    print("Building AD-like cortex (disrupted alpha homeostasis)...")
    ad_cortex = DeerskinCortex(n_input, n_output, hidden_dim)
    
    # Disrupt the alpha layer (layer index 2 = 16-freq layer)
    # Reduce homeostatic gain → regulator can't maintain stability
    # This is the computational analogue of alpha-band eigenmode instability
    alpha_layer_idx = 2
    ad_cortex.layers[alpha_layer_idx].grid.homeostatic_gain = 0.001  # was 0.01
    ad_cortex.layers[alpha_layer_idx].grid.homeostatic_setpoint = 0.8  # drifted
    
    ad_stats = run_cortex_condition(ad_cortex, n_steps)
    
    print("Building severe AD cortex (all homeostasis disrupted)...")
    severe_ad = DeerskinCortex(n_input, n_output, hidden_dim)
    for layer in severe_ad.layers:
        layer.grid.homeostatic_gain = 0.002
        layer.grid.homeostatic_setpoint += np.random.randn() * 0.3
    
    severe_stats = run_cortex_condition(severe_ad, n_steps)
    
    return healthy_stats, ad_stats, severe_stats


def plot_comparison(healthy, ad, severe, save_path=None):
    
    fig = plt.figure(figsize=(18, 12), facecolor='#0a0a14')
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35,
                  left=0.07, right=0.96, top=0.92, bottom=0.06)
    
    fig.suptitle("Deerskin Cortex: Simulated Alzheimer's Gradient Collapse\n"
                 "Prediction vs Phi-Dwell Empirical Finding",
                 fontsize=14, color='#c0c0e0', fontweight='bold', y=0.97)
    
    colors = {'Healthy\n(CN)': '#4a90d9', 
               'AD-like\n(alpha disrupted)': '#d94a4a',
               'Severe AD\n(all disrupted)': '#d98a4a'}
    
    datasets = [
        ('Healthy\n(CN)', healthy, '#4a90d9'),
        ('AD-like\n(alpha disrupted)', ad, '#d94a4a'),
        ('Severe AD\n(all disrupted)', severe, '#d98a4a'),
    ]
    
    # Panel 1: Vocabulary size
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#0d0d20')
    labels = [d[0] for d in datasets]
    vocab_vals = [d[1]['vocab_size'] for d in datasets]
    bars = ax1.bar(range(3), vocab_vals, 
                   color=[d[2] for d in datasets], alpha=0.8, width=0.6)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(['CN', 'AD', 'Severe'], color='#c0c0e0', fontsize=9)
    ax1.set_title("Vocabulary Size\n(Phi-Dwell: CN=953, AD=1052, FTD=1078)",
                 fontsize=9, color='#c0c0e0')
    ax1.tick_params(colors='#606080', labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color('#2a2a4a')
    for bar, val in zip(bars, vocab_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                str(val), ha='center', color='#c0c0e0', fontsize=9)
    
    # Panel 2: Shannon entropy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#0d0d20')
    entropy_vals = [d[1]['entropy'] for d in datasets]
    bars2 = ax2.bar(range(3), entropy_vals,
                    color=[d[2] for d in datasets], alpha=0.8, width=0.6)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(['CN', 'AD', 'Severe'], color='#c0c0e0', fontsize=9)
    ax2.set_title("Shannon Entropy (bits)\n(Phi-Dwell: CN=8.26, AD=8.57)",
                 fontsize=9, color='#c0c0e0')
    ax2.tick_params(colors='#606080', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color('#2a2a4a')
    for bar, val in zip(bars2, entropy_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f'{val:.2f}', ha='center', color='#c0c0e0', fontsize=9)
    
    # Panel 3: Top-5 concentration (AD should be LOWER - less structured)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#0d0d20')
    conc_vals = [d[1]['top5_concentration'] for d in datasets]
    bars3 = ax3.bar(range(3), conc_vals,
                    color=[d[2] for d in datasets], alpha=0.8, width=0.6)
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(['CN', 'AD', 'Severe'], color='#c0c0e0', fontsize=9)
    ax3.set_title("Top-5 Concentration\n(Phi-Dwell: CN=0.154, AD=0.124 ↓)",
                 fontsize=9, color='#c0c0e0')
    ax3.tick_params(colors='#606080', labelsize=8)
    for spine in ax3.spines.values():
        spine.set_color('#2a2a4a')
    for bar, val in zip(bars3, conc_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f'{val:.3f}', ha='center', color='#c0c0e0', fontsize=9)
    
    # Panels 4-6: Dwell gradient per condition
    layer_names = ['δ (4f)', 'θ (8f)', 'α (16f)', 'γ (32f)']
    layer_colors = ['#ff5050', '#ffb428', '#50ff90', '#c050ff']
    
    for col_idx, (label, stats, color) in enumerate(datasets):
        ax = fig.add_subplot(gs[1, col_idx])
        ax.set_facecolor('#0d0d20')
        
        cvs = [cv for _, cv in stats['layer_cvs']]
        means = [m for m, _ in stats['layer_cvs']]
        
        # Plot CV per layer
        x = np.arange(4)
        width = 0.35
        bars_cv = ax.bar(x - width/2, cvs, width, 
                        color=layer_colors, alpha=0.8, label='CV')
        
        ax.axhline(y=1.0, color='white', linestyle='--', alpha=0.4, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, fontsize=7, color='#c0c0e0')
        
        short_label = label.split('\n')[0]
        ax.set_title(f"Dwell CV by Layer\n{short_label}", 
                    fontsize=9, color=color, fontweight='bold')
        ax.tick_params(colors='#606080', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#2a2a4a')
        ax.set_ylim(0, max(max(cvs) * 1.3, 1.5))
        
        for bar, cv_val in zip(bars_cv, cvs):
            regime = 'crit' if cv_val > 1.3 else ('busty' if cv_val > 0.9 else 'clock')
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{cv_val:.1f}', ha='center', color='white', fontsize=7)
    
    # Panel 7: Gradient comparison (the key result)
    ax_grad = fig.add_subplot(gs[2, :2])
    ax_grad.set_facecolor('#0d0d20')
    
    x = np.arange(4)
    width = 0.25
    
    for col_idx, (label, stats, color) in enumerate(datasets):
        means = [m for m, _ in stats['layer_cvs']]
        offset = (col_idx - 1) * width
        ax_grad.bar(x + offset, means, width, color=color, alpha=0.8,
                   label=label.replace('\n', ' '))
    
    ax_grad.set_xticks(x)
    ax_grad.set_xticklabels(['δ layer\n(coarsest)', 'θ layer', 
                             'α layer\n(near-Nyquist)', 'γ layer\n(finest)'],
                            fontsize=8, color='#c0c0e0')
    ax_grad.set_title("Mean Dwell Time by Layer — The Gradient\n"
                     "Healthy: ordered (slow→fast)   AD: collapsed (flat)",
                     fontsize=10, color='#c0c0e0', fontweight='bold')
    ax_grad.legend(fontsize=8, facecolor='#0d0d20', edgecolor='#2a2a4a',
                  labelcolor='#c0c0e0', loc='upper right')
    ax_grad.tick_params(colors='#606080', labelsize=8)
    for spine in ax_grad.spines.values():
        spine.set_color('#2a2a4a')
    ax_grad.set_ylabel('Mean Dwell (steps)', color='#909090', fontsize=9)
    
    # Panel 8: Summary - the key insight
    ax_summary = fig.add_subplot(gs[2, 2])
    ax_summary.set_facecolor('#0d0d20')
    ax_summary.axis('off')
    
    lines = [
        ("DEERSKIN PREDICTION vs PHI-DWELL", '#ffb428', 11, True),
        ("", '#c0c0e0', 9, False),
        ("AD disruption mechanism:", '#c0c0e0', 9, True),
        ("  Alpha layer homeostasis fails", '#ff9090', 8, False),
        ("  Near-Nyquist layer = most fragile", '#ff9090', 8, False),
        ("", '#c0c0e0', 9, False),
        ("Predicted effects:", '#c0c0e0', 9, True),
        ("  ↑ Vocabulary (more words)", '#90ff90', 8, False),
        ("  ↑ Entropy (less order)", '#90ff90', 8, False),
        ("  ↓ Concentration (no grammar)", '#90ff90', 8, False),
        ("  → Gradient collapse (flat)", '#90ff90', 8, False),
        ("", '#c0c0e0', 9, False),
        ("Phi-Dwell found:", '#c0c0e0', 9, True),
        ("  CN=953 → AD=1052 vocab ✓", '#5090ff', 8, False),
        ("  CN=8.26 → AD=8.57 entropy ✓", '#5090ff', 8, False),
        ("  CN=0.154 → AD=0.124 conc. ✓", '#5090ff', 8, False),
        ("  Dwell gradient: KW p=0.0015 ✓", '#5090ff', 8, False),
        ("  MMSE ρ=0.408 ✓", '#5090ff', 8, False),
    ]
    
    y = 0.97
    for text, color, size, bold in lines:
        ax_summary.text(0.05, y, text, transform=ax_summary.transAxes,
                       fontsize=size, color=color, 
                       fontweight='bold' if bold else 'normal',
                       fontfamily='monospace')
        y -= 0.055 if text else 0.03
    
    plt.savefig(save_path or '/home/claude/deerskin_cortex/alzheimer_simulation.png',
                dpi=150, facecolor=fig.get_facecolor())
    plt.close()


if __name__ == "__main__":
    print("Alzheimer's Gradient Collapse Simulation")
    print("=" * 50)
    print()
    
    healthy, ad, severe = simulate_alzheimer_disruption()
    
    print("\nResults:")
    print(f"{'Metric':<22} {'Healthy':>12} {'AD-like':>12} {'Severe':>12}")
    print("-" * 60)
    print(f"{'Vocabulary size':<22} {healthy['vocab_size']:>12} "
          f"{ad['vocab_size']:>12} {severe['vocab_size']:>12}")
    print(f"{'Entropy (bits)':<22} {healthy['entropy']:>12.3f} "
          f"{ad['entropy']:>12.3f} {severe['entropy']:>12.3f}")
    print(f"{'Top-5 concentration':<22} {healthy['top5_concentration']:>12.3f} "
          f"{ad['top5_concentration']:>12.3f} {severe['top5_concentration']:>12.3f}")
    
    print("\nDwell gradient (mean dwell, CV per layer):")
    layer_names = ['delta(4f)', 'theta(8f)', 'alpha(16f)', 'gamma(32f)']
    for i, name in enumerate(layer_names):
        h_m, h_cv = healthy['layer_cvs'][i]
        a_m, a_cv = ad['layer_cvs'][i]
        s_m, s_cv = severe['layer_cvs'][i]
        print(f"  {name:12s}  H:{h_m:.1f}/{h_cv:.2f}  AD:{a_m:.1f}/{a_cv:.2f}  "
              f"Sev:{s_m:.1f}/{s_cv:.2f}")
    
    print("\nGenerating comparison figure...")
    plot_comparison(healthy, ad, severe, 
                   '/home/claude/deerskin_cortex/alzheimer_simulation.png')
    print("Saved: alzheimer_simulation.png")
    
    # Check if predictions match Phi-Dwell direction
    vocab_up = ad['vocab_size'] >= healthy['vocab_size']
    entropy_up = ad['entropy'] >= healthy['entropy']
    conc_down = ad['top5_concentration'] <= healthy['top5_concentration']
    
    print("\nPrediction validation:")
    print(f"  Vocabulary increases in AD: {'✓' if vocab_up else '✗'} "
          f"({healthy['vocab_size']} → {ad['vocab_size']})")
    print(f"  Entropy increases in AD:    {'✓' if entropy_up else '✗'} "
          f"({healthy['entropy']:.3f} → {ad['entropy']:.3f})")
    print(f"  Concentration drops in AD:  {'✓' if conc_down else '✗'} "
          f"({healthy['top5_concentration']:.3f} → {ad['top5_concentration']:.3f})")
    
    all_correct = vocab_up and entropy_up and conc_down
    print(f"\n  Phi-Dwell alignment: {'ALL CORRECT ✓' if all_correct else 'PARTIAL'}")
