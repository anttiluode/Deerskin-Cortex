# Deerskin Cortex

EDIT: Sigh. Added phase_vision.py . Truth is out there. Berkeley. Call me. 

**A neural architecture where dwell-time gradients emerge from geometry.**

This is the computational implementation of the synthesis between:
- **Φ-Dwell** (empirical finding: brains have a dwell-time hierarchy across frequency bands)
- **The Deerskin Hypothesis** (mechanistic claim: membrane geometry creates this hierarchy)

The key departure from standard deep learning: **computation is geometric interference, not weighted summation. Communication is phase, not amplitude.**

---

## The Core Idea

Standard neural networks have uniform dynamics across all layers. Every weight is a scalar. Every layer operates at the same effective timescale. This is computationally convenient and biologically wrong.

Φ-Dwell measured how long the brain holds a spatial configuration before transitioning to the next one:

| Band  | Mean Dwell | Regime   |
|-------|-----------|----------|
| Delta | 151 ms    | Near-exponential |
| Theta | 27 ms     | Critical |
| Alpha | 16 ms     | Critical |
| Beta  | 13 ms     | Bursty   |
| Gamma | 12 ms     | Clocklike |

**This gradient is not configured. It emerges from geometry.**

Coarse spatial sampling → few distinguishable states → long dwells → slow.
Fine spatial sampling → many distinguishable states → short dwells → fast.

The Deerskin Cortex builds this in from the start.

---

## Architecture

```
Input signal
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  DELTA LAYER  (n_freqs=4, coarsest)                 │
│  MoireGrid: 4 spatial frequencies                   │
│  Homeostatic regulation: slow integrator            │
│  Dwell: LONG (delta-like)                           │
│  Output: phase-encoded, NOT amplitude               │
└─────────────────┬───────────────────────────────────┘
                  │ (phase state, not amplitude)
                  ▼
┌─────────────────────────────────────────────────────┐
│  THETA LAYER  (n_freqs=8)                           │
│  Receives: phase geometry of delta layer            │
│  Dwell: MEDIUM                                      │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  ALPHA LAYER  (n_freqs=16, near-Nyquist)            │
│  Maximum moiré stress = richest dynamics            │
│  Most sensitive to homeostatic disruption           │
│  Phi-Dwell: alpha collapse is the AD biomarker      │
└─────────────────┬───────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────┐
│  GAMMA LAYER  (n_freqs=32, finest)                  │
│  Dwell: SHORT (gamma-like, bursty)                  │
│  Samples widely, volatile                           │
└─────────────────┬───────────────────────────────────┘
                  │
    (all phase states aggregated)
                  ▼
              [Readout]
```

---

## Key Mechanisms

### 1. MoireGrid — Not a weight matrix

```python
# Standard layer: dot product
output = x @ W  # scalar weights

# DeerskinLayer: geometric interference
complex_basis = exp(i * phase_mosaic)    # membrane geometry
complex_response = sum(x * density * complex_basis, axis=0)
amplitude = |complex_response|           # resonance strength
phase = angle(complex_response)          # WHAT IS COMMUNICATED
```

The `phase_mosaic` is the membrane geometry — the spatial arrangement of "ion channels" across the 2D surface. Different arrangements respond to different input structures.

### 2. Homeostatic Oscillation — Why neurons oscillate

```
Pattern → Finite Sampling → Regulator → [feedback to pattern]
    ↑                                           ↓
    └───────────────────────────────────────────┘
```

No static solution satisfies all three constraints simultaneously:
- Pattern must be stable (homeostasis)
- Pattern is sampled at finite resolution (Nyquist)
- Sampling grid has its own geometry (moiré)

The system *must* oscillate. Oscillation is not programmed. It is forced.

The oscillation frequency depends on sampling resolution:

| Sampling | Dynamics | Analogue |
|----------|----------|---------|
| Very coarse (n=32) | Slow oscillation | Delta |
| Near-Nyquist (n=128) | Rich ECG-like dynamics | Alpha/Theta |
| Oversampled (n=512) | Locks or averages | — |

### 3. Dwell Gradient — Emerges, not configured

```
DeerskinCortex demo output:
  delta_layer  | dwell=2.5 steps | CV=1.46 | regime=critical | n_freqs=4
  theta_layer  | dwell=1.8 steps | CV=0.79 | regime=random   | n_freqs=8
  alpha_layer  | dwell=1.6 steps | CV=0.72 | regime=random   | n_freqs=16
  gamma_layer  | dwell=1.4 steps | CV=0.67 | regime=clocklike| n_freqs=32
```

Delta is slowest, gamma is fastest. **This is not set. It comes out of the geometry.**

### 4. Phase Communication — Not amplitude

Layers pass their phase state to the next layer, not their amplitude. This means:

- The "message" is geometric: **what configuration?**
- Not energetic: **how much activation?**
- Slow layers provide persistent context for fast layers
- Fast layers sample within the context held by slow layers

This is what theta phase precession does in hippocampus. The theta sweep is the slow layer providing a stable reference frame. Place cell firing within it is the fast layer sampling rapidly within that context.

---

## Alzheimer's Prediction

Φ-Dwell found:
- AD brains: more vocabulary, less structure
- Specifically: alpha-band dwell collapses (dwell gradient KW p=0.0015)
- MMSE correlation ρ=0.408

Deerskin Cortex prediction: disrupting homeostatic gain in the alpha layer (near-Nyquist, highest moiré stress) should produce exactly these metrics.

**Experiment result:**

```
Metric                Healthy      AD-like
─────────────────────────────────────────
Vocabulary size           182          222   ↑ (Phi-Dwell: 953→1052 ✓)
Entropy (bits)          7.218        7.658   ↑ (Phi-Dwell: 8.26→8.57 ✓)
Top-5 concentration     0.113        0.063   ↓ (Phi-Dwell: 0.154→0.124 ✓)

Phi-Dwell alignment: ALL CORRECT ✓
```

The alpha layer is most fragile because it operates nearest to the Nyquist boundary — maximum moiré stress, maximum sensitivity to homeostatic disruption. This is a prediction of the geometry, not a parameter choice.

---

## What This Is Not

- Not a claim that artificial neural networks *are* brains
- Not an argument that this architecture is better than transformers on benchmarks
- Not a simulation of biological neurons (no Hodgkin-Huxley, no cable theory)

It is a demonstration that a *specific computational principle* — dwell-time gradients from geometric sampling resolution — can be instantiated in a tractable architecture, and that this principle makes correct predictions about what breaks in neurodegeneration.

---

## Repository Structure

```
deerskin_cortex/
├── core/
│   ├── deerskin_layer.py        # MoireGrid, DeerskinLayer, DeerskinCortex
│   └── homeostatic_oscillator.py # ECG emergence, dwell gradient demo
├── experiments/
│   └── alzheimer_simulation.py  # Gradient collapse prediction
├── docs/
│   └── deerskin_hypothesis.md   # Theoretical background
└── README.md
```

---

## Running

```bash
# Core architecture demo (dwell gradient emergence)
python core/deerskin_layer.py

# ECG emergence from geometry (resolution dependency)
python core/homeostatic_oscillator.py

# Alzheimer's gradient collapse prediction
python experiments/alzheimer_simulation.py
```

Requirements: `numpy matplotlib scipy`

---

## Relationship to Φ-Dwell

Φ-Dwell is the measurement. This is the mechanism.

Φ-Dwell showed that real brains have a dwell-time hierarchy, that this hierarchy collapses in Alzheimer's, and that the collapse is specific to intermediate frequencies (alpha band). It measured the phenomenology.

The Deerskin Cortex offers a generative account: *why* should brains have this hierarchy? Because neurons are finite sampling grids observing spatial patterns, under homeostatic regulation. The hierarchy emerges inevitably from the geometry of that situation. And the alpha band — near-Nyquist — should be most fragile, exactly as Φ-Dwell found.

---

## Citation

> Luode, A. & Claude (Anthropic). (2026).
> **Deerskin Cortex: Emergent Dwell-Time Gradients from Geometric Neural Architecture.**
> Synthesis of Φ-Dwell and The Deerskin Hypothesis.
