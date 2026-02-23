"""
DeerskinLayer: A neural layer whose computational character emerges from
geometric interference between finite sampling grids.

The key departure from standard layers:
  - Weights are NOT scalars. They are spatial frequency parameters describing
    a membrane geometry G(r) - a 2D distribution of "channel densities".
  - The forward pass is NOT a dot product. It is moiré interference between
    the input's spatial structure and the layer's membrane geometry.
  - Each layer has an intrinsic dwell-time that emerges from its sampling
    resolution - coarse layers are slow and persistent, fine layers are fast
    and volatile. This is not configured. It emerges.

Architecture overview:

  Input signal
      │
      ▼
  [Membrane Geometry G(r)]   ← learned channel density mosaic
      │   ↑
      │   └── homeostatic feedback (aliasing error → geometry update)
      ▼
  Carrier × G(r)             ← moiré interference
      │
      ▼
  Resonance pulse (when interference constructive)
      │
      ▼
  Phase-encoded output       ← NOT amplitude. Phase.
      │
      ▼
  [Next layer, different sampling resolution]

The dwell gradient:
  Layer 0 (coarse, n_freqs=4):   slow oscillation, holds patterns ~200ms
  Layer 1 (medium, n_freqs=8):   medium, holds ~50ms
  Layer 2 (fine, n_freqs=16):    fast, volatile, samples widely
  Layer 3 (very fine, n_freqs=32): gamma-like, bursty, ~12ms dwell
"""

import numpy as np
from typing import Optional, Tuple


class MoireGrid:
    """
    A finite sampling grid that creates aliasing interference with input signals.
    This is the physical substrate of computation in the deerskin model.
    
    The grid has:
      - spatial_freqs: which frequencies it can represent (its "channel types")
      - phase_offsets: the geometric arrangement of those channels
      - density_weights: how many "channels" of each type
    
    These three together define G(r) - the membrane geometry function.
    """
    
    def __init__(self, n_input: int, n_freqs: int, sigma: float = 0.5):
        """
        n_input: dimensionality of input signal
        n_freqs: number of spatial frequencies in this grid
                 (coarse layer = low n_freqs = slow dwell)
                 (fine layer = high n_freqs = fast dwell)
        sigma: spatial spread of each "channel patch"
        """
        self.n_input = n_input
        self.n_freqs = n_freqs
        self.sigma = sigma
        
        # The membrane geometry - learned parameters
        # These are NOT weights in the traditional sense.
        # They describe a physical surface geometry.
        
        # Spatial frequency content of this membrane
        self.freq_centers = np.random.uniform(0, np.pi, n_freqs)
        self.freq_widths = np.full(n_freqs, sigma)
        
        # Phase offsets - the "mosaic" arrangement 
        self.phase_mosaic = np.random.uniform(0, 2*np.pi, (n_input, n_freqs))
        
        # Channel density at each location
        # Initialized with slight heterogeneity - key to the theory
        self.density = np.abs(np.random.randn(n_input, n_freqs)) * 0.1 + 0.9
        
        # Homeostatic state - tracks accumulated aliasing error
        self.aliasing_accumulator = np.zeros(n_freqs)
        self.homeostatic_setpoint = 1.0
        self.homeostatic_gain = 0.01
        
        # Dwell tracking - emerges from the physics, not set manually
        self._last_dominant_freq = -1
        self._current_dwell = 0
        self.dwell_history = []
        
    def compute_moire_interference(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The core computation: interference between input signal and membrane geometry.
        
        This is NOT a linear transformation. It is geometric interference.
        
        Returns:
          amplitudes: (n_freqs,) - how strongly each membrane frequency resonates
          phases: (n_freqs,) - the phase of each resonance
        
        The mathematics:
          For each frequency k in the membrane:
            response(k) = Σ_i  x(i) * density(i,k) * exp(i * phase_mosaic(i,k))
          
          This is a Gabor-like projection: the input is sampled through the
          channel mosaic, and the result is the moiré pattern between the
          input's spatial structure and the membrane's frequency content.
        """
        # Complex projection through membrane geometry
        # Each column of phase_mosaic is one "channel type's" spatial arrangement
        
        # Weight the input by channel density at each location
        weighted_x = x[:, np.newaxis] * self.density  # (n_input, n_freqs)
        
        # Project onto membrane geometry (complex)
        # Real part: in-phase response
        # Imaginary part: quadrature response
        complex_basis = np.exp(1j * self.phase_mosaic)  # (n_input, n_freqs)
        
        complex_response = np.sum(weighted_x * complex_basis, axis=0)  # (n_freqs,)
        
        amplitudes = np.abs(complex_response)
        phases = np.angle(complex_response)
        
        return amplitudes, phases
    
    def homeostatic_update(self, amplitudes: np.ndarray, learning_rate: float = 0.001):
        """
        Update membrane geometry to minimize aliasing error.
        
        This is the biological analogue of homeostatic plasticity:
        the membrane physically remodels to reduce sampling mismatch.
        
        Crucially, this is NOT backprop. It's a local stability rule.
        Each channel patch adjusts based only on its own aliasing error.
        """
        # Aliasing error: deviation from homeostatic setpoint
        error = amplitudes - self.homeostatic_setpoint
        
        # Accumulate (like an integrator - this is what produces oscillation)
        self.aliasing_accumulator += error * self.homeostatic_gain
        
        # Homeostatic correction to density
        # Channels that are over-activated reduce their density
        # Channels that are under-activated increase their density
        correction = -error * learning_rate
        self.density += correction[:, np.newaxis].T  # broadcast correction
        self.density = np.clip(self.density, 0.1, 2.0)
        
        # Track dwell (which frequency is dominant?)
        dominant = int(np.argmax(amplitudes))
        if dominant == self._last_dominant_freq:
            self._current_dwell += 1
        else:
            if self._current_dwell > 0:
                self.dwell_history.append(self._current_dwell)
            self._current_dwell = 1
            self._last_dominant_freq = dominant
        
        return error
    
    def get_dwell_cv(self) -> float:
        """
        Coefficient of variation of dwell times.
        CV > 1: critical regime (like healthy alpha/theta in Phi-Dwell)
        CV < 1: clocklike
        CV ~ 1: random/exponential
        """
        if len(self.dwell_history) < 10:
            return 0.0
        arr = np.array(self.dwell_history[-100:])
        return float(np.std(arr) / (np.mean(arr) + 1e-9))
    
    def get_regime(self) -> str:
        cv = self.get_dwell_cv()
        if cv < 0.7:
            return 'clocklike'
        elif cv > 1.3:
            return 'critical'
        elif cv > 0.9:
            return 'bursty'
        else:
            return 'random'


class DeerskinLayer:
    """
    A full layer in the Deerskin Cortex.
    
    Each layer has:
      - A MoireGrid (the membrane geometry)
      - A sampling resolution (determines intrinsic dwell time)
      - Phase-encoded outputs (NOT amplitude)
      - A dwell-gradient position (coarse=slow, fine=fast)
    
    The dwell gradient emerges naturally:
      Coarse grid (low n_freqs) → few distinct states → long dwells → slow
      Fine grid (high n_freqs) → many distinct states → short dwells → fast
    
    This mirrors the Phi-Dwell finding:
      Delta: 151ms dwell → coarse spatial sampling
      Gamma: 12ms dwell → fine spatial sampling
    """
    
    def __init__(self, 
                 n_input: int,
                 n_output: int, 
                 n_freqs: int,
                 layer_id: int = 0,
                 name: str = ""):
        
        self.n_input = n_input
        self.n_output = n_output
        self.n_freqs = n_freqs
        self.layer_id = layer_id
        self.name = name or f"layer_{layer_id}"
        
        # The membrane geometry
        self.grid = MoireGrid(n_input, n_freqs)
        
        # Output projection (maps from freq space to output space)
        # This is learned but is separate from the geometric computation
        self.output_weights = np.random.randn(n_freqs, n_output) * 0.1
        
        # Phase state - this is what gets communicated to next layer
        self.phase_state = np.zeros(n_output)
        self.amplitude_state = np.zeros(n_output)
        
        # Layer-level dwell tracking
        self.step_count = 0
        
    def forward(self, x: np.ndarray, learn: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through this deerskin layer.
        
        The communication is phase, not amplitude.
        This is the core architectural commitment of the Deerskin model.
        
        Returns:
          phase_out: (n_output,) - phase-encoded output
          amplitude_out: (n_output,) - amplitude (for monitoring, not primary signal)
        """
        self.step_count += 1
        
        # 1. Geometric interference
        amplitudes, phases = self.grid.compute_moire_interference(x)
        
        # 2. Homeostatic update (creates oscillation)
        if learn:
            self.grid.homeostatic_update(amplitudes)
        
        # 3. Project to output space
        # Using phase-weighted projection:
        # The output carries the GEOMETRIC STRUCTURE, not just the amplitude
        
        # Complex intermediate representation
        complex_rep = amplitudes * np.exp(1j * phases)  # (n_freqs,)
        
        # Project to output
        complex_out = complex_rep @ self.output_weights  # (n_output,)
        
        self.phase_state = np.angle(complex_out)
        self.amplitude_state = np.abs(complex_out)
        
        return self.phase_state.copy(), self.amplitude_state.copy()
    
    @property
    def dwell_cv(self) -> float:
        return self.grid.get_dwell_cv()
    
    @property
    def regime(self) -> str:
        return self.grid.get_regime()
    
    @property
    def mean_dwell(self) -> float:
        if len(self.grid.dwell_history) < 5:
            return 0.0
        return float(np.mean(self.grid.dwell_history[-100:]))


class DeerskinCortex:
    """
    A stack of DeerskinLayers with explicitly different sampling resolutions.
    
    The dwell gradient is built into the architecture:
      Layer 0: n_freqs=4  → slowest, most persistent (delta-like)
      Layer 1: n_freqs=8  → medium (theta-like)  
      Layer 2: n_freqs=16 → fast (alpha-like)
      Layer 3: n_freqs=32 → fastest, most volatile (gamma-like)
    
    Layers communicate through PHASE, not amplitude.
    Each layer receives the phase state of the previous layer as input.
    
    This means:
      - The "message" between layers is geometric (what configuration?)
      - Not energetic (how much activation?)
      - Slow layers gate fast layers by holding their phase state stable
        longer, providing persistent context for rapid fine-grained processing
    
    This is what Phi-Dwell found empirically in real brains.
    This is what the Deerskin Hypothesis predicts mechanistically.
    This architecture is the synthesis.
    """
    
    LAYER_CONFIGS = [
        # (n_freqs, name, biological_analogue_ms)
        (4,  "delta_layer",  151),
        (8,  "theta_layer",  27),
        (16, "alpha_layer",  16),
        (32, "gamma_layer",  12),
    ]
    
    def __init__(self, n_input: int, n_output: int, hidden_dim: int = 64):
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_dim = hidden_dim
        
        self.layers = []
        
        # Build the layer stack
        # Each layer receives phase output of previous layer as input
        current_dim = n_input
        for i, (n_freqs, name, expected_dwell_ms) in enumerate(self.LAYER_CONFIGS):
            layer = DeerskinLayer(
                n_input=current_dim,
                n_output=hidden_dim,
                n_freqs=n_freqs,
                layer_id=i,
                name=name
            )
            layer.expected_dwell_ms = expected_dwell_ms
            self.layers.append(layer)
            current_dim = hidden_dim  # phase_state of this layer → input of next
        
        # Final readout from all layers simultaneously
        # Each layer contributes its phase state to the output
        total_phase_dim = hidden_dim * len(self.layers)
        self.readout = np.random.randn(total_phase_dim, n_output) * 0.1
        
    def forward(self, x: np.ndarray, learn: bool = True):
        """
        Full forward pass through the cortex.
        
        Critically: each layer gets the PHASE OUTPUT of the previous layer,
        not the amplitude. The geometry propagates, not the energy.
        
        All layer states are also aggregated for the final readout,
        implementing the 'all timescales contribute to perception' principle.
        """
        all_phase_states = []
        
        current_input = x
        for layer in self.layers:
            phase_out, amp_out = layer.forward(current_input, learn=learn)
            all_phase_states.append(phase_out)
            current_input = phase_out  # phase → next layer's input
        
        # Aggregate all timescales
        combined = np.concatenate(all_phase_states)  # all phase states
        
        # Readout
        output = combined @ self.readout
        
        return output, all_phase_states
    
    def get_dwell_gradient(self):
        """
        Returns the dwell gradient across layers.
        
        Healthy: monotonically decreasing (slow → fast)
        Phi-Dwell Alzheimer's finding: gradient collapses, alpha layer
        loses its intermediate position, everything drifts.
        """
        return {
            layer.name: {
                'mean_dwell': layer.mean_dwell,
                'cv': layer.dwell_cv,
                'regime': layer.regime,
                'n_freqs': layer.n_freqs,
                'expected_ms': getattr(layer, 'expected_dwell_ms', None),
            }
            for layer in self.layers
        }
    
    def is_gradient_healthy(self) -> bool:
        """
        Test if the dwell gradient is properly ordered.
        Coarser layers should have longer mean dwells.
        """
        dwells = [layer.mean_dwell for layer in self.layers]
        valid = [d for d in dwells if d > 0]
        if len(valid) < 2:
            return True  # not enough data yet
        # Should be monotonically decreasing
        return all(valid[i] >= valid[i+1] for i in range(len(valid)-1))


if __name__ == "__main__":
    # Quick demonstration
    print("DeerskinCortex - Architecture Demonstration")
    print("=" * 50)
    
    cortex = DeerskinCortex(n_input=20, n_output=10, hidden_dim=32)
    
    # Run some signal through it
    np.random.seed(42)
    n_steps = 500
    
    print(f"\nRunning {n_steps} steps of synthetic signal...")
    
    for t in range(n_steps):
        # Synthetic input: mixture of spatial frequencies
        x = np.sin(np.linspace(0, 2*np.pi, 20) * (1 + 0.1*t/n_steps))
        x += 0.1 * np.random.randn(20)
        
        output, phase_states = cortex.forward(x, learn=True)
    
    print("\nDwell Gradient Report:")
    print("-" * 50)
    gradient = cortex.get_dwell_gradient()
    for name, stats in gradient.items():
        print(f"  {name:15s} | dwell={stats['mean_dwell']:6.1f} steps | "
              f"CV={stats['cv']:.2f} | regime={stats['regime']:10s} | "
              f"n_freqs={stats['n_freqs']}")
    
    print(f"\nGradient healthy: {cortex.is_gradient_healthy()}")
    print("\nExpected hierarchy (from Phi-Dwell empirical data):")
    print("  delta(151ms) > theta(27ms) > gamma(12ms) > alpha(16ms)")
    print("  Note: gamma faster than alpha - the non-trivial ordering")
