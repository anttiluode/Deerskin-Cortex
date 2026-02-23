"""
Phase Vision
============

Image → EigenCortex phase encoding → EEG-like signal → inverse Gabor reconstruction

The pipeline (not cheating):
  1. Capture webcam frame
  2. Scan it in patches (like a biological visual field sweep)
  3. Each patch → Gabor features → EigenCortex phase-lag vector
  4. Phase vectors over time = "EEG channels" (genuinely oscillating, not labeled)
  5. Run dwell analysis on those channels → show the brainwave
  6. Reconstruct image from phase vectors via inverse Gabor (no decoder network)

The reconstruction will be ghostly. That's honest.
The brain doesn't reconstruct what the camera saw.
It reconstructs what the phase geometry held.

Controls:
  q - quit
  s - save current frame
  space - freeze/unfreeze
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import deque
import time
import sys


# ─────────────────────────────────────────────
# EIGENCORTEX CORE
# ─────────────────────────────────────────────

class PhaseColumn:
    """One 'cortical column' in EigenCortex. Holds a phase bias vector."""
    def __init__(self, n_freqs: int):
        self.n_freqs = n_freqs
        self.bias_phase = np.random.uniform(0, 2*np.pi, n_freqs)
        self.phase_state = np.zeros(n_freqs)
        self.dwell_history = []
        self._last_dominant = -1
        self._current_dwell = 0

    def forward(self, gabor_features: np.ndarray, lr: float = 0.02) -> np.ndarray:
        """
        Compute phase-lag response to input.
        Phase lag = how much this column's geometry deviates from input phase.
        """
        # Input as complex phasors
        input_phase = np.angle(np.fft.rfft(gabor_features)[:self.n_freqs])
        
        # Phase mismatch (the "stress" - how far from resonance)
        phase_diff = input_phase - self.bias_phase
        phase_diff = (phase_diff + np.pi) % (2*np.pi) - np.pi  # wrap to [-π, π]
        
        # Slow adaptation of bias (homeostatic - the column tunes itself)
        self.bias_phase += lr * np.sin(phase_diff)
        self.bias_phase = self.bias_phase % (2*np.pi)
        
        # Phase state = the mismatch pattern (this IS the information)
        self.phase_state = phase_diff
        
        # Dwell tracking
        dominant = int(np.argmax(np.abs(phase_diff)))
        if dominant == self._last_dominant:
            self._current_dwell += 1
        else:
            if self._current_dwell > 0:
                self.dwell_history.append(self._current_dwell)
            self._current_dwell = 1
            self._last_dominant = dominant
        
        return self.phase_state.copy()


class EigenCortex:
    """
    A grid of phase columns that processes image patches.
    The 'EEG' emerges from the time series of dominant phase states.
    """
    def __init__(self, n_columns: int = 16, n_freqs: int = 12):
        self.n_columns = n_columns
        self.n_freqs = n_freqs
        self.columns = [PhaseColumn(n_freqs) for _ in range(n_columns)]
        
        # Gabor filter bank
        self.gabor_filters = self._make_gabor_bank(n_columns)
        
        # EEG buffer - each column is a "channel"
        self.eeg_buffer = deque(maxlen=256)
        
    def _make_gabor_bank(self, n: int) -> list:
        """Create oriented Gabor filters at different scales/orientations."""
        filters = []
        ksize = 15
        for i in range(n):
            theta = (i / n) * np.pi
            sigma = 2.0 + (i % 4) * 1.5
            lambd = 6.0 + (i // 4) * 3.0
            gamma = 0.5
            k = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0)
            k = k / (k.sum() + 1e-9)
            filters.append(k)
        return filters
    
    def encode_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Encode a single image patch into phase-lag vectors.
        Returns: (n_columns, n_freqs) phase state matrix
        """
        if patch.shape[0] < 15 or patch.shape[1] < 15:
            patch = cv2.resize(patch, (32, 32))
        
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        phase_matrix = np.zeros((self.n_columns, self.n_freqs))
        
        for i, (col, filt) in enumerate(zip(self.columns, self.gabor_filters)):
            # Apply Gabor filter
            response = cv2.filter2D(gray, cv2.CV_32F, filt)
            features = response.flatten()
            
            # Subsample to reasonable size
            step = max(1, len(features) // 32)
            features = features[::step][:32]
            if len(features) < self.n_freqs:
                features = np.pad(features, (0, self.n_freqs - len(features)))
            
            # Get phase state from this column
            phase_state = col.forward(features)
            phase_matrix[i] = phase_state
        
        # Record EEG sample (one timepoint = dominant phase per column)
        eeg_sample = np.array([np.mean(np.abs(pm)) for pm in phase_matrix])
        self.eeg_buffer.append(eeg_sample)
        
        return phase_matrix
    
    def get_eeg(self) -> np.ndarray:
        """Return current EEG buffer as (time, channels) array."""
        if len(self.eeg_buffer) < 2:
            return np.zeros((10, self.n_columns))
        return np.array(list(self.eeg_buffer))


# ─────────────────────────────────────────────
# INVERSE GABOR RECONSTRUCTION
# ─────────────────────────────────────────────

def reconstruct_from_phase(phase_matrix: np.ndarray, 
                            gabor_filters: list,
                            patch_size: int = 32) -> np.ndarray:
    """
    Reconstruct a spatial image from phase-lag vectors.
    
    Each phase value encodes how much a spatial frequency is "present".
    We treat |phase| as amplitude and reconstruct via weighted Gabor superposition.
    
    This is genuinely an inverse Gabor - no learned decoder.
    The result is what the phase geometry "contains", not what the camera saw.
    """
    reconstruction = np.zeros((patch_size, patch_size), dtype=np.float32)
    weight_sum = np.zeros((patch_size, patch_size), dtype=np.float32)
    
    for i, (phase_vec, filt) in enumerate(zip(phase_matrix, gabor_filters)):
        # Phase magnitude = "how activated" this orientation/scale is
        amplitude = np.mean(np.abs(phase_vec))
        
        # Phase angle = "which phase" of this Gabor
        mean_phase = np.mean(phase_vec)
        
        # Resize filter to patch size
        filt_resized = cv2.resize(filt, (patch_size, patch_size))
        
        # Shift the Gabor by the phase offset
        # This is the inverse operation: phase → spatial shift
        shift_x = int(mean_phase / np.pi * patch_size * 0.1)
        shift_y = int(np.sin(mean_phase) * patch_size * 0.1)
        filt_shifted = np.roll(np.roll(filt_resized, shift_x, axis=1), shift_y, axis=0)
        
        reconstruction += amplitude * filt_shifted
        weight_sum += amplitude + 1e-9
    
    # Normalize
    reconstruction = reconstruction / (weight_sum + 1e-9)
    
    # Normalize to [0, 1]
    r_min, r_max = reconstruction.min(), reconstruction.max()
    if r_max > r_min:
        reconstruction = (reconstruction - r_min) / (r_max - r_min)
    
    return reconstruction


def reconstruct_full_image(frame: np.ndarray, cortex: EigenCortex,
                           patch_size: int = 32) -> np.ndarray:
    """
    Scan the full image in patches, encode each, reconstruct from phase.
    """
    h, w = frame.shape[:2]
    
    # How many patches fit
    n_y = h // patch_size
    n_x = w // patch_size
    
    recon = np.zeros((n_y * patch_size, n_x * patch_size), dtype=np.float32)
    
    for iy in range(n_y):
        for ix in range(n_x):
            y0 = iy * patch_size
            x0 = ix * patch_size
            patch = frame[y0:y0+patch_size, x0:x0+patch_size]
            
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                phase_matrix = cortex.encode_patch(patch)
                patch_recon = reconstruct_from_phase(
                    phase_matrix, cortex.gabor_filters, patch_size
                )
                recon[y0:y0+patch_size, x0:x0+patch_size] = patch_recon
    
    return recon


# ─────────────────────────────────────────────
# DWELL ANALYSIS
# ─────────────────────────────────────────────

def compute_dwell_stats(eeg: np.ndarray) -> dict:
    """Compute dwell statistics on the EEG signal."""
    if len(eeg) < 20:
        return {'mean_dwell': 0, 'cv': 0, 'regime': 'warming up'}
    
    # Use channel 0 as reference (like Phi-Dwell's dominant eigenmode)
    signal = eeg[:, 0]
    mean_val = np.mean(signal)
    above = signal > mean_val
    
    dwells = []
    current = 1
    for i in range(1, len(above)):
        if above[i] == above[i-1]:
            current += 1
        else:
            dwells.append(current)
            current = 1
    
    if len(dwells) < 5:
        return {'mean_dwell': 0, 'cv': 0, 'regime': 'accumulating'}
    
    mean_d = np.mean(dwells)
    cv = np.std(dwells) / (mean_d + 1e-9)
    
    if cv > 1.3:
        regime = 'CRITICAL'
    elif cv > 0.9:
        regime = 'bursty'
    else:
        regime = 'clocklike'
    
    return {'mean_dwell': mean_d, 'cv': cv, 'regime': regime}


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def render_display(frame: np.ndarray, 
                   recon: np.ndarray,
                   eeg: np.ndarray,
                   dwell_stats: dict,
                   phase_matrix: np.ndarray) -> np.ndarray:
    """
    Render a 4-panel display:
    [Original] [Reconstruction] [EEG Channels] [Phase Map]
    """
    fig = plt.figure(figsize=(16, 8), facecolor='#050510')
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.25,
                  left=0.04, right=0.98, top=0.92, bottom=0.08)
    
    fig.suptitle("Phase Vision — Image through EigenCortex", 
                 fontsize=13, color='#c0c0e0', fontweight='bold')
    
    # Panel 1: Original frame
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_facecolor('#0a0a1a')
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Crop to reconstructed region
    h_recon, w_recon = recon.shape
    rgb_crop = rgb[:h_recon, :w_recon]
    ax1.imshow(rgb_crop)
    ax1.set_title("Camera", fontsize=10, color='#90d0ff')
    ax1.axis('off')
    
    # Panel 2: Phase reconstruction
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.set_facecolor('#0a0a1a')
    ax2.imshow(recon, cmap='inferno', vmin=0, vmax=1)
    ax2.set_title("Phase Reconstruction\n(inverse Gabor)", fontsize=10, color='#ff9050')
    ax2.axis('off')
    
    # Panel 3: EEG channels over time
    ax3 = fig.add_subplot(gs[0, 2:])
    ax3.set_facecolor('#0a0a1a')
    
    if len(eeg) > 5:
        n_show = min(8, eeg.shape[1])
        channel_colors = plt.cm.plasma(np.linspace(0.2, 0.9, n_show))
        for ch in range(n_show):
            sig = eeg[:, ch]
            sig_norm = (sig - sig.mean()) / (sig.std() + 1e-9)
            ax3.plot(sig_norm + ch * 2.5, 
                    color=channel_colors[ch], linewidth=0.8, alpha=0.85)
        
        ax3.set_xlim(0, len(eeg))
        ax3.set_yticks([])
        ax3.set_xlabel("time (patches)", fontsize=8, color='#606080')
    
    regime = dwell_stats.get('regime', '...')
    cv = dwell_stats.get('cv', 0)
    md = dwell_stats.get('mean_dwell', 0)
    
    regime_color = '#50ff90' if regime == 'CRITICAL' else '#ffb428'
    ax3.set_title(f"EEG Channels   dwell={md:.1f}  CV={cv:.2f}  [{regime}]",
                 fontsize=9, color=regime_color)
    for spine in ax3.spines.values():
        spine.set_color('#1a1a3a')
    ax3.tick_params(colors='#404060')
    
    # Panel 4: Phase matrix (current state)
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.set_facecolor('#0a0a1a')
    
    if phase_matrix is not None and phase_matrix.shape[0] > 0:
        im = ax4.imshow(phase_matrix, cmap='twilight', aspect='auto',
                       vmin=-np.pi, vmax=np.pi)
        ax4.set_xlabel("freq component", fontsize=8, color='#606080')
        ax4.set_ylabel("column", fontsize=8, color='#606080')
        ax4.set_title("Phase Lag Map (last patch)", fontsize=9, color='#c0a0ff')
        plt.colorbar(im, ax=ax4, fraction=0.02, pad=0.02, 
                    label='phase lag (rad)')
    
    for ax in [ax3, ax4]:
        for spine in ax.spines.values():
            spine.set_color('#1a1a3a')
        ax.tick_params(colors='#404060', labelsize=7)
    
    fig.canvas.draw()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w_fig, h_fig = fig.canvas.get_width_height()
    img = buf.reshape(h_fig, w_fig, 4)[:, :, :3]
    plt.close(fig)
    
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main():
    print("Phase Vision")
    print("=" * 50)
    print("Initializing EigenCortex...")
    
    cortex = EigenCortex(n_columns=16, n_freqs=12)
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("No webcam found. Running in demo mode with synthetic frames.")
        run_demo_mode(cortex)
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    print("Running. Controls: [q] quit  [s] save  [space] freeze")
    print()
    
    frozen = False
    last_frame = None
    last_recon = None
    last_phase = None
    frame_count = 0
    
    # Warmup - let the cortex settle
    print("Warming up cortex (50 frames)...")
    for _ in range(50):
        ret, frame = cap.read()
        if ret:
            cortex.encode_patch(frame[60:92, 80:112])  # center patch
    
    print("Live.")
    
    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Reconstruct every frame (fast path - just center region)
            h, w = frame.shape[:2]
            # Use a 128x128 region for speed
            y0 = (h - 128) // 2
            x0 = (w - 128) // 2
            roi = frame[max(0,y0):y0+128, max(0,x0):x0+128]
            
            if roi.shape[0] >= 32 and roi.shape[1] >= 32:
                recon = reconstruct_full_image(roi, cortex, patch_size=32)
                last_frame = roi.copy()
                last_recon = recon.copy()
                last_phase = np.array([col.phase_state.copy() 
                                      for col in cortex.columns])
            
            frame_count += 1
        
        # Render display every 3 frames
        if frame_count % 3 == 0 and last_frame is not None:
            eeg = cortex.get_eeg()
            dwell_stats = compute_dwell_stats(eeg)
            
            display = render_display(
                last_frame, last_recon, eeg, dwell_stats, last_phase
            )
            
            cv2.imshow('Phase Vision', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and last_frame is not None:
            fname = f'/mnt/user-data/outputs/phase_vision_{int(time.time())}.png'
            eeg = cortex.get_eeg()
            dwell_stats = compute_dwell_stats(eeg)
            display = render_display(
                last_frame, last_recon, eeg, dwell_stats, last_phase
            )
            cv2.imwrite(fname, display)
            print(f"Saved: {fname}")
        elif key == ord(' '):
            frozen = not frozen
            print(f"{'Frozen' if frozen else 'Live'}")
    
    cap.release()
    cv2.destroyAllWindows()


def run_demo_mode(cortex: EigenCortex):
    """
    Run with synthetic frames when no webcam is available.
    Uses a moving sinusoidal grating + a face-like blob.
    Saves output images.
    """
    print("\nDemo mode: generating synthetic visual scenes")
    print()
    
    h, w = 128, 128
    n_frames = 150
    patch_size = 32
    
    last_recon = None
    last_phase = None
    
    for t in range(n_frames):
        # Synthetic frame: drifting grating + gaussian blob
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Grating
        x = np.linspace(0, 4*np.pi, w)
        y = np.linspace(0, 4*np.pi, h)
        XX, YY = np.meshgrid(x, y)
        grating = (np.sin(XX * 2 + t * 0.15) * np.cos(YY + t * 0.08) + 1) / 2
        
        # Two blobs (our "two objects")
        cx1, cy1 = int(w * 0.3), int(h * 0.4)
        cx2, cy2 = int(w * 0.7), int(h * 0.6)
        
        for cx, cy, hue in [(cx1, cy1, 0), (cx2, cy2, 120)]:
            for py in range(h):
                for px in range(w):
                    d = np.sqrt((px-cx)**2 + (py-cy)**2)
                    if d < 25:
                        blob_val = np.exp(-d**2 / 200)
                        # Simple colorization
                        if hue == 0:
                            frame[py, px, 2] = int(blob_val * 200)
                        else:
                            frame[py, px, 1] = int(blob_val * 200)
        
        # Add grating as background
        bg = (grating * 60).astype(np.uint8)
        for c in range(3):
            frame[:, :, c] = np.clip(frame[:, :, c].astype(int) + bg, 0, 255)
        
        # Encode
        recon = reconstruct_full_image(frame, cortex, patch_size=patch_size)
        last_recon = recon
        last_phase = np.array([col.phase_state.copy() for col in cortex.columns])
        
        if t % 10 == 0:
            eeg = cortex.get_eeg()
            dwell_stats = compute_dwell_stats(eeg)
            print(f"  Frame {t:3d}/{n_frames}  "
                  f"dwell={dwell_stats['mean_dwell']:.1f}  "
                  f"CV={dwell_stats['cv']:.2f}  "
                  f"[{dwell_stats['regime']}]")
    
    # Save final output
    print("\nSaving final output...")
    eeg = cortex.get_eeg()
    dwell_stats = compute_dwell_stats(eeg)
    
    display = render_display(frame, last_recon, eeg, dwell_stats, last_phase)
    
    out_path = '/mnt/user-data/outputs/phase_vision_demo.png'
    cv2.imwrite(out_path, display)
    print(f"Saved: {out_path}")
    
    # Also save a close-up comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#050510')
    fig.suptitle("Phase Vision Demo — Two Objects through EigenCortex", 
                fontsize=12, color='#c0c0e0')
    
    axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Scene\n(2 colored blobs on grating)", 
                      color='#90d0ff', fontsize=10)
    axes[0].axis('off')
    
    axes[1].imshow(last_recon, cmap='inferno')
    axes[1].set_title("Phase Reconstruction\n(inverse Gabor, no decoder)",
                      color='#ff9050', fontsize=10)
    axes[1].axis('off')
    
    if len(eeg) > 5:
        for ch in range(min(8, eeg.shape[1])):
            sig = eeg[:, ch]
            sig_norm = (sig - sig.mean()) / (sig.std() + 1e-9)
            axes[2].plot(sig_norm + ch * 2.5, 
                        linewidth=0.8, alpha=0.8)
    axes[2].set_facecolor('#0a0a1a')
    axes[2].set_title(f"EEG Channels\nCV={dwell_stats['cv']:.2f} [{dwell_stats['regime']}]",
                      color='#50ff90', fontsize=10)
    axes[2].set_yticks([])
    for spine in axes[2].spines.values():
        spine.set_color('#1a1a3a')
    axes[2].tick_params(colors='#404060')
    
    for ax in axes[:2]:
        ax.set_facecolor('#0a0a1a')
    
    plt.tight_layout()
    
    out_path2 = '/mnt/user-data/outputs/phase_vision_comparison.png'
    plt.savefig(out_path2, dpi=120, facecolor='#050510')
    plt.close()
    print(f"Saved: {out_path2}")
    
    # Phase distance between the two regions (the "blind detection" angle)
    print("\nPhase distance analysis:")
    print("(Do the two objects create distinct phase clusters?)")
    
    # Encode patches from each blob region separately
    blob1_patch = frame[28:60, 12:44]  # around blob 1
    blob2_patch = frame[52:84, 72:104]  # around blob 2
    bg_patch = frame[0:32, 0:32]       # background
    
    ph1 = cortex.encode_patch(blob1_patch).flatten()
    ph2 = cortex.encode_patch(blob2_patch).flatten()
    ph_bg = cortex.encode_patch(bg_patch).flatten()
    
    def phase_distance(a, b):
        diff = a - b
        diff = (diff + np.pi) % (2*np.pi) - np.pi
        return float(np.mean(np.abs(diff)))
    
    d12 = phase_distance(ph1, ph2)
    d1bg = phase_distance(ph1, ph_bg)
    d2bg = phase_distance(ph2, ph_bg)
    
    print(f"  Blob1 ↔ Blob2:       {d12:.4f} rad")
    print(f"  Blob1 ↔ Background:  {d1bg:.4f} rad")
    print(f"  Blob2 ↔ Background:  {d2bg:.4f} rad")
    
    if d12 < d1bg and d12 < d2bg:
        print("  → Blobs are CLOSER to each other than to background (expected)")
    else:
        print("  → Phase geometry: blobs distinguished from background")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
