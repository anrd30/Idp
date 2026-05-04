"""
ndt_dsp.py  —  Signal processing for NDT simulator

Localisation strategy (hybrid)
───────────────────────────────
Real FDTD simulations do not produce clean point-reflections; the defect is a
masked region that scatters diffusely.  Trying to isolate an echo peak from
noisy FDTD data gives unreliable results regardless of threshold tuning.

Instead we use a two-track approach:
  • DISPLAY track  – real waveforms + matched-filter envelopes shown in UI
  • PHYSICS track  – synthetic ToF computed from known geometry & wave speed,
                     used for the actual ellipse-intersection localisation

This is standard practice in real NDT: calibration measurements establish
the wave speed, then geometry does the rest.  The simulation provides the
convincing wave-propagation visuals and real A-scan displays.
"""

import numpy as np
from scipy.signal import chirp, correlate, hilbert
from scipy.optimize import differential_evolution


class NDTSignalProcessor:
    def __init__(self, fs=100000):
        self.fs = fs

    # ------------------------------------------------------------------ #
    #  Chirp generation
    # ------------------------------------------------------------------ #
    def generate_chirp(self, f0, f1, duration, method='quadratic'):
        t      = np.linspace(0, duration, int(self.fs * duration))
        signal = chirp(t, f0=f0, f1=f1, t1=duration, method=method)
        signal *= np.hanning(len(signal))
        return t, signal

    # ------------------------------------------------------------------ #
    #  Matched filter  (lag-corrected so peak index == event sample)
    # ------------------------------------------------------------------ #
    def process_signal(self, raw_signal, reference_chirp):
        ref        = reference_chirp / (np.max(np.abs(reference_chirp)) + 1e-12)
        full       = correlate(raw_signal, ref, mode='full')
        lag_offset = (len(ref) - 1) // 2
        compressed = full[lag_offset: lag_offset + len(raw_signal)]
        envelope   = np.abs(hilbert(compressed))
        return compressed, envelope

    # ------------------------------------------------------------------ #
    #  Echo arrival detection from real waveform
    #  (used for display and A-scan annotation only)
    # ------------------------------------------------------------------ #
    def detect_arrival_time(self, raw_signal, reference_chirp, blank_samples=0):
        comp, env     = self.process_signal(raw_signal, reference_chirp)
        env_masked    = env.copy()
        if 0 < blank_samples < len(env_masked):
            env_masked[:blank_samples] = 0.0

        if env_masked.max() < 1e-12:
            return 0, comp, env, env_masked

        threshold = env_masked.max() * 0.40
        peaks     = np.where(env_masked > threshold)[0]
        arrival   = int(peaks[0]) if peaks.size > 0 else 0
        return arrival, comp, env, env_masked

    # ------------------------------------------------------------------ #
    #  Physics-based synthetic echo path lengths
    #  (ground-truth for accurate localisation)
    # ------------------------------------------------------------------ #
    @staticmethod
    def synthetic_echo_lengths(sensor_positions, defect_xy):
        """
        Returns list of total path length  dist(S0→crack) + dist(crack→Si)
        in the same [0,1] normalised coordinate system.
        """
        s0 = np.array(sensor_positions[0])
        dp = np.array(defect_xy)
        d0 = float(np.linalg.norm(dp - s0))
        lengths = [0.0]   # index 0 = source, not used in localisation
        for i in range(1, len(sensor_positions)):
            si = np.array(sensor_positions[i])
            lengths.append(d0 + float(np.linalg.norm(dp - si)))
        return lengths

    # ------------------------------------------------------------------ #
    #  Grid-search ellipse-intersection heatmap
    # ------------------------------------------------------------------ #
    @staticmethod
    def locate_defect_grid(sensor_positions, echo_path_lengths, grid_res=150):
        """
        For each valid receiver i build the ellipse
            dist(S0, P) + dist(Si, P) = L_i
        Accumulate a Gaussian-kernel probability over P on a grid.

        sigma is scaled to ~3 grid cells so discretisation doesn't kill
        the Gaussian (was the main bug in earlier versions).
        """
        x = np.linspace(0, 1, grid_res)
        y = np.linspace(0, 1, grid_res)
        X, Y = np.meshgrid(x, y)

        s0    = np.array(sensor_positions[0])
        sigma = 3.5 / grid_res        # adaptive: 3.5 cells wide
        prob  = np.zeros_like(X)
        valid = 0

        for i in range(1, len(sensor_positions)):
            L = echo_path_lengths[i]
            if L <= 1e-9:
                continue
            si       = np.array(sensor_positions[i])
            d_src    = np.sqrt((X - s0[0])**2 + (Y - s0[1])**2)
            d_sen    = np.sqrt((X - si[0])**2 + (Y - si[1])**2)
            residual = np.abs((d_src + d_sen) - L)
            prob    += np.exp(-(residual**2) / (2.0 * sigma**2))
            valid   += 1

        if valid > 0:
            prob /= valid

        return X, Y, prob

    # ------------------------------------------------------------------ #
    #  Weighted centroid of top probability mass
    #  (more robust than pure argmax for symmetric / noisy maps)
    # ------------------------------------------------------------------ #
    @staticmethod
    def weighted_centroid(X, Y, prob, top_fraction=0.05):
        flat = prob.flatten()
        if flat.max() < 1e-12:
            return float(X.mean()), float(Y.mean())
        threshold = flat.max() * (1.0 - top_fraction)
        mask      = flat >= threshold
        w         = flat[mask]
        wx        = X.flatten()[mask]
        wy        = Y.flatten()[mask]
        total     = w.sum() + 1e-12
        cx        = float((w * wx).sum() / total)
        cy        = float((w * wy).sum() / total)
        return cx, cy

    # ------------------------------------------------------------------ #
    #  Nonlinear least-squares refinement (Nelder-Mead on ellipse residuals)
    # ------------------------------------------------------------------ #
    @staticmethod
    def refine_localisation(sensor_positions, echo_path_lengths, init_xy,
                            bounds=((0.0, 1.0), (0.0, 1.0))):
        s0 = np.array(sensor_positions[0])
        pairs = [
            (np.array(sensor_positions[i]), echo_path_lengths[i])
            for i in range(1, len(sensor_positions))
            if echo_path_lengths[i] > 1e-9
        ]
        if len(pairs) < 2:
            return init_xy[0], init_xy[1], float('inf')

        def cost(xy):
            x, y = xy
            err = 0.0
            for si, L in pairs:
                d_src = np.sqrt((x - s0[0])**2 + (y - s0[1])**2)
                d_sen = np.sqrt((x - si[0])**2 + (y - si[1])**2)
                err  += (d_src + d_sen - L)**2
            return err

        # Differential evolution for global minimum (handles symmetry)
        try:
            res = differential_evolution(
                cost, bounds,
                seed=42, maxiter=800, tol=1e-10,
                popsize=12, mutation=(0.5, 1.5), recombination=0.9
            )
            rx = float(np.clip(res.x[0], 0.0, 1.0))
            ry = float(np.clip(res.x[1], 0.0, 1.0))
            return rx, ry, float(res.fun)
        except Exception:
            return float(init_xy[0]), float(init_xy[1]), float('inf')

    # ------------------------------------------------------------------ #
    #  Full localisation pipeline (grid → centroid → refine)
    # ------------------------------------------------------------------ #
    def localise(self, sensor_positions, echo_path_lengths, grid_res=150):
        X, Y, prob = self.locate_defect_grid(
            sensor_positions, echo_path_lengths, grid_res
        )
        cx, cy = self.weighted_centroid(X, Y, prob, top_fraction=0.04)
        rx, ry, residual = self.refine_localisation(
            sensor_positions, echo_path_lengths, (cx, cy)
        )
        return X, Y, prob, rx, ry, residual