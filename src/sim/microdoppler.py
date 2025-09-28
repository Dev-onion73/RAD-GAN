# src/sim/microdoppler.py
import numpy as np
from scipy.signal import stft

def simulate_rotating_scatterer(fs=2000, duration=1.0, rot_rate_hz=2.0, v_trans=0.0, noise_level=0.0):
    """
    Simple single-scatterer simulator: radial velocity is sinusoidal (rotation-like).
    All units are normalized (no RF transmitter). This produces a baseband time-series.
    """
    t = np.arange(0, duration, 1.0/fs)
    v_amp = rot_rate_hz * 0.1              # tunable amplitude
    radial_v = v_trans + v_amp * np.sin(2*np.pi*rot_rate_hz*t)
    inst_freq = 2.0 * radial_v             # normalized instantaneous freq (arbitrary units)
    phase = 2*np.pi * np.cumsum(inst_freq) / fs
    signal = np.cos(phase)
    if noise_level > 0:
        signal = signal + np.random.normal(scale=noise_level, size=signal.shape)
    return signal, fs

def spectrogram_from_signal(signal, fs, nperseg=256, noverlap=128):
    f, t, Zxx = stft(signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    S = np.abs(Zxx)
    S = np.log1p(S)                         # log compression
    # normalize to [0,1] per-sample
    S = (S - S.min()) / (S.max() - S.min() + 1e-9)
    return S  # shape (freq_bins, time_bins)

def simulate_and_spectrogram(duration=1.0, fs=2000, rot_rate_hz=2.0, noise_level=0.01,
                             nperseg=256, noverlap=128):
    sig, sr = simulate_rotating_scatterer(fs=fs, duration=duration, rot_rate_hz=rot_rate_hz,
                                          noise_level=noise_level)
    S = spectrogram_from_signal(sig, fs=sr, nperseg=nperseg, noverlap=noverlap)
    return S.astype(np.float32)
