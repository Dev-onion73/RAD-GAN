# scripts/gen_dataset.py
import numpy as np
from src.sim.microdoppler import simulate_and_spectrogram
from src.data.io import save_dataset_npy

def gen_and_save(path='data/specs_32.npy', N=32, fs=2000):
    specs = []
    metas = []
    for i in range(N):
        rot = float(np.random.uniform(1.0, 4.0))
        s = simulate_and_spectrogram(duration=1.0, fs=fs, rot_rate_hz=rot, noise_level=0.03)
        specs.append(s)
        metas.append({'rot_rate_hz': rot})
    specs = np.stack(specs)    # (N, F, T)
    save_dataset_npy(path, specs, metas)
    print("Saved:", path, "shape:", specs.shape)

if __name__ == '__main__':
    gen_and_save()
