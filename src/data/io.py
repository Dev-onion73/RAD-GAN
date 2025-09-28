# src/data/io.py
import numpy as np
import os, json

def save_dataset_npy(path, specs: np.ndarray, metas=None):
    """
    specs: np.ndarray (N, F, T) float32
    metas: list of dicts (len N) optional metadata
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, specs)
    if metas is not None:
        with open(path + '.meta.json', 'w') as f:
            json.dump(metas, f)

def load_dataset_npy(path):
    specs = np.load(path)
    meta_path = path + '.meta.json'
    metas = None
    if os.path.exists(meta_path):
        import json
        metas = json.load(open(meta_path))
    return specs, metas
