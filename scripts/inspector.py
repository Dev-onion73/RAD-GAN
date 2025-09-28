import numpy as np
from src.data.io import load_dataset_npy
from src.viz.viz import show_spectrogram
import matplotlib.pyplot as plt

specs, metas = load_dataset_npy('data/specs_32.npy')
show_spectrogram(specs[0])
plt.show()
