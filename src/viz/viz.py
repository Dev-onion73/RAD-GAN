# src/viz/viz.py
import matplotlib.pyplot as plt
import numpy as np

def show_spectrogram(spec, ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots()
    spec = np.squeeze(spec)
    ax.imshow(spec, aspect='auto', origin='lower')
    ax.set_xlabel('Time bins'); ax.set_ylabel('Freq bins')
    if title: ax.set_title(title)

def compare_grid(reals, fakes, n=4):
    import matplotlib.pyplot as plt
    n = min(n, len(reals), len(fakes))
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    for i in range(n):
        show_spectrogram(reals[i], ax=axes[0,i], title=f"Real {i}")
        show_spectrogram(fakes[i], ax=axes[1,i], title=f"Fake {i}")
    plt.tight_layout()
    return fig
