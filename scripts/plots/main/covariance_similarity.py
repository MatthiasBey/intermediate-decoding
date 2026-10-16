"""Cosine similarity between the covariance matrices of the latent states of each model, raw (top
row) and after the translator (bottom row). Columns: GraphCast, Aurora, ArchesWeather.

Input: the time-averaged covariance matrices of methods_without_data/covariance_matrices/<model>/
covariance_untranslated.py and covariance_translated.py (file prefix "withtranslator_").
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config

# font and tick sizes as in relative_mse_all_variables.py
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 22,
    "axes.labelsize": 24,
    "axes.titlesize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "figure.titlesize": 18,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
})

AURORA_STAGES = [("enc", 0, 6), ("enc", 1, 10), ("enc", 2, 8), ("dec", 0, 8), ("dec", 1, 10), ("dec", 2, 6)]


def frobenius_cosine_similarity(A, B):
    return np.sum(A * B) / (np.linalg.norm(A, "fro") * np.linalg.norm(B, "fro"))


def similarity_matrix(cov_matrices):
    """Pairwise cosine similarity of all covariance matrices."""
    n = len(cov_matrices)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = frobenius_cosine_similarity(cov_matrices[i], cov_matrices[j])
    return sim


def load_graphcast(translated):
    # raw files start with the step number ("00_...", "01_...")
    folder = f"{config.COVARIANCE_DIR}/graphcast"
    files = glob.glob(os.path.join(folder, "withtranslator_*_avg_covariance_step*.npz" if translated
                                   else "*_avg_covariance_step*.npz"))
    if not translated:
        files = [f for f in files if not os.path.basename(f).startswith("withtranslator_")]
    files = sorted(files, key=lambda f: int(re.search(r"step(\d+)\.npz$", f).group(1)))
    return np.array([np.load(f)["avg_cov"] for f in files])


def load_aurora(translated):
    """Encoder output, then all blocks in network order. The raw matrices are zero-padded to
    (2048, 2048) by covariance_untranslated.py, the translated ones are (1024, 1024)."""
    folder = f"{config.COVARIANCE_DIR}/aurora"
    tags = ["encoder_output"] + [f"{part}_stage{stage}_block_{b}" for part, stage, n in AURORA_STAGES for b in range(n)]
    prefix = "withtranslator_" if translated else ""
    return np.array([np.load(os.path.join(folder, f"{prefix}{tag}_avg_covariance.npz"))["avg_cov"] for tag in tags])


def load_arches(translated):
    # raw files start with "layer", so the raw pattern does not match the "withtranslator_" files
    folder = f"{config.COVARIANCE_DIR}/archesweather"
    pattern = "withtranslator_layer*_block*_avg_covariance.npz" if translated else "layer*_block*_avg_covariance.npz"
    files = sorted(glob.glob(os.path.join(folder, pattern)),
                   key=lambda f: tuple(map(int, re.findall(r"layer(\d+)_block(\d+)", f)[0])))
    return np.array([np.load(f)["avg_cov"] for f in files])


# (title, loader, every n-th tick label, colour range raw, colour range translated); the
# translated matrices all map towards the same final state, so their similarities are higher
models = [
    ("GraphCast", load_graphcast, 2, (0.5, 1.0), (0.95, 1.0)),
    ("Aurora", load_aurora, 10, (0.0, 1.0), (0.95, 1.0)),
    ("ArchesWeather", load_arches, 5, (0.0, 1.0), (0.5, 1.0)),
]

fig, axes = plt.subplots(2, 3, figsize=(20, 12.5))  # same width as the relative MSE figure
for row, translated in enumerate([False, True]):
    for col, (name, load, tick_step, raw_range, translated_range) in enumerate(models):
        covariances = load(translated)
        print(f"{name} {'with translators' if translated else 'raw'}: {covariances.shape[0]} states")
        sim = similarity_matrix(covariances)
        vmin, vmax = translated_range if translated else raw_range
        ax = axes[row, col]
        sns.heatmap(sim, ax=ax, cmap="viridis", vmin=vmin, vmax=vmax, cbar=True, square=True,
                    cbar_kws={"fraction": 0.05, "pad": 0.02, "label": "Cosine similarity"},
                    xticklabels=range(sim.shape[0]), yticklabels=range(sim.shape[0]))
        ax.set_title(f"{name} ({'translated' if translated else 'raw'})")

for col, (_, _, tick_step, _, _) in enumerate(models):
    for ax in axes[:, col]:
        ax.tick_params(axis="both", rotation=0)
        ax.set_xticks(ax.get_xticks()[::tick_step])
        ax.set_yticks(ax.get_yticks()[::tick_step])
        ax.set_xlabel("")
        ax.set_ylabel("")

fig.text(0.5, 0.07, "Latent state", ha="center", fontsize=24)
fig.text(0.035, 0.5, "Latent state", va="center", rotation="vertical", fontsize=24)

plt.tight_layout(rect=[0.04, 0.05, 1, 0.97], h_pad=-0.5)
save_path = f"{config.FIGURES_DIR}/covariance_similarity.png"
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Saved {save_path}")
