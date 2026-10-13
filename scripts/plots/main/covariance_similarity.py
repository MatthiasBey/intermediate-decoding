import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %%
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re

# =============================================================================
# Figure style (all font and tick sizes are set here)
# =============================================================================
FONT_FAMILY = "sans-serif"             # e.g. "sans-serif", "serif", "monospace"
FONT_NAME = ["DejaVu Sans"]            # preferred fonts in order, e.g. ["Arial", "Helvetica", "DejaVu Sans"]

FONT_SIZE_AXIS_LABEL = 24              # colorbar label
FONT_SIZE_TICK_LABEL = 22              # heatmap tick numbers and colorbar tick numbers
FONT_SIZE_TITLE = 22                  # panel titles
FONT_SIZE_SUPTITLE = 18                # figure title
FONT_SIZE_SHARED_LABEL = 24            # the "Latent state" labels along the figure edges

TICK_LENGTH = 3.5                      # tick mark length (points)
TICK_WIDTH = 1.5                       # tick mark width (points)

FIGSIZE = (20, 12.5)                   # same width as the relative MSE figure
CBAR_LABEL = "Cosine similarity"
DPI = 300

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    f"font.{FONT_FAMILY}": FONT_NAME,
    "font.size": FONT_SIZE_TICK_LABEL,
    "axes.labelsize": FONT_SIZE_AXIS_LABEL,
    "axes.titlesize": FONT_SIZE_TITLE,
    "xtick.labelsize": FONT_SIZE_TICK_LABEL,
    "ytick.labelsize": FONT_SIZE_TICK_LABEL,
    "figure.titlesize": FONT_SIZE_SUPTITLE,
    "xtick.major.size": TICK_LENGTH,
    "ytick.major.size": TICK_LENGTH,
    "xtick.major.width": TICK_WIDTH,
    "ytick.major.width": TICK_WIDTH,
})

# %% Paths
# Time-averaged covariance matrices of the latent states, from
# methods_without_data/covariance_matrices/<model>/covariance_untranslated.py (raw) and
# covariance_translated.py (after the translator, filename prefix "withtranslator_").
arches_dir = f"{config.COVARIANCE_DIR}/archesweather"
gc_dir = f"{config.COVARIANCE_DIR}/graphcast"
aurora_dir = f"{config.COVARIANCE_DIR}/aurora"

# %% Functions
def frobenius_cosine_similarity(A, B):
    return np.sum(A * B) / (np.linalg.norm(A, "fro") * np.linalg.norm(B, "fro"))

def remove_top_pcs(cov, k):
    # project out the k leading eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    top_vecs = eigvecs[:, idx[:k]]
    proj = top_vecs @ top_vecs.T
    I = np.eye(cov.shape[0])
    return (I - proj) @ cov @ (I - proj)

def compute_similarity_matrix(cov_matrices, remove_top_k=0):
    # pairwise cosine similarity, optionally after removing the top-k PCs of each matrix
    if remove_top_k > 0:
        cov_matrices = [remove_top_pcs(cov, remove_top_k) for cov in cov_matrices]
    n = len(cov_matrices)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = frobenius_cosine_similarity(cov_matrices[i], cov_matrices[j])
    return sim

def load_arches_cov(path, translated=False):
    # raw files start with "layer", so the raw pattern excludes "withtranslator_" files
    pattern = "withtranslator_layer*_block*_avg_covariance.npz" if translated else "layer*_block*_avg_covariance.npz"
    files = sorted(
        glob.glob(os.path.join(path, pattern)),
        key=lambda f: tuple(map(int, re.findall(r"layer(\d+)_block(\d+)", f)[0]))
    )
    covs = [np.load(f)["avg_cov"] for f in files]
    return np.array(covs)

def load_gc_cov(path, translated=False):
    # raw files start with the step number ("00_...", "01_..."); the raw pattern also
    # matches "withtranslator_" files, which are filtered out below
    pattern = "withtranslator_*_avg_covariance_step*.npz" if translated else "*_avg_covariance_step*.npz"
    files = glob.glob(os.path.join(path, pattern))
    if not translated:
        files = [f for f in files if not os.path.basename(f).startswith("withtranslator_")]
    files = sorted(files, key=lambda f: int(re.search(r"step(\d+)\.npz$", f).group(1)))
    covs = [np.load(f)["avg_cov"] for f in files]
    return np.array(covs)

def load_aurora_cov(path):
    # encoder output, then all blocks in forward-pass order; matrices are zero-padded
    # to (2048, 2048) by covariance_untranslated.py
    stage_block_counts = [
        ("enc", 0, 6), ("enc", 1, 10), ("enc", 2, 8),
        ("dec", 0, 8), ("dec", 1, 10), ("dec", 2, 6),
    ]
    tags = ["encoder_output"]
    for part, stage, n in stage_block_counts:
        tags += [f"{part}_stage{stage}_block_{b}" for b in range(n)]
    covs = [np.load(os.path.join(path, f"{tag}_avg_covariance.npz"))["avg_cov"] for tag in tags]
    return np.array(covs)

def load_aurora_cov_translated(path):
    # same order as load_aurora_cov; missing files are skipped and reported.
    # All translated matrices have the common shape (1024, 1024).
    stage_block_counts = [
        ("enc", 0, 6), ("enc", 1, 10), ("enc", 2, 8),
        ("dec", 0, 8), ("dec", 1, 10), ("dec", 2, 6),
    ]
    tags = ["encoder_output"]
    for part, stage, n in stage_block_counts:
        tags += [f"{part}_stage{stage}_block_{b}" for b in range(n)]

    covs, missing = [], []
    for tag in tags:
        fpath = os.path.join(path, f"withtranslator_{tag}_avg_covariance.npz")
        if os.path.exists(fpath):
            covs.append(np.load(fpath)["avg_cov"])
        else:
            missing.append(tag)

    if missing:
        print(f"Aurora translated: {len(missing)}/{len(tags)} matrices missing: {missing}")

    return np.array(covs)

# %% Load data
arches_raw = load_arches_cov(arches_dir, translated=False)
arches_trans = load_arches_cov(arches_dir, translated=True)
gc_raw = load_gc_cov(gc_dir, translated=False)
gc_trans = load_gc_cov(gc_dir, translated=True)
aurora_raw = load_aurora_cov(aurora_dir)
aurora_trans = load_aurora_cov_translated(aurora_dir)

print(f"Arches raw: {arches_raw.shape[0]} steps")
print(f"Arches with translators: {arches_trans.shape[0]} steps")
print(f"GC raw: {gc_raw.shape[0]} steps")
print(f"GC with translators: {gc_trans.shape[0]} steps")
print(f"Aurora raw: {aurora_raw.shape[0]} steps")
print(f"Aurora with translators: {aurora_trans.shape[0]} steps")



# %% Figure
# Columns: GraphCast, Aurora, ArchesWeather; rows: raw, translated.
# remove_top_k > 0 removes the k leading principal components of every matrix first.

style_raw = {
    "cmap": "viridis",
    "figsize": (12, 10),
    "arches_vmin": 0.0, "arches_vmax": 1.0,
    "gc_vmin": 0.5, "gc_vmax": 1.0,
    "cbar_label": CBAR_LABEL,
    "cbar_fraction": 0.05,   # relative width of colorbar
    "cbar_pad": 0.02,         # distance from subplot
    "label_offset_x": 0.06,   # position of global ylabel
    "label_offset_y": 0.04,   # position of global xlabel
}

style_trans = {
    "cmap": "viridis",
    "figsize": (14, 6),
    "arches_vmin": 0.5, "arches_vmax": 1.0,
    "gc_vmin": 0.95, "gc_vmax": 1.0,
    "cbar_label": CBAR_LABEL,
    "cbar_fraction": 0.05,
    "cbar_pad": 0.02,
    "label_offset_x": 0.04,
    "label_offset_y": 0.04,
}

style_aurora = {
    "cmap": "viridis",
    "raw_vmin": 0.0, "raw_vmax": 1.0,
    # translated matrices are all mapped towards the same final state, so their
    # similarities are much higher than the raw ones
    "trans_vmin": 0.95, "trans_vmax": 1.0,
    "cbar_label": CBAR_LABEL,
    "cbar_fraction": 0.05,
    "cbar_pad": 0.02,
}


def make_covariance_figure(remove_top_k, save_path):
    """Build and save the 2x3 (raw/translated x GC/Aurora/Arches) similarity-heatmap
    figure for one remove_top_k value."""
    arches_raw_sim = compute_similarity_matrix(arches_raw, remove_top_k)
    arches_trans_sim = compute_similarity_matrix(arches_trans, remove_top_k)
    gc_raw_sim = compute_similarity_matrix(gc_raw, remove_top_k)
    gc_trans_sim = compute_similarity_matrix(gc_trans, remove_top_k)
    aurora_raw_sim = compute_similarity_matrix(aurora_raw, remove_top_k)
    aurora_trans_sim = compute_similarity_matrix(aurora_trans, remove_top_k)

    # the fixed color ranges are for remove_top_k=0; otherwise each panel is auto-scaled
    use_fixed_range = (remove_top_k == 0)

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE)

    # =======================
    # Top row: RAW
    # =======================

    # --- GraphCast RAW (top-left)
    im0 = sns.heatmap(gc_raw_sim, ax=axes[0, 0], cmap=style_raw["cmap"],
                      vmin=style_raw["gc_vmin"] if use_fixed_range else None,
                      vmax=style_raw["gc_vmax"] if use_fixed_range else None,
                      cbar=True, square=True,
                      cbar_kws={"fraction": style_raw["cbar_fraction"],
                                "pad": style_raw["cbar_pad"],
                                "label": style_raw["cbar_label"]},
                      xticklabels=range(gc_raw_sim.shape[0]),
                      yticklabels=range(gc_raw_sim.shape[0]))
    axes[0, 0].set_title("GraphCast (raw)")

    # --- Aurora RAW (top-middle)
    im4 = sns.heatmap(aurora_raw_sim, ax=axes[0, 1], cmap=style_aurora["cmap"],
                      vmin=style_aurora["raw_vmin"] if use_fixed_range else None,
                      vmax=style_aurora["raw_vmax"] if use_fixed_range else None,
                      cbar=True, square=True,
                      cbar_kws={"fraction": style_aurora["cbar_fraction"],
                                "pad": style_aurora["cbar_pad"],
                                "label": style_aurora["cbar_label"]},
                      xticklabels=range(aurora_raw_sim.shape[0]),
                      yticklabels=range(aurora_raw_sim.shape[0]))
    axes[0, 1].set_title("Aurora (raw)")

    # --- Arches RAW (top-right)
    im1 = sns.heatmap(arches_raw_sim, ax=axes[0, 2], cmap=style_raw["cmap"],
                      vmin=style_raw["arches_vmin"] if use_fixed_range else None,
                      vmax=style_raw["arches_vmax"] if use_fixed_range else None,
                      cbar=True, square=True,
                      cbar_kws={"fraction": style_raw["cbar_fraction"],
                                "pad": style_raw["cbar_pad"],
                                "label": style_raw["cbar_label"]},
                      xticklabels=range(arches_raw_sim.shape[0]),
                      yticklabels=range(arches_raw_sim.shape[0]))
    axes[0, 2].set_title("ArchesWeather (raw)")

    # =======================
    # Bottom row: TRANSLATED
    # =======================

    # --- GraphCast TRANSLATED (bottom-left)
    im2 = sns.heatmap(gc_trans_sim, ax=axes[1, 0], cmap=style_trans["cmap"],
                      vmin=style_trans["gc_vmin"] if use_fixed_range else None,
                      vmax=style_trans["gc_vmax"] if use_fixed_range else None,
                      cbar=True, square=True,
                      cbar_kws={"fraction": style_trans["cbar_fraction"],
                                "pad": style_trans["cbar_pad"],
                                "label": style_trans["cbar_label"]},
                      xticklabels=range(gc_trans_sim.shape[0]),
                      yticklabels=range(gc_trans_sim.shape[0]))
    axes[1, 0].set_title("GraphCast (translated)")

    # --- Aurora TRANSLATED (bottom-middle)
    im5 = sns.heatmap(aurora_trans_sim, ax=axes[1, 1], cmap=style_aurora["cmap"],
                      vmin=style_aurora["trans_vmin"] if use_fixed_range else None,
                      vmax=style_aurora["trans_vmax"] if use_fixed_range else None,
                      cbar=True, square=True,
                      cbar_kws={"fraction": style_aurora["cbar_fraction"],
                                "pad": style_aurora["cbar_pad"],
                                "label": style_aurora["cbar_label"]},
                      xticklabels=range(aurora_trans_sim.shape[0]),
                      yticklabels=range(aurora_trans_sim.shape[0]))
    axes[1, 1].set_title("Aurora (translated)")

    # --- Arches TRANSLATED (bottom-right)
    im3 = sns.heatmap(arches_trans_sim, ax=axes[1, 2], cmap=style_trans["cmap"],
                      vmin=style_trans["arches_vmin"] if use_fixed_range else None,
                      vmax=style_trans["arches_vmax"] if use_fixed_range else None,
                      cbar=True, square=True,
                      cbar_kws={"fraction": style_trans["cbar_fraction"],
                                "pad": style_trans["cbar_pad"],
                                "label": style_trans["cbar_label"]},
                      xticklabels=range(arches_trans_sim.shape[0]),
                      yticklabels=range(arches_trans_sim.shape[0]))
    axes[1, 2].set_title("ArchesWeather (translated)")

    # =======================
    # Formatting
    # =======================

    for ax in axes[0:2, 0].flat:
        ax.tick_params(axis="both", rotation=0)
        ax.set_xticks(ax.get_xticks()[::2])
        ax.set_yticks(ax.get_yticks()[::2])
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes[0:2, 1].flat:
        ax.tick_params(axis="both", rotation=0)
        ax.set_xticks(ax.get_xticks()[::10])
        ax.set_yticks(ax.get_yticks()[::10])
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes[0:2, 2].flat:
        ax.tick_params(axis="both", rotation=0)
        ax.set_xticks(ax.get_xticks()[::5])
        ax.set_yticks(ax.get_yticks()[::5])
        ax.set_xlabel("")
        ax.set_ylabel("")

    # global labels
    fig.text(0.5, 0.07, "Latent state", ha="center", fontsize=FONT_SIZE_SHARED_LABEL)
    fig.text(0.035, 0.5, "Latent state", va="center", rotation="vertical", fontsize=FONT_SIZE_SHARED_LABEL)

    plt.tight_layout(rect=[0.04, 0.05, 1, 0.97], h_pad=-0.5)
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    print(f"Saved (remove_top_k={remove_top_k}): {save_path}")


make_covariance_figure(0, f"{config.FIGURES_DIR}/covariance_similarity.png")  # remove_top_k=0: no PCs removed
