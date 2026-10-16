import os
import sys
import numpy as np
import torch
import glob
import psutil
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # scripts/
import config

# ---------------------------
# RAM Usage Utility
# ---------------------------
def print_ram_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    print(f"[{tag}] RAM usage: {mem:.2f} MB")

# ---------------------------
# Config
# ---------------------------
data_dir = f"{config.LATENT_DIR}/archesweather"
LEGACY_TRANSLATOR_DIR = f"{config.TRANSLATOR_DIR}/archesweather/skip/weights/"
EXPAND_TRANSLATOR_DIR = f"{config.TRANSLATOR_DIR}/archesweather/no_skip/weights/"
# Output file names get the prefix "withtranslator_" to distinguish them from
# the output of covariance_untranslated.py.
SAVE_DIR = f"{config.COVARIANCE_DIR}/archesweather"
os.makedirs(SAVE_DIR, exist_ok=True)
PLOT_DIR = f"{config.FIGURES_DIR}/covariance_diagnostics/archesweather"  # diagnostic plots
os.makedirs(PLOT_DIR, exist_ok=True)
device = torch.device("cpu")

# Layers 1 and 4: per-token affine translators (skip/weights). Layers 2 and 3:
# ExpandTranslator (no_skip/weights, see
# translator_training/archesweather/train_no_skip_translators_layer2_and_3.py), which
# also upsamples from the coarse (14400 tokens) to the fine grid (57600 tokens).
# layer4.blocks.3 is the target space itself and is used without translator.
BLOCKS = (
    [(1, b) for b in range(4)] +
    [(2, b) for b in range(12)] +
    [(3, b) for b in range(12)] +
    [(4, b) for b in range(4)]
)

# Coarse grid of layers 2 and 3; ExpandTranslator upsamples it by 2 in lat and lon
PL, COARSE_LAT, COARSE_LON = 8, 30, 60
FINE_TOKENS = PL * COARSE_LAT * 2 * COARSE_LON * 2  # 57600
D = 384  # channels after translation

def key_for(layer, block):
    if layer == 0:
        return f"layer{layer}_block{block}"
    return f"layer{layer}.blocks.{block}"

def tag_for(layer, block):
    return f"layer{layer}_block{block}"

# ---------------------------
# Data discovery
# ---------------------------
def list_latent_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "all_latents_*.npz")))
    if not files:
        raise FileNotFoundError(f"No latent files found in {data_dir}")
    return files

latent_files = list_latent_files(data_dir)
print(f"Found {len(latent_files)} timestep files in {data_dir}")

# ---------------------------
# Load translators
# ---------------------------
translators = {}

for block in range(4):
    npz = np.load(os.path.join(LEGACY_TRANSLATOR_DIR, f"Arches_translator_matrix_layer1_block{block}.npz"))
    translators[(1, block)] = {"kind": "affine", "W": npz["W"].astype(np.float32), "b": npz["b"].astype(np.float32)}

for block in range(3):
    npz = np.load(os.path.join(LEGACY_TRANSLATOR_DIR, f"Arches_translator_matrix_layer4_block{block}.npz"))
    translators[(4, block)] = {"kind": "affine", "W": npz["W"].astype(np.float32), "b": npz["b"].astype(np.float32)}
translators[(4, 3)] = {"kind": "raw"}

for layer in (2, 3):
    for block in range(12):
        npz = np.load(os.path.join(
            EXPAND_TRANSLATOR_DIR, f"Arches_translator_rep_stab_matrix_layer{layer}_block{block}.npz"
        ))
        translators[(layer, block)] = {"kind": "expand", "W": npz["W"].astype(np.float32), "b": npz["b"].astype(np.float32)}

print(f"Loaded {len(translators)} translators")

# ---------------------------
# Translator application
# ---------------------------
def apply_translator(entry, x_np):
    """x_np: (N, D_in) raw latent for one file/block. Returns (FINE_TOKENS, 384)."""
    kind = entry["kind"]
    if kind == "raw":
        return x_np
    if kind == "affine":
        return x_np @ entry["W"].T + entry["b"]
    if kind == "expand":
        x = x_np @ entry["W"].T + entry["b"]  # (14400, 1536)
        x = x.reshape(PL, COARSE_LAT, COARSE_LON, 2, 2, 384)
        # Same as permute(0, 1, 2, 4, 3, 5, 6) in ExpandTranslator.forward(), without
        # the batch axis
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(FINE_TOKENS, 384)
        return x
    raise ValueError(f"Unknown translator kind: {kind}")

# ---------------------------
# Streaming covariance accumulation
# ---------------------------
# Running sums as in ../graphcast/covariance_untranslated.py. All translated latents
# have shape (FINE_TOKENS, 384), so no padding is needed.
sum_x = {}
sum_xx = {}
counts = {}
for layer, block in BLOCKS:
    key = key_for(layer, block)
    sum_x[key] = torch.zeros(FINE_TOKENS, D, dtype=torch.float64, device=device)
    sum_xx[key] = torch.zeros(D, D, dtype=torch.float64, device=device)
    counts[key] = 0

print_ram_usage("Before streaming")

for fi, path in enumerate(latent_files):
    with np.load(path) as f:
        for layer, block in BLOCKS:
            key = key_for(layer, block)
            try:
                x_np = f[key][0].astype(np.float32, copy=False)  # (N, D_in)
            except Exception as e:
                print(f"[{os.path.basename(path)}] Skipping {key}: {e}")
                continue
            x_np = apply_translator(translators[(layer, block)], x_np)
            x = torch.from_numpy(x_np).to(dtype=torch.float64, device=device)
            sum_x[key] += x
            sum_xx[key] += x.T @ x
            counts[key] += 1

    if fi % 100 == 0 or fi == len(latent_files) - 1:
        print(f"Processed {fi + 1}/{len(latent_files)}: {os.path.basename(path)}")
        print_ram_usage(f"After file {fi + 1}")

print_ram_usage("After streaming all files")

# ---------------------------
# Finalize, save, and plot per block
# ---------------------------
def finalize_and_plot(layer, block):
    key = key_for(layer, block)
    tag = tag_for(layer, block)
    T = counts[key]
    if T < 2:
        print(f"[{tag}] Skipping: only {T} usable timestep(s)")
        return

    mean = sum_x[key] / T                     # [N, D]
    mean_outer_sum = mean.T @ mean             # [D, D] = sum_i mean_i mean_i^T
    avg_cov = (sum_xx[key] - T * mean_outer_sum) / (FINE_TOKENS * (T - 1))
    avg_cov_np = avg_cov.to(torch.float32).cpu().numpy()

    print(f"[{tag}] Used {T}/{len(latent_files)} timesteps, D={avg_cov_np.shape[0]}")

    # Save
    ptag = f"withtranslator_{tag}"

    npz_path = os.path.join(SAVE_DIR, f"{ptag}_avg_covariance.npz")
    np.savez_compressed(npz_path, avg_cov=avg_cov_np)
    print(f"[{tag}] Saved {npz_path}")

    # Metrics
    min_val, max_val, mean_val = avg_cov_np.min(), avg_cov_np.max(), avg_cov_np.mean()
    diag_mean = np.mean(np.diag(avg_cov_np))
    print(f"[{tag}] Min: {min_val}")
    print(f"[{tag}] Max: {max_val}")
    print(f"[{tag}] Mean: {mean_val}")
    print(f"[{tag}] Diagonal mean: {diag_mean}")

    with open(os.path.join(SAVE_DIR, f"{ptag}_metrics.txt"), "w") as f:
        f.write(f"Min: {min_val}\nMax: {max_val}\nMean: {mean_val}\nDiagonal mean: {diag_mean}\n")

    # Plot variance per dimension
    plt.figure()
    plt.plot(np.diag(avg_cov_np))
    plt.title(f"withtranslator Temporal Variance per Dimension ({tag})")
    plt.xlabel("Hidden Dimension Index")
    plt.ylabel("Variance")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{ptag}_variance_per_dimension.png"))
    plt.close()

    # Eigenvalue spectrum
    eigvals = np.linalg.eigvalsh(avg_cov_np)[::-1]
    plt.figure()
    plt.plot(eigvals)
    plt.title(f"withtranslator Eigenvalue Spectrum ({tag})")
    plt.xlabel("Principal Component Index")
    plt.ylabel("Eigenvalue (Variance)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{ptag}_eigvals.png"))
    plt.close()

    # Heatmap (clipped)
    plt.figure(figsize=(10, 8))
    plt.imshow(np.clip(avg_cov_np, -1, 1), cmap='viridis', interpolation='nearest')
    plt.title(f"withtranslator Averaged Covariance Matrix ({tag})")
    plt.colorbar(label="Covariance")
    plt.xlabel("Hidden dimension")
    plt.ylabel("Hidden dimension")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{ptag}_avg_covariance.png"), dpi=300)
    plt.close()

    print(f"[{tag}] Plots saved.")


if __name__ == "__main__":
    for layer, block in BLOCKS:
        try:
            finalize_and_plot(layer, block)
        except Exception as e:
            print(f"[layer{layer}_block{block}] Error: {e}")

    print_ram_usage("Done")