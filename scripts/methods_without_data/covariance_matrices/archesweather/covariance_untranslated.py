import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
import numpy as np
import torch
import glob
import psutil
import matplotlib.pyplot as plt

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
SAVE_DIR = f"{config.COVARIANCE_DIR}/archesweather"
os.makedirs(SAVE_DIR, exist_ok=True)
PLOT_DIR = f"{config.FIGURES_DIR}/covariance_diagnostics/archesweather"  # diagnostic plots
os.makedirs(PLOT_DIR, exist_ok=True)
device = torch.device("cpu")

# All (layer, block) pairs. Layers 0 and 1 have 192 channels, layers 2 to 4 have 384.
# The 192-channel latents are zero-padded to 384 by prepending zeros, so the data
# sits in the last 192 channels (where the layer 0 and 1 translators are
# initialised as identity).
BLOCKS = (
    [(0, 0)] +
    [(1, b) for b in range(4)] +
    [(2, b) for b in range(12)] +
    [(3, b) for b in range(12)] +
    [(4, b) for b in range(4)]
)
PAD_TARGET_DIM = 384

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

# Read the shapes of all blocks from the first file
block_info = {}  # key: (N, D_orig, pad_width)
with np.load(latent_files[0]) as f0:
    for layer, block in BLOCKS:
        key = key_for(layer, block)
        N, D_orig = f0[key].shape[1], f0[key].shape[2]  # (1, N, D)
        pad_width = PAD_TARGET_DIM - D_orig if D_orig < PAD_TARGET_DIM else 0
        block_info[key] = (N, D_orig, pad_width)
        print(f"{tag_for(layer, block)}: key={key}, N={N}, D_orig={D_orig}, pad_width={pad_width}")

# ---------------------------
# Streaming covariance accumulation
# ---------------------------
# Running sums as in ../graphcast/covariance_untranslated.py; one file is read at a
# time.
sum_x = {}
sum_xx = {}
counts = {}
for layer, block in BLOCKS:
    key = key_for(layer, block)
    N, D_orig, pad_width = block_info[key]
    D = D_orig + pad_width
    sum_x[key] = torch.zeros(N, D, dtype=torch.float64, device=device)
    sum_xx[key] = torch.zeros(D, D, dtype=torch.float64, device=device)
    counts[key] = 0

print_ram_usage("Before streaming")

for fi, path in enumerate(latent_files):
    with np.load(path) as f:
        for layer, block in BLOCKS:
            key = key_for(layer, block)
            N, D_orig, pad_width = block_info[key]
            try:
                x_np = f[key][0]  # (N, D_orig)
            except Exception as e:
                print(f"[{os.path.basename(path)}] Skipping {key}: {e}")
                continue
            if pad_width > 0:
                x_np = np.pad(x_np, ((0, 0), (pad_width, 0)), mode='constant')
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
    N, D_orig, pad_width = block_info[key]
    T = counts[key]
    if T < 2:
        print(f"[{tag}] Skipping: only {T} usable timestep(s)")
        return

    mean = sum_x[key] / T                     # [N, D]
    mean_outer_sum = mean.T @ mean             # [D, D] = sum_i mean_i mean_i^T
    avg_cov = (sum_xx[key] - T * mean_outer_sum) / (N * (T - 1))
    avg_cov_np = avg_cov.to(torch.float32).cpu().numpy()

    print(f"[{tag}] Used {T}/{len(latent_files)} timesteps, D={avg_cov_np.shape[0]} "
          f"(padded from {D_orig})" if pad_width > 0 else
          f"[{tag}] Used {T}/{len(latent_files)} timesteps, D={avg_cov_np.shape[0]}")

    # Save
    npz_path = os.path.join(SAVE_DIR, f"{tag}_avg_covariance.npz")
    np.savez_compressed(npz_path, avg_cov=avg_cov_np)
    print(f"[{tag}] Saved {npz_path}")

    # Metrics
    min_val, max_val, mean_val = avg_cov_np.min(), avg_cov_np.max(), avg_cov_np.mean()
    diag_mean = np.mean(np.diag(avg_cov_np))
    print(f"[{tag}] Min: {min_val}")
    print(f"[{tag}] Max: {max_val}")
    print(f"[{tag}] Mean: {mean_val}")
    print(f"[{tag}] Diagonal mean: {diag_mean}")

    with open(os.path.join(SAVE_DIR, f"{tag}_metrics.txt"), "w") as f:
        f.write(f"Min: {min_val}\nMax: {max_val}\nMean: {mean_val}\nDiagonal mean: {diag_mean}\n")

    # Plot variance per dimension
    plt.figure()
    plt.plot(np.diag(avg_cov_np))
    plt.title(f"Temporal Variance per Dimension ({tag})")
    plt.xlabel("Hidden Dimension Index")
    plt.ylabel("Variance")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{tag}_variance_per_dimension.png"))
    plt.close()

    # Eigenvalue spectrum
    eigvals = np.linalg.eigvalsh(avg_cov_np)[::-1]
    plt.figure()
    plt.plot(eigvals)
    plt.title(f"Eigenvalue Spectrum ({tag})")
    plt.xlabel("Principal Component Index")
    plt.ylabel("Eigenvalue (Variance)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{tag}_eigvals.png"))
    plt.close()

    # Heatmap (clipped)
    plt.figure(figsize=(10, 8))
    plt.imshow(np.clip(avg_cov_np, -1, 1), cmap='viridis', interpolation='nearest')
    plt.title(f"Averaged Covariance Matrix ({tag})")
    plt.colorbar(label="Covariance")
    plt.xlabel("Hidden dimension")
    plt.ylabel("Hidden dimension")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{tag}_avg_covariance.png"), dpi=300)
    plt.close()

    print(f"[{tag}] Plots saved.")


if __name__ == "__main__":
    for layer, block in BLOCKS:
        try:
            finalize_and_plot(layer, block)
        except Exception as e:
            print(f"[layer{layer}_block{block}] Error: {e}")

    print_ram_usage("Done")