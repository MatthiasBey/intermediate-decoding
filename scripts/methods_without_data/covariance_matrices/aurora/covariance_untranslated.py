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
data_dir = f"{config.LATENT_DIR}/aurora/"
# Encoder output (input of the first backbone block), stored in separate files
# with the same basename as the block latents
ENCODER_DATA_DIR = f"{config.LATENT_DIR}/aurora/encoder_output"
ENCODER_KEY = "encoder_output"

SAVE_DIR = f"{config.COVARIANCE_DIR}/aurora"
os.makedirs(SAVE_DIR, exist_ok=True)
PLOT_DIR = f"{config.FIGURES_DIR}/covariance_diagnostics/aurora"  # diagnostic plots
os.makedirs(PLOT_DIR, exist_ok=True)
device = torch.device("cpu")

# All (part, stage, block) triples of the 3 encoder and 3 decoder stages, plus the
# encoder output. Depending on the stage the latents have 512, 1024 or 2048 channels;
# the covariance matrices are zero-padded to 2048 (zeros appended) to make them
# comparable. Aurora's patch merging and splitting layers are dense linear maps, so
# the position of the padding does not matter.
STAGE_BLOCK_COUNTS = {
    ("enc", 0): 6,
    ("enc", 1): 10,
    ("enc", 2): 8,
    ("dec", 0): 8,
    ("dec", 1): 10,
    ("dec", 2): 6,
}
BLOCKS = [(part, stage, b) for (part, stage), n in STAGE_BLOCK_COUNTS.items() for b in range(n)]
ALL_KEYS = [ENCODER_KEY] + BLOCKS
PAD_TARGET_DIM = 2048  # largest channel number (enc stage 2, dec stage 0)

def tag_for(key):
    if key == ENCODER_KEY:
        return ENCODER_KEY
    part, stage, block = key
    return f"{part}_stage{stage}_block_{block}"

# ---------------------------
# Data discovery
# ---------------------------
def list_latent_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "all_latents_aurora-ft_*.npz")))
    if not files:
        raise FileNotFoundError(f"No latent files found in {data_dir}")
    return files

latent_files = list_latent_files(data_dir)
print(f"Found {len(latent_files)} timestep files in {data_dir}")

# ---------------------------
# Map (part, stage, block) to the index i of x_i and read the shapes
# ---------------------------
# The latent files store x_0 ... x_47 together with metadata arrays part, stage and
# block. The mapping is read from the first file and assumed to hold for all files.
# The encoder output is stored under the key "encoder_x" in ENCODER_DATA_DIR.
block_info = {}  # key: (record_index_or_None, N, D_orig, pad_width)
with np.load(latent_files[0], allow_pickle=False) as f0:
    parts = f0["part"]
    stages = f0["stage"]
    blocks = f0["block"]
    record_index = {}
    for i in range(len(parts)):
        key = (str(parts[i]), int(stages[i]), int(blocks[i]))
        record_index[key] = i

    for key in BLOCKS:
        i = record_index[key]
        N, D_orig = f0[f"x_{i}"].shape[1], f0[f"x_{i}"].shape[2]  # (1, N, D)
        pad_width = PAD_TARGET_DIM - D_orig if D_orig < PAD_TARGET_DIM else 0
        block_info[key] = (i, N, D_orig, pad_width)
        print(f"{tag_for(key)}: i={i}, N={N}, D_orig={D_orig}, pad_width={pad_width}")

encoder_file0 = os.path.join(ENCODER_DATA_DIR, os.path.basename(latent_files[0]))
with np.load(encoder_file0, allow_pickle=False) as fenc0:
    N_enc, D_enc = fenc0["encoder_x"].shape[1], fenc0["encoder_x"].shape[2]
    pad_width_enc = PAD_TARGET_DIM - D_enc if D_enc < PAD_TARGET_DIM else 0
    block_info[ENCODER_KEY] = (None, N_enc, D_enc, pad_width_enc)
    print(f"{ENCODER_KEY}: N={N_enc}, D_orig={D_enc}, pad_width={pad_width_enc}")

# ---------------------------
# Streaming covariance accumulation
# ---------------------------
# Running sums as in ../graphcast/covariance_untranslated.py; one file is read at a
# time. Sums are accumulated at each block's own dimension and only the final
# covariance matrix is zero-padded, which gives the same result with far less memory.
sum_x = {}
sum_xx = {}
counts = {}
for key in ALL_KEYS:
    _, N, D_orig, pad_width = block_info[key]
    sum_x[key] = torch.zeros(N, D_orig, dtype=torch.float64, device=device)
    sum_xx[key] = torch.zeros(D_orig, D_orig, dtype=torch.float64, device=device)
    counts[key] = 0

print_ram_usage("Before streaming")

for fi, path in enumerate(latent_files):
    with np.load(path) as f:
        for key in BLOCKS:
            i, N, D_orig, pad_width = block_info[key]
            try:
                x_np = f[f"x_{i}"][0]  # (N, D_orig), float16
            except Exception as e:
                print(f"[{os.path.basename(path)}] Skipping {tag_for(key)}: {e}")
                continue
            x = torch.from_numpy(x_np.astype(np.float32)).to(dtype=torch.float64, device=device)
            sum_x[key] += x
            sum_xx[key] += x.T @ x
            counts[key] += 1

    # encoder output from its separate file
    encoder_path = os.path.join(ENCODER_DATA_DIR, os.path.basename(path))
    try:
        with np.load(encoder_path, allow_pickle=False) as fenc:
            x_np = fenc["encoder_x"][0]
        x = torch.from_numpy(x_np.astype(np.float32)).to(dtype=torch.float64, device=device)
        sum_x[ENCODER_KEY] += x
        sum_xx[ENCODER_KEY] += x.T @ x
        counts[ENCODER_KEY] += 1
    except Exception as e:
        print(f"[{os.path.basename(path)}] Skipping {ENCODER_KEY}: {e}")

    if fi % 100 == 0 or fi == len(latent_files) - 1:
        print(f"Processed {fi + 1}/{len(latent_files)}: {os.path.basename(path)}")
        print_ram_usage(f"After file {fi + 1}")

print_ram_usage("After streaming all files")

# ---------------------------
# Finalize, save, and plot per block
# ---------------------------
def finalize_and_plot(key):
    tag = tag_for(key)
    _, N, D_orig, pad_width = block_info[key]
    T = counts[key]
    if T < 2:
        print(f"[{tag}] Skipping: only {T} usable timestep(s)")
        return

    mean = sum_x[key] / T                     # [N, D_orig]
    mean_outer_sum = mean.T @ mean             # [D_orig, D_orig] = sum_i mean_i mean_i^T
    avg_cov = (sum_xx[key] - T * mean_outer_sum) / (N * (T - 1))
    avg_cov_np = avg_cov.to(torch.float32).cpu().numpy()

    if pad_width > 0:
        avg_cov_np = np.pad(avg_cov_np, ((0, pad_width), (0, pad_width)), mode='constant')

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
    for key in ALL_KEYS:
        try:
            finalize_and_plot(key)
        except Exception as e:
            print(f"[{tag_for(key)}] Error: {e}")

    print_ram_usage("Done")