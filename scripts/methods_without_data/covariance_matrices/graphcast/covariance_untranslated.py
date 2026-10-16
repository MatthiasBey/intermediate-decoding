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
data_dir = f"{config.LATENT_DIR}/graphcast"
SAVE_DIR = f"{config.COVARIANCE_DIR}/graphcast"
os.makedirs(SAVE_DIR, exist_ok=True)
PLOT_DIR = f"{config.FIGURES_DIR}/covariance_diagnostics/graphcast"  # diagnostic plots
os.makedirs(PLOT_DIR, exist_ok=True)
device = torch.device("cpu")

NUM_STEPS = 17  # latents step_0 ... step_16
STEP_KEYS = [f"step_{i}" for i in range(NUM_STEPS)]

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

with np.load(latent_files[0]) as f0:
    N, _, D = f0[STEP_KEYS[0]].shape  # (N, 1, D)
print(f"Mesh nodes N={N}, hidden dim D={D}")

# ---------------------------
# Streaming covariance accumulation
# ---------------------------
# Average over mesh nodes i of the temporal covariance of each node,
#   avg_cov = (1/N) sum_i Cov_t(x_{t,i}),
# computed in one pass over the files with running sums (float64):
#   sum_x = sum_t X_t (N, D),  sum_xx = sum_t X_t^T X_t (D, D)
#   avg_cov = (sum_xx - T * mean^T mean) / (N * (T - 1)),  mean = sum_x / T
sum_x = {k: torch.zeros(N, D, dtype=torch.float64, device=device) for k in STEP_KEYS}
sum_xx = {k: torch.zeros(D, D, dtype=torch.float64, device=device) for k in STEP_KEYS}
counts = {k: 0 for k in STEP_KEYS}

print_ram_usage("Before streaming")

for fi, path in enumerate(latent_files):
    with np.load(path) as f:
        for k in STEP_KEYS:
            try:
                x_np = f[k].squeeze(1)  # (N, D)
            except Exception as e:
                print(f"[{os.path.basename(path)}] Skipping {k}: {e}")
                continue
            x = torch.from_numpy(x_np).to(dtype=torch.float64, device=device)
            sum_x[k] += x
            sum_xx[k] += x.T @ x
            counts[k] += 1

    if fi % 100 == 0 or fi == len(latent_files) - 1:
        print(f"Processed {fi + 1}/{len(latent_files)}: {os.path.basename(path)}")
        print_ram_usage(f"After file {fi + 1}")

print_ram_usage("After streaming all files")

# ---------------------------
# Finalize, save, and plot per step
# ---------------------------
def finalize_and_plot(step_idx, key):
    T = counts[key]
    if T < 2:
        print(f"[Step {step_idx:02d}] Skipping: only {T} usable timestep(s)")
        return

    mean = sum_x[key] / T                    # [N, D]
    mean_outer_sum = mean.T @ mean            # [D, D] = sum_i mean_i mean_i^T
    avg_cov = (sum_xx[key] - T * mean_outer_sum) / (N * (T - 1))
    avg_cov_np = avg_cov.to(torch.float32).cpu().numpy()

    print(f"[Step {step_idx:02d}] Used {T}/{len(latent_files)} timesteps")

    # Save
    np.savez_compressed(os.path.join(SAVE_DIR, f"{step_idx:02d}_avg_covariance_step{step_idx:02d}.npz"), avg_cov=avg_cov_np)
    print(f"[Step {step_idx:02d}] Saved avg_covariance_step{step_idx:02d}.npz")

    # Metrics
    min_val = avg_cov_np.min()
    max_val = avg_cov_np.max()
    mean_val = avg_cov_np.mean()
    diag_mean = np.mean(np.diag(avg_cov_np))
    print(f"[Step {step_idx:02d}] Min: {min_val}")
    print(f"[Step {step_idx:02d}] Max: {max_val}")
    print(f"[Step {step_idx:02d}] Mean: {mean_val}")
    print(f"[Step {step_idx:02d}] Diagonal mean: {diag_mean}")

    with open(f"{step_idx:02d}_metrics_step{step_idx:02d}.txt", "w") as f:
        f.write(f"Min: {min_val}\n")
        f.write(f"Max: {max_val}\n")
        f.write(f"Mean: {mean_val}\n")
        f.write(f"Diagonal mean: {diag_mean}\n")

    # Plot variance per dimension
    plt.figure()
    plt.plot(np.diag(avg_cov_np))
    plt.title(f"Temporal Variance per Dimension (Step {step_idx:02d})")
    plt.xlabel("Hidden Dimension Index")
    plt.ylabel("Variance")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{step_idx:02d}_variance_per_dimension_step{step_idx:02d}.png")
    plt.close()

    # Eigenvalue spectrum
    eigvals = np.linalg.eigvalsh(avg_cov_np)[::-1]
    plt.figure()
    plt.plot(eigvals)
    plt.title(f"Eigenvalue Spectrum (Step {step_idx:02d})")
    plt.xlabel("Principal Component Index")
    plt.ylabel("Eigenvalue (Variance)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{step_idx:02d}_eigvals_step{step_idx:02d}.png")
    plt.close()

    # Heatmap (clipped)
    plt.figure(figsize=(10, 8))
    plt.imshow(np.clip(avg_cov_np, -1, 1), cmap='viridis', interpolation='nearest')
    plt.title(f"Averaged Covariance Matrix (Step {step_idx:02d})")
    plt.colorbar(label="Covariance")
    plt.xlabel("Hidden dimension")
    plt.ylabel("Hidden dimension")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{step_idx:02d}_avg_covariance_step{step_idx:02d}.png", dpi=300)
    plt.close()

    print(f"[Step {step_idx:02d}] Plots saved.")


for step_idx, key in enumerate(STEP_KEYS):
    try:
        finalize_and_plot(step_idx, key)
    except Exception as e:
        print(f"[Step {step_idx:02d}] Error: {e}")