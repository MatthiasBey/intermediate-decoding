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
ENCODER_DATA_DIR = f"{config.LATENT_DIR}/aurora/encoder_output"
ENCODER_KEY = "encoder_output"

TRANSLATOR_DIR = f"{config.TRANSLATOR_DIR}/aurora/skip/weights/"
NO_SKIP_TRANSLATOR_DIR = f"{config.TRANSLATOR_DIR}/aurora/no_skip/weights/"

# Output file names get the prefix "withtranslator_" to distinguish them from
# the output of covariance_untranslated.py.
SAVE_DIR = f"{config.COVARIANCE_DIR}/aurora"
os.makedirs(SAVE_DIR, exist_ok=True)
PLOT_DIR = f"{config.FIGURES_DIR}/covariance_diagnostics/aurora"  # diagnostic plots
os.makedirs(PLOT_DIR, exist_ok=True)
device = torch.device("cpu")

STAGE_BLOCK_COUNTS = {
    ("enc", 0): 6,
    ("enc", 1): 10,
    ("enc", 2): 8,
    ("dec", 0): 8,
    ("dec", 1): 10,
    ("dec", 2): 6,
}
BLOCKS = [(part, stage, b) for (part, stage), n in STAGE_BLOCK_COUNTS.items() for b in range(n)]
ALL_KEYS = [ENCODER_KEY] + BLOCKS  # indexed by the command line argument

# Index i of x_i for each (part, stage, block), in the order of the latent files
FLAT_INDEX = {}
_i = 0
for (_part, _stage), _n in STAGE_BLOCK_COUNTS.items():
    for _b in range(_n):
        FLAT_INDEX[(_part, _stage, _b)] = _i
        _i += 1

# All translated latents live in the space of the backbone output concat(x_47, x_5),
# the target of the Aurora translators (see train_no_skip_translators.py).
COMMON_TOKENS = 259200
COMMON_DIM = 1024

def tag_for(key):
    if key == ENCODER_KEY:
        return ENCODER_KEY
    part, stage, block = key
    return f"{part}_stage{stage}_block_{block}"

# ---------------------------
# Command line argument: index into ALL_KEYS (0 to 48). Each run computes the
# covariance matrix of one key.
model_id = int(sys.argv[1])
target_key = ALL_KEYS[model_id]
tag = tag_for(target_key)
print(f"model_id={model_id}, target_key={target_key}, tag={tag}")

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
# Load the translator for this key
# ---------------------------
# Translator kinds:
#   raw_plus_skip     x_47 (dec stage 2, block 5): concat(x_47, x_5), no translator
#   affine_direct     encoder output and enc stage 0: translated to concat(x_47, x_5)
#                     (train_translator_encoder_output.py, train_translators_encoder_stage0.py)
#   affine_plus_skip  dec stage 2, blocks 0-4: translated to x_47, then concatenated
#                     with the raw x_5 (train_translators_decoder_stage2.py)
#   upscale           enc stages 1, 2 and dec stages 0, 1: UpscaleTranslator
#                     (train_no_skip_translators.py), includes spatial upsampling
# The skip translators store W transposed (apply as x @ W + b), the upscale
# translators untransposed (apply as x @ W.T + b, then reshape).
# The training scripts save files with flat indices (translator_matrix_{i}_gnn.npz,
# translator_enc_matrix_enc_gnn.npz); the published weights use the names
# translator_matrix_{part}_{stage}_{block}_gnn.npz and
# translator_matrix_encoder_output_gnn.npz.
UPSCALE_STAGE_PARAMS = {
    # (part, stage): (stage_name_in_filename, c_lat, h_src, w_src, block_factor)
    ("enc", 1): ("enc_1", 4, 90, 180, 2),
    ("enc", 2): ("enc_2", 4, 45, 90, 4),
    ("dec", 0): ("dec_0", 4, 45, 90, 4),
    ("dec", 1): ("dec_1", 4, 90, 180, 2),
}

if target_key == ENCODER_KEY:
    npz = np.load(os.path.join(TRANSLATOR_DIR, "translator_matrix_encoder_output_gnn.npz"))
    translator = {"kind": "affine_direct", "W": npz["W"].astype(np.float32), "b": npz["b"].astype(np.float32)}
else:
    part, stage, block = target_key
    if (part, stage) == ("enc", 0):
        npz = np.load(os.path.join(TRANSLATOR_DIR, f"translator_matrix_enc_0_{block}_gnn.npz"))
        translator = {"kind": "affine_direct", "W": npz["W"].astype(np.float32), "b": npz["b"].astype(np.float32)}
    elif (part, stage) == ("dec", 2) and block == 5:
        # x_47 is part of the target space itself, no translator needed
        translator = {"kind": "raw_plus_skip"}
    elif (part, stage) == ("dec", 2):
        npz = np.load(os.path.join(TRANSLATOR_DIR, f"translator_matrix_dec_2_{block}_gnn.npz"))
        translator = {"kind": "affine_plus_skip", "W": npz["W"].astype(np.float32), "b": npz["b"].astype(np.float32)}
    elif (part, stage) in UPSCALE_STAGE_PARAMS:
        stage_name, c_lat, h_src, w_src, bf = UPSCALE_STAGE_PARAMS[(part, stage)]
        path = os.path.join(NO_SKIP_TRANSLATOR_DIR, f"translator_matrix_upscale_{stage_name}_{block}_gnn.npz")
        if not os.path.exists(path):
            print(f"[{tag}] Translator not trained yet ({path}) - nothing to do this run. Exiting.")
            sys.exit(0)
        npz = np.load(path)
        translator = {
            "kind": "upscale", "W": npz["W"].astype(np.float32), "b": npz["b"].astype(np.float32),
            "c_lat": c_lat, "h_src": h_src, "w_src": w_src, "bf": bf,
        }
    else:
        raise ValueError(f"Unrecognized key {target_key} - not covered by any known translator family")

kind = translator["kind"]
print(f"[{tag}] kind={kind}")

# ---------------------------
# Translator application
# ---------------------------
def apply_translator(entry, x_np, x5_raw, x47_raw):
    """x_np: (N, D_in) raw latent (unused for raw_plus_skip). Returns (COMMON_TOKENS,
    COMMON_DIM)."""
    if kind == "raw_plus_skip":
        return np.concatenate([x47_raw, x5_raw], axis=-1)
    if kind == "affine_direct":
        return x_np @ entry["W"] + entry["b"]
    if kind == "affine_plus_skip":
        y = x_np @ entry["W"] + entry["b"]  # (N, 512)
        return np.concatenate([y, x5_raw], axis=-1)
    if kind == "upscale":
        c_lat, h_src, w_src, bf = entry["c_lat"], entry["h_src"], entry["w_src"], entry["bf"]
        x = x_np @ entry["W"].T + entry["b"]  # (N, COMMON_DIM*bf*bf)
        x = x.reshape(c_lat, h_src, w_src, bf, bf, COMMON_DIM)
        # Same as permute(0, 1, 2, 4, 3, 5, 6) in UpscaleTranslator.forward(), without
        # the batch axis
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(c_lat * h_src * bf * w_src * bf, COMMON_DIM)
        return x
    raise ValueError(f"Unknown translator kind: {kind}")

# ---------------------------
# Streaming covariance accumulation
# ---------------------------
# Running sums as in ../graphcast/covariance_untranslated.py, at the common shape
# (COMMON_TOKENS, COMMON_DIM). x_5 and x_47 are only loaded if the translator kind
# needs them.
NEEDS_X5 = {"affine_plus_skip", "raw_plus_skip"}
NEEDS_X47 = {"raw_plus_skip"}

sum_x = torch.zeros(COMMON_TOKENS, COMMON_DIM, dtype=torch.float64, device=device)
sum_xx = torch.zeros(COMMON_DIM, COMMON_DIM, dtype=torch.float64, device=device)
count = 0
i = None if target_key == ENCODER_KEY else FLAT_INDEX[target_key]

print_ram_usage("Before streaming")

for fi, path in enumerate(latent_files):
    try:
        if target_key == ENCODER_KEY:
            encoder_path = os.path.join(ENCODER_DATA_DIR, os.path.basename(path))
            with np.load(encoder_path, allow_pickle=False) as fenc:
                x_np = fenc["encoder_x"][0].astype(np.float32, copy=False)
            x5_raw = x47_raw = None
        else:
            with np.load(path) as f:
                x5_raw = f[f"x_{FLAT_INDEX[('enc', 0, 5)]}"][0].astype(np.float32, copy=False) if kind in NEEDS_X5 else None
                x47_raw = f[f"x_{FLAT_INDEX[('dec', 2, 5)]}"][0].astype(np.float32, copy=False) if kind in NEEDS_X47 else None
                x_np = None if kind == "raw_plus_skip" else f[f"x_{i}"][0].astype(np.float32, copy=False)
    except Exception as e:
        print(f"[{os.path.basename(path)}] Skipping {tag}: {e}")
        continue

    x_translated = apply_translator(translator, x_np, x5_raw, x47_raw)
    x = torch.from_numpy(x_translated).to(dtype=torch.float64, device=device)
    sum_x += x
    sum_xx += x.T @ x
    count += 1

    if fi % 100 == 0 or fi == len(latent_files) - 1:
        print(f"Processed {fi + 1}/{len(latent_files)}: {os.path.basename(path)}")
        print_ram_usage(f"After file {fi + 1}")

print_ram_usage("After streaming all files")

# ---------------------------
# Finalize, save, and plot
# ---------------------------
def finalize_and_plot():
    if count < 2:
        print(f"[{tag}] Skipping: only {count} usable timestep(s)")
        return

    mean = sum_x / count                       # [N, D]
    mean_outer_sum = mean.T @ mean              # [D, D] = sum_i mean_i mean_i^T
    avg_cov = (sum_xx - count * mean_outer_sum) / (COMMON_TOKENS * (count - 1))
    avg_cov_np = avg_cov.to(torch.float32).cpu().numpy()

    print(f"[{tag}] Used {count}/{len(latent_files)} timesteps, D={avg_cov_np.shape[0]}")

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
    finalize_and_plot()
    print_ram_usage("Done")
