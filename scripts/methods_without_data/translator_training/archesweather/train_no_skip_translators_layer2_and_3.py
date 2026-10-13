import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import psutil
from datetime import datetime

# ---------------------------
# 0. Command line argument
# ---------------------------
model_id = int(sys.argv[1])

# model_id selects the (layer, block) latent to translate (layers 2 and 3)
list_of_tuples = [
    # Layer 2
    [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5],
    [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11],
    # Layer 3
    [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5],
    [3, 6], [3, 7], [3, 8], [3, 9], [3, 10], [3, 11]
]

layer_x, block_x = list_of_tuples[model_id]
layer_y, block_y = 4, 3  # target: last block of layer 4

# ---------------------------
# 1. Logging setup
# ---------------------------
output_dir = f"{config.TRANSLATOR_DIR}/archesweather/no_skip/weights"
os.makedirs(output_dir, exist_ok=True)

log_dir = f"{config.TRANSLATOR_DIR}/archesweather/no_skip/logs"
os.makedirs(log_dir, exist_ok=True)

# When resuming from a checkpoint, append to the existing logs
checkpoint_path = os.path.join(output_dir, f"Arches_translator_rep_stab_layer{layer_x}_block{block_x}.pt")
resuming = os.path.exists(checkpoint_path)

all_log_path    = os.path.join(log_dir, f"layer{layer_x}_block{block_x}_all.txt")
train_loss_path = os.path.join(log_dir, f"layer{layer_x}_block{block_x}_train_loss.txt")
val_loss_path   = os.path.join(log_dir, f"layer{layer_x}_block{block_x}_val_loss.txt")

all_log_file    = open(all_log_path, "a" if resuming else "w")
train_loss_file = open(train_loss_path, "a" if resuming else "w")
val_loss_file   = open(val_loss_path, "a" if resuming else "w")

def log_print(*args, **kwargs):
    msg = " ".join(map(str, args))
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{time_str}] {msg}"
    print(full_msg, **kwargs)
    print(full_msg, file=all_log_file, flush=True)

def print_ram_usage(tag=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    log_print(f"[{tag}] RAM usage: {mem:.2f} MB")

if resuming:
    log_print("=== Resuming training (existing checkpoint found) ===")

# ---------------------------
# 2. Config
# ---------------------------
data_dir = f"{config.LATENT_DIR}/archesweather"
batch_size = 2         # files (timestamps) per batch
max_epochs = 20        # upper limit, training normally stops earlier
rtol = 0.001           # early stopping: improvement of val loss below 0.1% counts as no improvement
patience = 1           # stop after this many epochs without improvement
val_fraction = 0.2
learning_rate = 5e-4
device = torch.device("cpu")

model_save_path = os.path.join(output_dir, f"Arches_translator_rep_stab_matrix_layer{layer_x}_block{block_x}.npz")
init_save_path  = os.path.join(output_dir, f"Arches_translator_rep_stab_matrix_layer{layer_x}_block{block_x}_INIT.npz")

# ---------------------------
# 3. Load Data
# ---------------------------
# Layers 2 and 3 (x) are on the coarse grid (pl=8, lat=30, lon=60; 14400 tokens),
# layer4.blocks.3 (y) on the fine grid (pl=8, lat=60, lon=120; 57600 tokens).
# The translator learns the 2x2 lat/lon upsampling (ExpandTranslator).
PL, COARSE_LAT, COARSE_LON = 8, 30, 60

def key_for(layer, block):
    if layer == 0:
        return f"layer{layer}_block{block}"
    return f"layer{layer}.blocks.{block}"

def list_latent_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "all_latents_*.npz")))
    if not files:
        raise FileNotFoundError(f"No latent files found in {data_dir}")
    return files

def load_file_xy(path, key_x, key_y):
    with np.load(path) as f:
        x = f[key_x][0].astype(np.float32, copy=False)  # (14400, in_dim)
        y = f[key_y][0].astype(np.float32, copy=False)  # (57600, out_dim)
    return x, y

key_x = key_for(layer_x, block_x)
key_y = key_for(layer_y, block_y)
latent_files = list_latent_files(data_dir)

with np.load(latent_files[0]) as f0:
    tokens_per_file = f0[key_x].shape[1]

total_samples = len(latent_files) * tokens_per_file

class LatentFileDataset(Dataset):
    """One sample is one file (timestamp) with all its tokens."""
    def __init__(self, files, key_x, key_y):
        self.files = files
        self.key_x = key_x
        self.key_y = key_y

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x, y = load_file_xy(self.files[idx], self.key_x, self.key_y)
        return torch.from_numpy(x), torch.from_numpy(y)

full_dataset = LatentFileDataset(latent_files, key_x, key_y)

val_size = int(len(full_dataset) * val_fraction)
train_size = len(full_dataset) - val_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], generator=generator
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

log_print(f"Training translator for layer {layer_x}, block {block_x}")
log_print(f"Total files: {len(latent_files)} ({tokens_per_file} tokens/file, {total_samples} total samples)")
log_print(f"Train files: {train_size}, Val files: {val_size}")
print_ram_usage("Before training")

# ---------------------------
# 4. Define Model
# ---------------------------
class ExpandTranslator(nn.Module):
    """
    Affine map nn.Linear(in_dim, 4 * out_dim) followed by a reshape that places the
    4 outputs of each coarse token on a 2x2 lat/lon patch of the fine grid (same
    token layout as y). Unlike ArchesWeather's UpSample module, there is no
    LayerNorm or second linear layer.
    """
    def __init__(self, in_dim, out_dim, pl, coarse_lat, coarse_lon):
        super().__init__()
        self.out_dim = out_dim
        self.pl = pl
        self.coarse_lat = coarse_lat
        self.coarse_lon = coarse_lon
        self.linear = nn.Linear(in_dim, out_dim * 4, bias=True)
        with torch.no_grad():
            # Identity-like init: all 4 positions of the patch start as a copy of
            # the input (in_dim == out_dim)
            self.linear.weight.copy_(torch.eye(in_dim).repeat(4, 1))
            self.linear.bias.zero_()

    def forward(self, x):
        # x: (B, pl*coarse_lat*coarse_lon, in_dim)
        B = x.shape[0]
        x = self.linear(x)  # (B, pl*coarse_lat*coarse_lon, out_dim*4)
        x = x.reshape(B, self.pl, self.coarse_lat, self.coarse_lon, 2, 2, self.out_dim)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)  # (B, pl, coarse_lat, 2, coarse_lon, 2, out_dim)
        x = x.reshape(B, self.pl * self.coarse_lat * 2 * self.coarse_lon * 2, self.out_dim)
        return x

# 384 channels in and out; the output has 4 times as many tokens
model = ExpandTranslator(in_dim=384, out_dim=384, pl=PL, coarse_lat=COARSE_LAT, coarse_lon=COARSE_LON).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Save the initial weights
W_init = model.linear.weight.detach().cpu().numpy()
b_init = model.linear.bias.detach().cpu().numpy()
np.savez(init_save_path, W=W_init, b=b_init)
log_print(f"Initial weights saved to: {init_save_path}")

# ---------------------------
# 5. Helper: average loss over a loader (no grad)
# ---------------------------
def evaluate_avg_loss(loader):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device, dtype=torch.float32)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)
    return total_loss / n_samples

# ---------------------------
# Initial loss on train and val sets
# ---------------------------
# Skipped when resuming, since the initial loss is already in the logs
if resuming:
    log_print("Resuming from existing checkpoint - skipping duplicate initial-loss computation "
              "(epoch-0 loss was already recorded by the original run).")
else:
    log_print("Computing initial loss on train and val sets...")

    initial_train_loss = evaluate_avg_loss(train_loader)
    log_print(f"Initial train loss: {initial_train_loss:.6f}")
    print(initial_train_loss, file=train_loss_file, flush=True)

    initial_val_loss = evaluate_avg_loss(val_loader)
    log_print(f"Initial val loss: {initial_val_loss:.6f}")
    print(initial_val_loss, file=val_loss_file, flush=True)

# ---------------------------
# 6. Resume checkpoint if available
# ---------------------------
start_epoch = 0
best_val_loss = float("inf")
num_bad_epochs = 0
best_state_dict = None

if resuming:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float("inf"))
    num_bad_epochs = checkpoint.get('num_bad_epochs', 0)
    best_state_dict = checkpoint.get('best_state_dict', None)
    log_print(f"Resuming from checkpoint at epoch {start_epoch}")

# ---------------------------
# 7. Training Loop
# ---------------------------
for epoch in range(start_epoch, start_epoch + max_epochs):
    model.train()
    total_loss = 0.0
    n_samples = 0
    log_print(f"Epoch {epoch + 1}/{start_epoch + max_epochs}")

    for xb, yb in train_loader:
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)

    train_loss = total_loss / n_samples
    val_loss = evaluate_avg_loss(val_loader)

    print(train_loss, file=train_loss_file, flush=True)
    print(val_loss, file=val_loss_file, flush=True)

    log_print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    # Early stopping: improvement relative to the best val loss so far
    if val_loss < best_val_loss * (1 - rtol):
        best_val_loss = val_loss
        num_bad_epochs = 0
        best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        num_bad_epochs += 1

    log_print(f"  best_val_loss={best_val_loss:.6f}, num_bad_epochs={num_bad_epochs}/{patience}")

    # Checkpoint for resuming, including the best weights so far
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'num_bad_epochs': num_bad_epochs,
        'best_state_dict': best_state_dict,
    }, checkpoint_path)
    log_print(f"Checkpoint saved to: {checkpoint_path}")

    if num_bad_epochs >= patience:
        log_print(f"Converged: relative improvement below {rtol*100:.2f}% "
                   f"for {patience} consecutive epoch(s). Stopping at epoch {epoch + 1}.")
        break
else:
    log_print(f"WARNING: reached max_epochs={max_epochs} without meeting the "
               f"convergence criterion (rtol={rtol*100:.2f}%, patience={patience}). "
               f"Using best checkpoint found (val_loss={best_val_loss:.6f}).")

log_print("Training complete")

# Restore the weights with the lowest val loss
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

# ---------------------------
# 8. Save Final Weights
# ---------------------------
# W is stored untransposed, shape (1536, 384): apply as x @ W.T + b, followed by
# the reshape in ExpandTranslator.forward().
W = model.linear.weight.detach().cpu().numpy()
b = model.linear.bias.detach().cpu().numpy()
np.savez(model_save_path, W=W, b=b)
log_print(f"Final model weights saved to: {model_save_path}")

# ---------------------------
# 9. Close logs
# ---------------------------
all_log_file.close()
train_loss_file.close()
val_loss_file.close()
