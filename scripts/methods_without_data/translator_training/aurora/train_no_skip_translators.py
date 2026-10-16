"""Translators from the coarser Aurora stages directly to the backbone output.

Usage: python train_no_skip_translators.py <model_id>
model_id indexes all blocks of encoder stages 1 and 2 and decoder stages 0 and 1
(see STAGE_CONFIGS). The skip translators map to the output of the stage with
the same resolution; these instead map directly to the final backbone output
concat(x_47, x_5) (180 x 360 grid, 1024 channels), so the translator also
learns the spatial upsampling (UpscaleTranslator).
"""
import os
import sys
import psutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # scripts/
import config

# %% Configuration

model_id = int(sys.argv[1])
print(f"Using model_id = {model_id}")

# (first_x_index, count, in_dim, h_src, w_src, block_factor, stage_name)
# block_factor: upsampling factor in lat and lon to the 180 x 360 output grid
STAGE_CONFIGS = [
    (6,  10, 1024, 90, 180, 2, "enc_1"),   # encoder stage 1
    (16, 8,  2048, 45, 90,  4, "enc_2"),   # encoder stage 2 / bottleneck
    (24, 8,  2048, 45, 90,  4, "dec_0"),   # decoder stage 0
    (32, 10, 1024, 90, 180, 2, "dec_1"),   # decoder stage 1
]

list_of_configs = [
    (f"x_{first + b}", in_dim, h_src, w_src, bf, stage_name, b)
    for first, count, in_dim, h_src, w_src, bf, stage_name in STAGE_CONFIGS
    for b in range(count)
]

source_key, in_dim, h_src, w_src, block_factor, stage_name, block_idx = list_of_configs[model_id]
tag = f"upscale_{stage_name}_{block_idx}"

# Target token grid: C_LAT latent levels x H_OUT x W_OUT, OUT_DIM channels
C_LAT = 4
OUT_DIM = 1024
H_OUT, W_OUT = 180, 360

print(f"source_key={source_key}, in_dim={in_dim}, h_src={h_src}, w_src={w_src}, "
      f"block_factor={block_factor}, tag={tag}")

data_dir = f"{config.LATENT_DIR}/aurora/"

log_dir = f"{config.TRANSLATOR_DIR}/aurora/no_skip/logs/"

os.makedirs(log_dir, exist_ok=True)

checkpoint_path = f"{config.TRANSLATOR_DIR}/aurora/no_skip/weights/translator_checkpoint_{tag}_gnn.pt"
model_save_path = f"{config.TRANSLATOR_DIR}/aurora/no_skip/weights/translator_matrix_{tag}_gnn.npz"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

learning_rate = 5e-4
max_epochs = 30      # upper limit, training normally stops earlier
rtol = 0.001         # early stopping: improvement of val loss below 0.1% counts as no improvement
patience = 1         # stop after this many epochs without improvement
batch_size = 2
val_fraction = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% Logging

all_log_file = open(os.path.join(log_dir, f"all_prints_{tag}.txt"), "a")
train_loss_file = open(os.path.join(log_dir, f"train_loss_{tag}.txt"), "a")
val_loss_file = open(os.path.join(log_dir, f"val_loss_{tag}.txt"), "a")

def log_print(*args):
    msg = " ".join(map(str, args))
    print(msg)
    print(msg, file=all_log_file, flush=True)

def print_ram_usage(label=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    log_print(f"[{label}] RAM usage: {mem:.2f} MB")

log_print(f"Using model_id = {model_id}, tag = {tag}")

# %% Latent loading

def load_latents_for_timestamp(data_dir, source_key, dt):
    time_str = dt.strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"all_latents_aurora-ft_{time_str}.npz"
    path = os.path.join(data_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # mmap avoids copying large arrays
    with np.load(path, mmap_mode="r") as data:
        x = data[source_key]
        y1 = data["x_5"]
        y2 = data["x_47"]
        y = np.concatenate([y2, y1], axis=2)  # target: backbone output, concat(x_47, x_5) along channels
    return x, y

# %% Dataset

class LatentTimestampDataset(Dataset):
    def __init__(self, data_dir, timestamps, source_key):
        self.data_dir = data_dir
        self.timestamps = timestamps
        self.source_key = source_key

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        dt = self.timestamps[idx]

        x, y = load_latents_for_timestamp(
            self.data_dir,
            source_key=self.source_key,
            dt=dt
        )

        # remove batch dim
        x = x.squeeze(0)
        y = y.squeeze(0)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        return x, y

# %% Timestamp discovery

log_print("Scanning data directory for timestamps...")

timestamps = sorted([
    datetime.strptime(f.split("_")[-1][:-4], "%Y-%m-%d-%H-%M-%S")
    for f in os.listdir(data_dir)
    if f.startswith("all_latents_") and f.endswith(".npz")
])

log_print(f"Found {len(timestamps)} timestamps")

assert len(timestamps) > 0, "No latent files found!"

# %% Train/validation split and data loaders

full_dataset = LatentTimestampDataset(
    data_dir=data_dir,
    timestamps=timestamps,
    source_key=source_key
)

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
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

log_print(f"Train size: {train_size}, Val size: {val_size}")
print_ram_usage("After DataLoader creation")

# %% Model

class UpscaleTranslator(nn.Module):
    """
    Affine map nn.Linear(in_dim, out_dim * bf**2) followed by a reshape that places
    the bf * bf outputs of each source token on a bf x bf lat/lon patch of the output
    grid (bf = block_factor). Aurora version of ExpandTranslator in
    ../archesweather/train_no_skip_translators_layer2_and_3.py.
    """
    def __init__(self, in_dim, out_dim, c_lat, h_src, w_src, block_factor):
        super().__init__()
        self.out_dim = out_dim
        self.c_lat = c_lat
        self.h_src = h_src
        self.w_src = w_src
        self.bf = block_factor
        self.linear = nn.Linear(in_dim, out_dim * block_factor * block_factor, bias=True)
        with torch.no_grad():
            # Identity-like init: every position of the patch starts as a copy of
            # the input (first out_dim channels if in_dim > out_dim)
            base = torch.eye(in_dim)[:out_dim, :]  # (out_dim, in_dim)
            self.linear.weight.copy_(base.repeat(block_factor * block_factor, 1))
            self.linear.bias.zero_()

    def forward(self, x):
        B = x.shape[0]
        bf = self.bf
        x = self.linear(x)  # (B, c_lat*h_src*w_src, out_dim*bf*bf)
        x = x.reshape(B, self.c_lat, self.h_src, self.w_src, bf, bf, self.out_dim)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)  # (B, c_lat, h_src, bf, w_src, bf, out_dim)
        x = x.reshape(B, self.c_lat * self.h_src * bf * self.w_src * bf, self.out_dim)
        return x

model = UpscaleTranslator(
    in_dim=in_dim, out_dim=OUT_DIM, c_lat=C_LAT,
    h_src=h_src, w_src=w_src, block_factor=block_factor,
).to(device)  # float32

if not os.path.exists(checkpoint_path):
    log_print("Initialized with identity-flavored weights")

# %% Optimizer and checkpoint

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

start_epoch = 0
best_val_loss = float("inf")
num_bad_epochs = 0
best_state_dict = None

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    num_bad_epochs = checkpoint.get("num_bad_epochs", 0)
    best_state_dict = checkpoint.get("best_state_dict", None)
    log_print(f"Resumed from epoch {start_epoch}")

# %% Initial loss
model.eval()
initial_train_loss = 0.0
with torch.no_grad():
    for xb, yb in train_loader:
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device, dtype=torch.float32)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        initial_train_loss += loss.item() * xb.size(0)

initial_train_loss /= train_size
log_print(f"Initial train loss: {initial_train_loss:.6f}")

initial_val_loss = 0.0
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device, dtype=torch.float32)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        initial_val_loss += loss.item() * xb.size(0)

initial_val_loss /= val_size
log_print(f"Initial val loss: {initial_val_loss:.6f}")

# %% Training loop

for epoch in range(start_epoch, start_epoch + max_epochs):
    model.train()
    train_loss = 0.0

    print_ram_usage(f"Before epoch {epoch+1}")

    for xb, yb in train_loader:
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * xb.size(0)

    train_loss /= train_size

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, dtype=torch.float32)
            yb = yb.to(device, dtype=torch.float32)

            pred = model(xb)
            loss = loss_fn(pred, yb)
            val_loss += loss.item() * xb.size(0)

    val_loss /= val_size

    print(train_loss, file=train_loss_file, flush=True)
    print(val_loss, file=val_loss_file, flush=True)

    log_print(
        f"Epoch {epoch+1}: "
        f"train_loss={train_loss:.6f}, "
        f"val_loss={val_loss:.6f}"
    )

    # Early stopping: improvement relative to the best val loss so far
    if val_loss < best_val_loss * (1 - rtol):
        best_val_loss = val_loss
        num_bad_epochs = 0
        best_state_dict = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
    else:
        num_bad_epochs += 1

    log_print(
        f"  best_val_loss={best_val_loss:.6f}, "
        f"num_bad_epochs={num_bad_epochs}/{patience}"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "num_bad_epochs": num_bad_epochs,
            "best_state_dict": best_state_dict,
        },
        checkpoint_path,
    )

    print_ram_usage(f"After epoch {epoch+1}")

    if num_bad_epochs >= patience:
        log_print(
            f"Converged: relative improvement below {rtol*100:.2f}% "
            f"for {patience} consecutive epoch(s). Stopping at epoch {epoch+1}."
        )
        break
else:
    log_print(
        f"WARNING: reached max_epochs={max_epochs} without meeting the "
        f"convergence criterion (rtol={rtol*100:.2f}%, patience={patience}). "
        f"Using best checkpoint found (val_loss={best_val_loss:.6f})."
    )

log_print("Training complete.")

# Restore the weights with the lowest val loss
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

# %% Save final matrix
# W is stored untransposed, shape (out_dim * bf**2, in_dim): apply as x @ W.T + b,
# followed by the reshape in UpscaleTranslator.forward().

model.eval()
W = model.linear.weight.detach().cpu().numpy()
b = model.linear.bias.detach().cpu().numpy()

np.savez(
    model_save_path,
    W=W,
    b=b,
    in_dim=in_dim,
    out_dim=OUT_DIM,
    c_lat=C_LAT,
    h_src=h_src,
    w_src=w_src,
    h_out=H_OUT,
    w_out=W_OUT,
    block_factor=block_factor,
    source_key=source_key,
)
log_print(f"Saved final model to {model_save_path}")

# %% Cleanup

all_log_file.close()
train_loss_file.close()
val_loss_file.close()
