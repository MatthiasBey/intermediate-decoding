"""Translators for the Aurora decoder stage 0 latents (x_24 ... x_30, 2048 channels).

Usage: python train_translators_decoder_stage0.py <model_id>
Maps x_<model_id> to the output of decoder stage 0 (x_31).
Trained in float32 (float16 gave NaN losses); early stopping on the val loss.
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

data_dir = f"{config.LATENT_DIR}/aurora/"

log_dir = f"{config.TRANSLATOR_DIR}/aurora/skip/logs/"

os.makedirs(log_dir, exist_ok=True)

checkpoint_path = f"{config.TRANSLATOR_DIR}/aurora/skip/weights/translator_checkpoint_{model_id}_gnn.pt"
model_save_path = f"{config.TRANSLATOR_DIR}/aurora/skip/weights/translator_matrix_{model_id}_gnn.npz"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

learning_rate = 5e-4
max_epochs = 20      # upper limit, training normally stops earlier
rtol = 0.001         # early stopping: improvement of val loss below 0.1% counts as no improvement
patience = 1         # stop after this many epochs without improvement
batch_size = 2
val_fraction = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% Logging

all_log_file = open(os.path.join(log_dir, f"all_prints_{model_id}.txt"), "w")
train_loss_file = open(os.path.join(log_dir, f"train_loss_{model_id}.txt"), "w")
val_loss_file = open(os.path.join(log_dir, f"val_loss_{model_id}.txt"), "w")

def log_print(*args):
    msg = " ".join(map(str, args))
    print(msg)
    print(msg, file=all_log_file, flush=True)

def print_ram_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    log_print(f"[{tag}] RAM usage: {mem:.2f} MB")

log_print(f"Using model_id = {model_id}")

# %% Latent loading

def load_latents_for_timestamp(data_dir, step, dt):
    time_str = dt.strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"all_latents_aurora-ft_{time_str}.npz"
    path = os.path.join(data_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # mmap avoids copying large arrays
    with np.load(path, mmap_mode="r") as data:
        x = data[f"x_{step}"]
        y = data["x_31"]  # target: output of decoder stage 0
    return x, y

# %% Dataset

class LatentTimestampDataset(Dataset):
    def __init__(self, data_dir, timestamps, translator_step):
        self.data_dir = data_dir
        self.timestamps = timestamps
        self.translator_step = translator_step

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, idx):
        dt = self.timestamps[idx]

        x, y = load_latents_for_timestamp(
            self.data_dir,
            step=self.translator_step,
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
    translator_step=model_id
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

class LinearTranslator(nn.Module):
    def __init__(self, in_dim=2048, out_dim=2048):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        return self.linear(x)
    
model = LinearTranslator(in_dim=2048, out_dim=2048).to(device) # float32 (float16 gave NaN losses)

# Identity init (not when resuming from a checkpoint)
if not os.path.exists(checkpoint_path):
    with torch.no_grad():
        weight = torch.zeros(2048, 2048)
        weight[:, :] = torch.eye(2048)
        model.linear.weight.copy_(weight)
        model.linear.bias.zero_() 
    log_print("Initialized with identity weights")

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
log_print(f"Initial train loss (identity weights): {initial_train_loss:.6f}")

initial_val_loss = 0.0
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device, dtype=torch.float32)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        initial_val_loss += loss.item() * xb.size(0)

initial_val_loss /= val_size
log_print(f"Initial val loss (identity weights): {initial_val_loss:.6f}")


# %% Training loop

for epoch in range(start_epoch, start_epoch + max_epochs):
    model.train()
    train_loss = 0.0

    print_ram_usage(f"Before epoch {epoch+1}")

    for xb, yb in train_loader:
        print_ram_usage("Loading a batch for training")
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * xb.size(0)
        print_ram_usage("After processing a batch for training")
        log_print(f"Train loss: {loss}")

    train_loss /= train_size

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for xb, yb in val_loader:
            print_ram_usage(f"Loading a batch for validation, epoch {epoch+1}")
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

# W is stored transposed, shape (in_dim, out_dim): apply as x @ W + b
model.eval()
W = model.linear.weight.detach().cpu().numpy().T
b = model.linear.bias.detach().cpu().numpy()

np.savez(model_save_path, W=W, b=b)
log_print(f"Saved final model to {model_save_path}")

# %% Cleanup

all_log_file.close()
train_loss_file.close()
val_loss_file.close()
