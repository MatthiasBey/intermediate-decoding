import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import psutil
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # scripts/
import config

# ---------------------------
# 0. Stage translator: maps the input of layer4.blocks.0 (stored in separate
# layer4_block0_input_*.npz files) to the output of layer4.blocks.3.
# ---------------------------
layer_x, block_x = 5, 5  # label used only in log and checkpoint file names
layer_y, block_y = 4, 3  # target: last block of layer 4

STAGE_INPUT_KEY = 'layer4_block0_input'

# ---------------------------
# 1. Logging setup
# ---------------------------
log_dir = f"{config.TRANSLATOR_DIR}/archesweather/skip/logs"
os.makedirs(log_dir, exist_ok=True)
all_log_file = open(os.path.join(log_dir, f"all_prints_layer{layer_x}_block{block_x}.txt"), "w")
train_loss_file = open(os.path.join(log_dir, f"train_loss_layer{layer_x}_block{block_x}.txt"), "w")
val_loss_file = open(os.path.join(log_dir, f"val_loss_layer{layer_x}_block{block_x}.txt"), "w")

def log(msg):
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{time_str}] {msg}"
    print(line, file=all_log_file, flush=True)
    print(line)

def print_ram_usage(tag=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    log(f"[{tag}] RAM usage: {mem:.2f} MB")

# ---------------------------
# 2. Config
# ---------------------------
data_dir = f"{config.LATENT_DIR}/archesweather"
batch_size = 2        # files (timestamps) per batch
max_epochs = 20       # upper limit, training normally stops earlier
rtol = 0.001          # early stopping: improvement of val loss below 0.1% counts as no improvement
patience = 1          # stop after this many epochs without improvement
val_fraction = 0.2
learning_rate = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_path = f'{config.TRANSLATOR_DIR}/archesweather/skip/weights/'
os.makedirs(save_path, exist_ok=True)

checkpoint_path = os.path.join(save_path, f"Arches_translator_layer{layer_x}_block{block_x}.pt")
model_save_path = os.path.join(save_path, "Arches_stage_translator_layer4.npz")
init_save_path = os.path.join(save_path, f"Arches_translator_matrix_layer{layer_x}_block{block_x}_INIT.npz")

# ---------------------------
# 3. Load Data
# ---------------------------
def key_for(layer, block):
    if layer == 0:
        return f"layer{layer}_block{block}"
    return f"layer{layer}.blocks.{block}"

def date_suffix(all_latents_path):
    # e.g. all_latents_2020-01-03-00-00.npz gives 2020-01-03-00-00
    base = os.path.basename(all_latents_path)
    return base[len("all_latents_"):-len(".npz")]

def list_paired_files(data_dir):
    """Pair each all_latents_*.npz file (target) with the
    layer4_block0_input_*.npz file (input) of the same date. Dates without
    an input file are skipped."""
    y_files = sorted(glob.glob(os.path.join(data_dir, "all_latents_*.npz")))
    if not y_files:
        raise FileNotFoundError(f"No latent files found in {data_dir}")

    pairs = []
    for y_path in y_files:
        suffix = date_suffix(y_path)
        x_path = os.path.join(data_dir, f"{STAGE_INPUT_KEY}_{suffix}.npz")
        if os.path.exists(x_path):
            pairs.append((x_path, y_path))

    if not pairs:
        raise FileNotFoundError(
            f"No '{STAGE_INPUT_KEY}_*.npz' companion files found in {data_dir}. "
            "Extract the layer-4 block-0 input latents first (latent extraction is not included in this repository)."
        )
    if len(pairs) < len(y_files):
        log(f"WARNING: only {len(pairs)}/{len(y_files)} dates have a stage-input "
            f"companion file - missing dates are skipped.")
    return pairs

def load_pair_xy(x_path, y_path, key_y):
    with np.load(x_path) as fx:
        x = fx[STAGE_INPUT_KEY][0].astype(np.float32, copy=False)  # (tokens, in_dim)
    with np.load(y_path) as fy:
        y = fy[key_y][0].astype(np.float32, copy=False)  # (tokens, out_dim)
    return x, y

key_y = key_for(layer_y, block_y)
paired_files = list_paired_files(data_dir)

with np.load(paired_files[0][0]) as f0:
    tokens_per_file = f0[STAGE_INPUT_KEY].shape[1]

total_samples = len(paired_files) * tokens_per_file

class StageInputDataset(Dataset):
    """One sample is one timestamp with all its tokens; x and y come from
    two files of the same date."""
    def __init__(self, paired_files, key_y):
        self.paired_files = paired_files
        self.key_y = key_y

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        x_path, y_path = self.paired_files[idx]
        x, y = load_pair_xy(x_path, y_path, self.key_y)
        return torch.from_numpy(x), torch.from_numpy(y)

full_dataset = StageInputDataset(paired_files, key_y)

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
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

log(f"Training stage translator: input of layer4.blocks.0 -> {key_y}")
log(f"Total files: {len(paired_files)} ({tokens_per_file} tokens/file, {total_samples} total samples)")
log(f"Train files: {train_size}, Val files: {val_size}")
print_ram_usage("Before training")

# ---------------------------
# 4. Define Model
# ---------------------------
class AffineTranslator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

        # Identity init (in_dim == out_dim)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(in_dim))
            self.linear.bias.zero_()

    def forward(self, x):
        return self.linear(x)

model = AffineTranslator(in_dim=384, out_dim=384).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Save the initial weights
W_init = model.linear.weight.detach().cpu().numpy()
b_init = model.linear.bias.detach().cpu().numpy()
np.savez(init_save_path, W=W_init, b=b_init)
log(f"Initial weights saved to: {init_save_path}")

# ---------------------------
# Helper: average loss over a loader (no grad)
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
log("Computing initial loss on train and val sets...")

initial_train_loss = evaluate_avg_loss(train_loader)
log(f"Initial train loss (identity weights): {initial_train_loss:.6f}")

initial_val_loss = evaluate_avg_loss(val_loader)
log(f"Initial val loss (identity weights): {initial_val_loss:.6f}")

# Resume checkpoint if available
start_epoch = 0
best_val_loss = float("inf")
num_bad_epochs = 0
best_state_dict = None

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float("inf"))
    num_bad_epochs = checkpoint.get('num_bad_epochs', 0)
    best_state_dict = checkpoint.get('best_state_dict', None)
    log(f"Resuming from checkpoint at epoch {start_epoch}")

# ---------------------------
# 5. Training Loop
# ---------------------------
for epoch in range(start_epoch, start_epoch + max_epochs):
    model.train()
    total_loss = 0.0
    n_samples = 0
    log(f"Epoch {epoch + 1}/{start_epoch + max_epochs}")

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

    log(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    # Early stopping: improvement relative to the best val loss so far
    if val_loss < best_val_loss * (1 - rtol):
        best_val_loss = val_loss
        num_bad_epochs = 0
        best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
    else:
        num_bad_epochs += 1

    log(f"  best_val_loss={best_val_loss:.6f}, num_bad_epochs={num_bad_epochs}/{patience}")

    # Checkpoint for resuming, including the best weights so far
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'num_bad_epochs': num_bad_epochs,
        'best_state_dict': best_state_dict,
    }, checkpoint_path)
    log(f"Checkpoint saved to: {checkpoint_path}")

    if num_bad_epochs >= patience:
        log(f"Converged: relative improvement below {rtol*100:.2f}% "
            f"for {patience} consecutive epoch(s). Stopping at epoch {epoch + 1}.")
        break
else:
    log(f"WARNING: reached max_epochs={max_epochs} without meeting the "
        f"convergence criterion (rtol={rtol*100:.2f}%, patience={patience}). "
        f"Using best checkpoint found (val_loss={best_val_loss:.6f}).")

log("Training complete")

# Restore the weights with the lowest val loss
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

# ---------------------------
# 6. Save Final Weights
# ---------------------------
# W is stored untransposed, shape (out_dim, in_dim): apply as x @ W.T + b
W = model.linear.weight.detach().cpu().numpy()
b = model.linear.bias.detach().cpu().numpy()
np.savez(model_save_path, W=W, b=b)
log(f"Final model weights saved to: {model_save_path}")

# ---------------------------
# 7. Cleanup
# ---------------------------
all_log_file.close()
train_loss_file.close()
val_loss_file.close()
