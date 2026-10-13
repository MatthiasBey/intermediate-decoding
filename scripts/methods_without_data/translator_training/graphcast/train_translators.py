import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %%
import psutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime

# %% Configuration

# model_id: processor step whose latent (step_<model_id>) is mapped to step_16
# current_version: label that is only printed
model_id = int(sys.argv[1])
current_version = sys.argv[2]

print(f"Using model_id = {model_id}, version = {current_version}")

data_dir = f"{config.LATENT_DIR}/graphcast/"

log_dir = f"{config.TRANSLATOR_DIR}/graphcast/logs/"

os.makedirs(log_dir, exist_ok=True)

checkpoint_path = f"{config.TRANSLATOR_DIR}/graphcast/weights/translator_checkpoint_{model_id}_gnn.pt"
model_save_path = f"{config.TRANSLATOR_DIR}/graphcast/weights/translator_matrix_{model_id}_gnn.npz"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)


learning_rate = 5e-4
epochs = 20  # fixed number of epochs, no early stopping
batch_size = 2
val_fraction = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    time_str = dt.strftime("%Y-%m-%d-%H-%M")
    filename = f"all_latents_{time_str}.npz"
    path = os.path.join(data_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # mmap avoids copying large arrays
    with np.load(path, mmap_mode="r") as data:
        x = data[f"step_{step}"]
        y = data["step_16"]

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

        # float16 on disk, train in float32; drop the singleton dim 1
        x = x.astype(np.float32, copy=False).squeeze(1)
        y = y.astype(np.float32, copy=False).squeeze(1)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        return x, y
    
# %% Timestamp discovery

log_print("Scanning data directory for timestamps...")

timestamps = sorted([
    datetime.strptime(f[12:-4], "%Y-%m-%d-%H-%M")
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
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        return self.linear(x)

model = LinearTranslator(dim=512).to(device)

# Identity init (not when resuming from a checkpoint)
if not os.path.exists(checkpoint_path):
    with torch.no_grad():
        model.linear.weight.copy_(torch.eye(512))
        model.linear.bias.zero_()
    log_print("Initialized with identity weights")

# %% Optimizer, scheduler and checkpoint

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# halve the learning rate after 5 epochs without improvement of the val loss
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5)

start_epoch = 0

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    log_print(f"Resumed from epoch {start_epoch}")

# %% Initial loss
model.eval()
initial_train_loss = 0.0
with torch.no_grad():
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        initial_train_loss += loss.item() * xb.size(0)

initial_train_loss /= train_size
log_print(f"Initial train loss (identity weights): {initial_train_loss:.6f}")

initial_val_loss = 0.0
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        initial_val_loss += loss.item() * xb.size(0)

initial_val_loss /= val_size
log_print(f"Initial val loss (identity weights): {initial_val_loss:.6f}")


# %% Training loop

for epoch in range(start_epoch, start_epoch + epochs):
    model.train()
    train_loss = 0.0

    print_ram_usage(f"Before epoch {epoch+1}")

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

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
            xb = xb.to(device)
            yb = yb.to(device)

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

    scheduler.step(val_loss)

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    print_ram_usage(f"After epoch {epoch+1}")

log_print("Training complete.")

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
