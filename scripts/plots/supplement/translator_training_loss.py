"""Training (solid) and validation loss (dashed) of the skip translators of GraphCast,
ArchesWeather and Aurora, one figure per model (supplement)."""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
from utils.translator_logs import load_run, plot_runs

GC_LOG_DIR = f"{config.TRANSLATOR_DIR}/graphcast/logs"
ARCHES_LOG_DIR = f"{config.TRANSLATOR_DIR}/archesweather/skip/logs"
AURORA_LOG_DIR = f"{config.TRANSLATOR_DIR}/aurora/skip/logs"

TITLE_SIZE = 18
LABEL_SIZE = 16
TICK_SIZE = 14

# first latent-state index of each ArchesWeather layer
# (layer0 0, layer1 1-4, layer2 5-16, layer3 17-28, layer4 29-32)
ARCHES_LAYER_OFFSETS = {0: 0, 1: 1, 2: 5, 3: 17, 4: 29}

# Aurora translator ids per stage: enc1 0-5, enc2 6-15, enc3 16-23, dec1 24-31, dec2 32-41,
# dec3 42-47 ('enc' is the encoder output, shown with enc1)
AURORA_STAGE_IDS = {"enc1": (0, 5), "enc2": (6, 15), "enc3": (16, 23),
                    "dec1": (24, 31), "dec2": (32, 41), "dec3": (42, 47)}


def state_label(index):
    """Latent-state label as in plots/main/latent_state_development.py."""
    return rf"$\tilde{{X}}_{{{index}}}$"


def load_logs(log_dir, run_id):
    return load_run(os.path.join(log_dir, f"train_loss_{run_id}.txt"),
                    os.path.join(log_dir, f"val_loss_{run_id}.txt"),
                    os.path.join(log_dir, f"all_prints_{run_id}.txt"))


def run_ids_with_val_loss(log_dir, pattern):
    """Run ids of the files train_loss_<id>.txt that also have a validation loss file."""
    run_ids = []
    for path in glob.glob(os.path.join(log_dir, "train_loss_*.txt")):
        run_id = os.path.basename(path)[len("train_loss_"):-len(".txt")]
        if re.fullmatch(pattern, run_id) and os.path.exists(os.path.join(log_dir, f"val_loss_{run_id}.txt")):
            run_ids.append(run_id)
    return run_ids


def finish_stacked_figure(fig, axes, title, save_path):
    """Epoch label, layout and a title just above the top panel, then save."""
    axes[-1].set_xlabel("Epoch", fontsize=LABEL_SIZE)
    plt.tight_layout()
    top = max(ax.get_tightbbox().transformed(fig.transFigure.inverted()).y1 for ax in axes)
    fig.suptitle(title, fontsize=TITLE_SIZE, y=top + 0.01, va="bottom")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def style_panel(ax, title, n_runs, legend_size):
    ax.set_ylabel("Loss", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=legend_size,
              ncol=2 if n_runs > 12 else 1)


# %% GraphCast: one translator per processor step
run_ids = sorted(int(run_id) for run_id in run_ids_with_val_loss(GC_LOG_DIR, r"\d+"))
runs = [(state_label(run_id), *load_logs(GC_LOG_DIR, run_id)) for run_id in run_ids]

fig, ax = plt.subplots(figsize=(12, 7))
plot_runs(ax, runs)
ax.set_xlabel("Epoch", fontsize=LABEL_SIZE)
ax.set_ylabel("Loss", fontsize=LABEL_SIZE)
ax.set_title("Training loss of GraphCast translators", fontsize=TITLE_SIZE)
ax.tick_params(axis="both", labelsize=TICK_SIZE)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(ncol=2, fontsize=12)
plt.tight_layout()
save_path = os.path.join(config.SUPPLEMENT_DIR, "translator_training_loss_graphcast.png")
plt.savefig(save_path, dpi=200)
plt.close()
print(f"Saved {save_path}")

# %% ArchesWeather: one panel for the coarse layers 2 and 3, one for layers 0, 1 and 4
# (run ids "layer<L>_block<B>"; layer 5 is the stage translator, which is not shown)
arches_runs = {}
for run_id in run_ids_with_val_loss(ARCHES_LOG_DIR, r"layer\d+_block\d+"):
    layer, block = map(int, re.findall(r"\d+", run_id))
    arches_runs[(layer, block)] = load_logs(ARCHES_LOG_DIR, run_id)

panels = [("Layer 2 & 3", (2, 3)), ("Layer 1 & 4", (0, 1, 4))]
fig, axes = plt.subplots(len(panels), 1, figsize=(14, 3.5 * len(panels) * 1.6))
for ax, (title, layers) in zip(axes, panels):
    runs = [(state_label(ARCHES_LAYER_OFFSETS[layer] + block), *losses)
            for (layer, block), losses in sorted(arches_runs.items()) if layer in layers]
    plot_runs(ax, runs)
    style_panel(ax, title, len(runs), legend_size=10)
finish_stacked_figure(fig, axes, "Training loss of ArchesWeather translators",
                      os.path.join(config.SUPPLEMENT_DIR, "translator_training_loss_arches.png"))

# %% Aurora: one panel per stage (run ids 'enc' and 0-46; the stage translators 'stage_0',
# 'stage_1' and the no_skip translators 'upscale_*' are not shown)
aurora_runs = {stage: [] for stage in AURORA_STAGE_IDS}
for run_id in run_ids_with_val_loss(AURORA_LOG_DIR, r"enc|\d+"):
    if run_id == "enc":
        stage, block, state = "enc1", -1, 0
    else:
        stage = next(s for s, (first, last) in AURORA_STAGE_IDS.items() if first <= int(run_id) <= last)
        block, state = int(run_id) - AURORA_STAGE_IDS[stage][0], int(run_id) + 1
    aurora_runs[stage].append((block, state_label(state), *load_logs(AURORA_LOG_DIR, run_id)))

fig, axes = plt.subplots(len(aurora_runs), 1, figsize=(14, 3.0 * len(aurora_runs)))
for ax, (stage, stage_runs) in zip(axes, aurora_runs.items()):
    runs = [(label, train, val) for block, label, train, val in sorted(stage_runs, key=lambda run: run[0])]
    plot_runs(ax, runs)
    style_panel(ax, stage, len(runs), legend_size=9)
finish_stacked_figure(fig, axes, "Training loss of Aurora translators",
                      os.path.join(config.SUPPLEMENT_DIR, "translator_training_loss_aurora.png"))
