import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% Training and validation loss of the (skip) translators of GraphCast, ArchesWeather and Aurora
import glob
import re
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

# ---------------------------
# Paths
# ---------------------------
GC_LOG_DIR = f"{config.TRANSLATOR_DIR}/graphcast/logs"
ARCHES_LOG_DIR = f"{config.TRANSLATOR_DIR}/archesweather/skip/logs"
AURORA_LOG_DIR = f"{config.TRANSLATOR_DIR}/aurora/skip/logs"
OUTPUT_DIR = config.SUPPLEMENT_DIR

# ---------------------------
# Shared helpers
# ---------------------------
_FLOAT = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
INIT_TRAIN_PAT = re.compile(r"Initial train loss.*?:\s*" + _FLOAT)
INIT_VAL_PAT = re.compile(r"Initial val loss.*?:\s*" + _FLOAT)


def load_losses(path):
    """Read a loss file with one float per line (one line per epoch)."""
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line))
            except ValueError:
                continue
    return vals


def load_initial_losses(all_prints_path):
    """Return (initial_train_loss, initial_val_loss) from the 'Initial train/val loss'
    lines of an all_prints file, or (None, None) if the file does not exist."""
    if not os.path.exists(all_prints_path):
        return None, None
    init_train = init_val = None
    with open(all_prints_path) as f:
        for line in f:
            if init_train is None:
                m = INIT_TRAIN_PAT.search(line)
                if m:
                    init_train = float(m.group(1))
            if init_val is None:
                m = INIT_VAL_PAT.search(line)
                if m:
                    init_val = float(m.group(1))
            if init_train is not None and init_val is not None:
                break
    assert init_train is not None and init_val is not None, (
        f"{all_prints_path} exists but is missing an 'Initial train/val loss' line"
    )
    return init_train, init_val


def with_initial(vals, initial):
    return ([initial] + vals) if initial is not None else vals


def xtilde_label(idx):
    """Latent-state label X~_idx, as in plots/main/latent_state_development.py."""
    return rf"$\tilde{{X}}_{{{idx}}}$"


# offset of the global ArchesWeather latent-state index per layer
# (layer0 0, layer1 1-4, layer2 5-16, layer3 17-28, layer4 29-32)
ARCHES_LAYER_OFFSETS = {0: 0, 1: 1, 2: 5, 3: 17, 4: 29}


# ---------------------------
# GraphCast
# ---------------------------
def plot_graphcast_training_loss(
    log_dir,
    save_path,
    figsize=(12, 7),
    line_width=2,
    title_fontsize=18,
    label_fontsize=16,
    tick_fontsize=14,
    legend_fontsize=12,
    grid_alpha=0.5,
    grid_style="--",
):
    train_files = glob.glob(os.path.join(log_dir, "train_loss_*.txt"))
    run_ids = sorted(int(re.search(r"train_loss_(\d+)\.txt$", p).group(1)) for p in train_files)

    results = {}
    for run_id in run_ids:
        tfile = os.path.join(log_dir, f"train_loss_{run_id}.txt")
        vfile = os.path.join(log_dir, f"val_loss_{run_id}.txt")
        pfile = os.path.join(log_dir, f"all_prints_{run_id}.txt")
        if not os.path.exists(vfile):
            continue

        init_train, init_val = load_initial_losses(pfile)
        train_vals = with_initial(load_losses(tfile), init_train)
        val_vals = with_initial(load_losses(vfile), init_val)
        results[run_id] = {"train": train_vals, "val": val_vals}

    if not results:
        raise RuntimeError(f"No GraphCast runs found in {log_dir}")

    n_runs = len(results)
    cmap = plt.colormaps["viridis"].resampled(n_runs)

    plt.figure(figsize=figsize)
    for i, (run_id, data) in enumerate(sorted(results.items())):
        color = cmap(i / max(1, n_runs - 1))
        plt.plot(range(len(data["train"])), data["train"], color=color, linestyle="-",
                  linewidth=line_width, label=xtilde_label(run_id))
        plt.plot(range(len(data["val"])), data["val"], color=color, linestyle="--",
                  linewidth=line_width, label=None)

    plt.xlabel("Epoch", fontsize=label_fontsize)
    plt.ylabel("Loss", fontsize=label_fontsize)
    plt.title("Training loss of GraphCast translators", fontsize=title_fontsize)
    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    plt.grid(True, linestyle=grid_style, alpha=grid_alpha)
    plt.legend(ncol=2, fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved GraphCast plot to: {save_path}")


# ---------------------------
# ArchesWeather
# ---------------------------
def plot_arches_training_loss(
    log_dir,
    save_path,
    row_height=3.5,
    fig_width=14,
    line_width=2,
    title_fontsize=18,
    label_fontsize=16,
    tick_fontsize=14,
    legend_fontsize=10,
    grid_alpha=0.5,
    grid_style="--",
):
    train_files = glob.glob(os.path.join(log_dir, "train_loss_layer*_block*.txt"))

    parsed = {}
    for tfile in train_files:
        prefix = os.path.basename(tfile)[len("train_loss_"):-len(".txt")]
        m = re.match(r"layer(\d+)_block(\d+)$", prefix)
        if not m:
            continue
        layer, block = int(m.group(1)), int(m.group(2))

        vfile = os.path.join(log_dir, f"val_loss_{prefix}.txt")
        pfile = os.path.join(log_dir, f"all_prints_{prefix}.txt")
        if not os.path.exists(vfile):
            continue

        init_train, init_val = load_initial_losses(pfile)
        train_vals = with_initial(load_losses(tfile), init_train)
        val_vals = with_initial(load_losses(vfile), init_val)
        parsed[prefix] = {"layer": layer, "block": block, "train": train_vals, "val": val_vals}

    if not parsed:
        raise RuntimeError(f"No ArchesWeather runs found in {log_dir}")

    groups = [
        ("Layer 2 & 3", (2, 3)),
        ("Layer 1 & 4", (0, 1, 4)),
    ]
    fig, axes = plt.subplots(len(groups), 1, figsize=(fig_width, row_height * len(groups) * 1.6))
    if len(groups) == 1:
        axes = [axes]

    for ax, (title, layer_set) in zip(axes, groups):
        items = sorted(
            ((k, v) for k, v in parsed.items() if v["layer"] in layer_set),
            key=lambda kv: (kv[1]["layer"], kv[1]["block"]),
        )
        if not items:
            ax.text(0.5, 0.5, f"No runs for {title}", ha="center", va="center")
            ax.set_axis_off()
            continue

        n_runs = len(items)
        cmap = plt.colormaps["viridis"].resampled(n_runs)

        for i, (prefix, data) in enumerate(items):
            color = cmap(i / max(1, n_runs - 1))
            label = xtilde_label(ARCHES_LAYER_OFFSETS[data['layer']] + data['block'])
            ax.plot(range(len(data["train"])), data["train"], color=color, linestyle="-",
                     linewidth=line_width, label=label)
            ax.plot(range(len(data["val"])), data["val"], color=color, linestyle="--",
                     linewidth=line_width, label=None)

        ax.set_ylabel("Loss", fontsize=label_fontsize)
        ax.set_title(title, fontsize=title_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=legend_fontsize,
                  ncol=2 if n_runs > 12 else 1)

    axes[-1].set_xlabel("Epoch", fontsize=label_fontsize)
    plt.tight_layout()
    # suptitle just above the top axes
    top = max(a.get_tightbbox().transformed(fig.transFigure.inverted()).y1 for a in axes)
    fig.suptitle("Training loss of ArchesWeather translators", fontsize=title_fontsize, y=top + 0.01, va="bottom")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved ArchesWeather plot to: {save_path}")


# ---------------------------
# Aurora
# ---------------------------
def _aurora_stage_and_block(run_id):
    """Return (stage_name, block) of an Aurora translator run_id ('enc' or '0'-'46').
    Flat indices per stage: enc1 0-5, enc2 6-15, enc3 16-23, dec1 24-31, dec2 32-41, dec3 42-47."""
    if run_id == "enc":
        return "enc1", -1  # encoder output, shown with enc1
    model_id = int(run_id)
    ranges = [
        ("enc1", 0, 5), ("enc2", 6, 15), ("enc3", 16, 23),
        ("dec1", 24, 31), ("dec2", 32, 41), ("dec3", 42, 47),
    ]
    for stage_name, lo, hi in ranges:
        if lo <= model_id <= hi:
            return stage_name, model_id - lo
    raise ValueError(f"Unrecognized Aurora run_id: {run_id}")


def plot_aurora_training_loss(
    log_dir,
    save_path,
    row_height=3.0,
    fig_width=14,
    line_width=2,
    title_fontsize=18,
    label_fontsize=16,
    tick_fontsize=14,
    legend_fontsize=9,
    grid_alpha=0.5,
    grid_style="--",
):
    train_files = glob.glob(os.path.join(log_dir, "train_loss_*.txt"))

    parsed = {}
    for tfile in train_files:
        run_id = os.path.basename(tfile)[len("train_loss_"):-len(".txt")]
        if run_id in ("stage_0", "stage_1"):
            continue  # stage translators are not shown
        if run_id.startswith("upscale_"):
            continue  # upscale translators, see no_skip_translator_training_loss_aurora.py

        vfile = os.path.join(log_dir, f"val_loss_{run_id}.txt")
        pfile = os.path.join(log_dir, f"all_prints_{run_id}.txt")
        if not os.path.exists(vfile):
            continue

        stage_name, block = _aurora_stage_and_block(run_id)
        init_train, init_val = load_initial_losses(pfile)
        train_vals = with_initial(load_losses(tfile), init_train)
        val_vals = with_initial(load_losses(vfile), init_val)
        parsed[run_id] = {"stage": stage_name, "block": block, "train": train_vals, "val": val_vals}

    if not parsed:
        raise RuntimeError(f"No Aurora runs found in {log_dir}")

    groups = ["enc1", "enc2", "enc3", "dec1", "dec2", "dec3"]
    fig, axes = plt.subplots(len(groups), 1, figsize=(fig_width, row_height * len(groups)))

    for ax, stage_name in zip(axes, groups):
        items = sorted(
            ((k, v) for k, v in parsed.items() if v["stage"] == stage_name),
            key=lambda kv: kv[1]["block"],
        )
        if not items:
            ax.text(0.5, 0.5, f"No runs for {stage_name}", ha="center", va="center")
            ax.set_axis_off()
            continue

        n_runs = len(items)
        cmap = plt.colormaps["viridis"].resampled(n_runs)

        for i, (run_id, data) in enumerate(items):
            color = cmap(i / max(1, n_runs - 1))
            # global latent-state index: 0 for the encoder output, k + 1 for run_id k
            label = xtilde_label(0 if run_id == "enc" else int(run_id) + 1)
            ax.plot(range(len(data["train"])), data["train"], color=color, linestyle="-",
                     linewidth=line_width, label=label)
            ax.plot(range(len(data["val"])), data["val"], color=color, linestyle="--",
                     linewidth=line_width, label=None)

        ax.set_ylabel("Loss", fontsize=label_fontsize)
        ax.set_title(stage_name, fontsize=title_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=legend_fontsize,
                  ncol=2 if n_runs > 12 else 1)

    axes[-1].set_xlabel("Epoch", fontsize=label_fontsize)
    plt.tight_layout()
    # suptitle just above the top axes
    top = max(a.get_tightbbox().transformed(fig.transFigure.inverted()).y1 for a in axes)
    fig.suptitle("Training loss of Aurora translators", fontsize=title_fontsize, y=top + 0.01, va="bottom")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Aurora plot to: {save_path}")


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    plot_graphcast_training_loss(
        log_dir=GC_LOG_DIR,
        save_path=os.path.join(OUTPUT_DIR, "translator_training_loss_graphcast.png"),
    )

    plot_arches_training_loss(
        log_dir=ARCHES_LOG_DIR,
        save_path=os.path.join(OUTPUT_DIR, "translator_training_loss_arches.png"),
    )

    plot_aurora_training_loss(
        log_dir=AURORA_LOG_DIR,
        save_path=os.path.join(OUTPUT_DIR, "translator_training_loss_aurora.png"),
    )
