import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% Training and validation loss of the ArchesWeather no_skip translators
# (skip translators for layer0/1/4, expand translators for layer2/3)
import glob
import re
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

# ---------------------------
# Paths
# ---------------------------
ARCHES_LOG_DIR = f"{config.TRANSLATOR_DIR}/archesweather/skip/logs"
# layer2/3 expand translators (with 2x2 upsampling, target layer4 block3 like layer0/1/4)
ARCHES_EXPAND_LOG_DIR = f"{config.TRANSLATOR_DIR}/archesweather/no_skip/logs"
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


# ---------------------------
# ArchesWeather - skip translator runs (layer5 = stage translator is excluded)
# ---------------------------
def load_old_format_runs(log_dir):
    """Files train_loss_layer{L}_block{B}.txt; the initial loss is read from
    all_prints_layer{L}_block{B}.txt and prepended."""
    train_files = glob.glob(os.path.join(log_dir, "train_loss_layer*_block*.txt"))

    parsed = {}
    for tfile in train_files:
        prefix = os.path.basename(tfile)[len("train_loss_"):-len(".txt")]
        m = re.match(r"layer(\d+)_block(\d+)$", prefix)
        if not m:
            continue
        layer, block = int(m.group(1)), int(m.group(2))
        if layer == 5:  # stage translator
            continue

        vfile = os.path.join(log_dir, f"val_loss_{prefix}.txt")
        pfile = os.path.join(log_dir, f"all_prints_{prefix}.txt")
        if not os.path.exists(vfile):
            continue

        init_train, init_val = load_initial_losses(pfile)
        train_vals = with_initial(load_losses(tfile), init_train)
        val_vals = with_initial(load_losses(vfile), init_val)
        parsed[f"L{layer}_B{block}"] = {"layer": layer, "block": block, "train": train_vals, "val": val_vals}

    return parsed


# ---------------------------
# ArchesWeather - "expand" translator runs (layer2/3 only)
# ---------------------------
def load_expand_format_runs(log_dir):
    """Files layer{L}_block{B}_train_loss.txt; the initial loss is already the first line."""
    train_files = glob.glob(os.path.join(log_dir, "layer*_block*_train_loss.txt"))

    parsed = {}
    for tfile in train_files:
        prefix = os.path.basename(tfile)[:-len("_train_loss.txt")]
        m = re.match(r"layer(\d+)_block(\d+)$", prefix)
        if not m:
            continue
        layer, block = int(m.group(1)), int(m.group(2))

        vfile = os.path.join(log_dir, f"{prefix}_val_loss.txt")
        if not os.path.exists(vfile):
            continue

        train_vals = load_losses(tfile)
        val_vals = load_losses(vfile)
        parsed[f"L{layer}_B{block}"] = {"layer": layer, "block": block, "train": train_vals, "val": val_vals}

    return parsed


# ---------------------------
# ArchesWeather - combined plot
# ---------------------------
def plot_arches_training_loss(
    old_log_dir,
    expand_log_dir,
    save_path,
    figsize=(16, 12),
    line_width=1.2,
    title_fontsize=18,
    label_fontsize=16,
    tick_fontsize=14,
    legend_fontsize=9,
    grid_alpha=0.5,
    grid_style="--",
    zoom_ymax=0.4,
):
    parsed = {}
    parsed.update(load_old_format_runs(old_log_dir))
    # the expand runs replace the layer2/3 skip runs (same keys)
    parsed.update(load_expand_format_runs(expand_log_dir))

    if not parsed:
        raise RuntimeError(f"No ArchesWeather runs found in {old_log_dir} or {expand_log_dir}")

    items = sorted(parsed.items(), key=lambda kv: (kv[1]["layer"], kv[1]["block"]))
    n_runs = len(items)
    cmap = plt.colormaps["turbo"].resampled(n_runs)

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                            gridspec_kw={"height_ratios": [2, 1]})

    for ax in (ax_full, ax_zoom):
        for i, (label, data) in enumerate(items):
            color = cmap(i / max(1, n_runs - 1))
            ax.plot(range(len(data["train"])), data["train"], color=color, linestyle="-",
                     linewidth=line_width, label=label if ax is ax_full else None)
            ax.plot(range(len(data["val"])), data["val"], color=color, linestyle="--",
                     linewidth=line_width, label=None)
        ax.grid(True, linestyle=grid_style, alpha=grid_alpha)
        ax.tick_params(axis="both", labelsize=tick_fontsize)

    ax_full.set_ylabel("Loss", fontsize=label_fontsize)
    ax_full.set_title("Training loss of ArchesWeather translators (solid=train, dashed=val), no skip logic", fontsize=title_fontsize)
    ax_full.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=legend_fontsize, ncol=2)

    ax_zoom.set_ylim(0, zoom_ymax)
    ax_zoom.set_xlabel("Epoch", fontsize=label_fontsize)
    ax_zoom.set_ylabel(f"Loss (zoomed, 0-{zoom_ymax})", fontsize=label_fontsize)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved ArchesWeather plot to: {save_path}")


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    plot_arches_training_loss(
        old_log_dir=ARCHES_LOG_DIR,
        expand_log_dir=ARCHES_EXPAND_LOG_DIR,
        save_path=os.path.join(OUTPUT_DIR, "no_skip_translator_training_loss_arches.png"),
    )
