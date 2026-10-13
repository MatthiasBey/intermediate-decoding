import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% Training and validation loss of the Aurora no_skip (upscale) translators
import glob
import re
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

# ---------------------------
# Paths
# ---------------------------
AURORA_LOG_DIR = f"{config.TRANSLATOR_DIR}/aurora/no_skip/logs/"
OUTPUT_DIR = config.SUPPLEMENT_DIR

STAGE_ORDER = {"enc_1": 0, "enc_2": 1, "dec_0": 2, "dec_1": 3}

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
    lines of an all_prints file; None for values that are not found."""
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
    return init_train, init_val


def with_initial(vals, initial):
    return ([initial] + vals) if initial is not None else vals


# ---------------------------
# Aurora upscale translator runs (methods_without_data/translator_training/aurora/train_no_skip_translators.py)
# ---------------------------
def load_upscale_format_runs(log_dir):
    """Files train_loss_upscale_{stage}_{block}.txt; the initial loss is read from
    all_prints_upscale_{stage}_{block}.txt and prepended."""
    train_files = glob.glob(os.path.join(log_dir, "train_loss_upscale_*.txt"))

    parsed = {}
    for tfile in train_files:
        prefix = os.path.basename(tfile)[len("train_loss_"):-len(".txt")]  # e.g. "upscale_enc_1_0"
        m = re.match(r"upscale_(enc_1|enc_2|dec_0|dec_1)_(\d+)$", prefix)
        if not m:
            continue
        stage_name, block = m.group(1), int(m.group(2))

        vfile = os.path.join(log_dir, f"val_loss_{prefix}.txt")
        pfile = os.path.join(log_dir, f"all_prints_{prefix}.txt")
        if not os.path.exists(vfile):
            continue

        init_train, init_val = load_initial_losses(pfile)
        train_vals = with_initial(load_losses(tfile), init_train)
        val_vals = with_initial(load_losses(vfile), init_val)
        parsed[f"{stage_name}_b{block}"] = {
            "stage_order": STAGE_ORDER[stage_name],
            "block": block,
            "train": train_vals,
            "val": val_vals,
        }

    return parsed


# ---------------------------
# Plot
# ---------------------------
def plot_aurora_upscale_training_loss(
    log_dir,
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
    parsed = load_upscale_format_runs(log_dir)

    for stage_name, order in STAGE_ORDER.items():
        if not any(v["stage_order"] == order for v in parsed.values()):
            print(f"No training logs found for {stage_name}")

    if not parsed:
        raise RuntimeError(f"No Aurora upscale-translator runs found in {log_dir}")

    items = sorted(parsed.items(), key=lambda kv: (kv[1]["stage_order"], kv[1]["block"]))
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
    ax_full.set_title("Training loss of Aurora upscale-to-final translators (solid=train, dashed=val)", fontsize=title_fontsize)
    ax_full.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=legend_fontsize, ncol=2)

    ax_zoom.set_ylim(0, zoom_ymax)
    ax_zoom.set_xlabel("Epoch", fontsize=label_fontsize)
    ax_zoom.set_ylabel(f"Loss (zoomed, 0-{zoom_ymax})", fontsize=label_fontsize)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Aurora upscale-translator plot to: {save_path}")


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    plot_aurora_upscale_training_loss(
        log_dir=AURORA_LOG_DIR,
        save_path=os.path.join(OUTPUT_DIR, "no_skip_translator_training_loss_aurora.png"),
    )
