import os, sys  # find scripts/ (folder containing config.py), put it on sys.path, import path config
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_SCRIPTS_DIR, "config.py")):
    _SCRIPTS_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
import config
# %% Convergence epochs of the (skip) translator training runs, from the validation loss logs
import re
import glob
import csv

# ---------------------------
# Paths
# ---------------------------
GC_LOG_DIR = f"{config.TRANSLATOR_DIR}/graphcast/logs"
ARCHES_LOG_DIR = f"{config.TRANSLATOR_DIR}/archesweather/skip/logs"
AURORA_LOG_DIR = f"{config.TRANSLATOR_DIR}/aurora/skip/logs"
OUTPUT_DIR = f"{config.RESULTS_DIR}/translator_convergence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLDS = [0.01, 0.001]  # 1% and 0.1%

# ---------------------------
# Helpers
# ---------------------------
def load_losses(path):
    """One float per line, one line per epoch."""
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


def epoch_at_threshold(val_losses, threshold):
    """First epoch (1-indexed) at which the validation loss improves on the best value
    so far by less than `threshold` (relative), as in the convergence check of the
    training scripts. None if this does not happen within the recorded epochs."""
    best = float("inf")
    for epoch_num, val in enumerate(val_losses, start=1):
        if val < best * (1 - threshold):
            best = val
        else:
            return epoch_num
    return None


def analyze_run(val_loss_path):
    val_losses = load_losses(val_loss_path)
    total_epochs = len(val_losses)
    results = {}
    for t in THRESHOLDS:
        results[t] = epoch_at_threshold(val_losses, t)
    return total_epochs, results


# ---------------------------
# GraphCast
# ---------------------------
def collect_graphcast(log_dir):
    rows = []
    val_files = glob.glob(os.path.join(log_dir, "val_loss_*.txt"))
    run_ids = sorted(int(re.search(r"val_loss_(\d+)\.txt$", p).group(1)) for p in val_files)
    for run_id in run_ids:
        vfile = os.path.join(log_dir, f"val_loss_{run_id}.txt")
        total_epochs, res = analyze_run(vfile)
        rows.append({"label": f"h{run_id}", "total_epochs": total_epochs,
                      "epoch_at_1pct": res[0.01], "epoch_at_0.1pct": res[0.001]})
    return rows


# ---------------------------
# ArchesWeather
# ---------------------------
def collect_archesweather(log_dir):
    rows = []
    val_files = glob.glob(os.path.join(log_dir, "val_loss_layer*_block*.txt"))
    entries = []
    for vfile in val_files:
        prefix = os.path.basename(vfile)[len("val_loss_"):-len(".txt")]
        m = re.match(r"layer(\d+)_block(\d+)$", prefix)
        if not m:
            continue
        layer, block = int(m.group(1)), int(m.group(2))
        entries.append((layer, block, vfile))
    entries.sort()

    for layer, block, vfile in entries:
        total_epochs, res = analyze_run(vfile)
        label = "stage translator" if layer == 5 else f"L{layer}_B{block}"
        rows.append({"label": label, "total_epochs": total_epochs,
                      "epoch_at_1pct": res[0.01], "epoch_at_0.1pct": res[0.001]})
    return rows


# ---------------------------
# Aurora
# ---------------------------
# Aurora translator ids (argv[1] of methods_without_data/translator_training/aurora/
# train_translators_*.py) per stage. The last blocks of dec1, dec2 and dec3 (ids 31, 41, 47)
# are translator targets and have no translator.
AURORA_STAGES = [
    # (label_prefix, first_id, num_blocks, number of blocks with a translator)
    ("enc1", 0, 6, 6),
    ("enc2", 6, 10, 10),
    ("enc3", 16, 8, 8),
    ("dec1", 24, 8, 7),
    ("dec2", 32, 10, 9),
    ("dec3", 42, 6, 5),
]

# The validation loss logs of enc1 blocks 0 and 1 were overwritten by the stage translator
# runs (train_stage_translator_decoder_stage{1,2}.py, logs val_loss_stage_{0,1}.txt) and are
# not available; the trained weights are not affected.
AURORA_LOST_IDS = {0, 1}
AURORA_LOST_LABELS = ["enc1_b0", "enc1_b1"]
AURORA_STAGE_TRANSLATOR_FILES = {
    "stage_translator_1": "val_loss_stage_0.txt",
    "stage_translator_2": "val_loss_stage_1.txt",
}


def build_aurora_id_labels():
    """Return the label of every Aurora translator id, the set of all ids and the
    set of ids without a translator."""
    id_to_label = {}
    all_ids = set()
    unscheduled_ids = set()
    for prefix, first_id, num_blocks, num_scheduled in AURORA_STAGES:
        for b in range(num_blocks):
            i = first_id + b
            all_ids.add(i)
            id_to_label[i] = f"{prefix}_b{b}"
            if b >= num_scheduled:
                unscheduled_ids.add(i)
    return id_to_label, all_ids, unscheduled_ids


def collect_aurora(log_dir):
    id_to_label, all_ids, unscheduled_ids = build_aurora_id_labels()

    rows = []
    found_ids = set()
    for run_id in sorted(all_ids):
        vfile = os.path.join(log_dir, f"val_loss_{run_id}.txt")
        if not os.path.exists(vfile):
            continue
        found_ids.add(run_id)
        total_epochs, res = analyze_run(vfile)
        rows.append({"label": id_to_label[run_id], "total_epochs": total_epochs,
                      "epoch_at_1pct": res[0.01], "epoch_at_0.1pct": res[0.001]})

    # encoder output translator and the two stage translators (not part of the id scheme)
    extra_files = {"enc": "val_loss_enc.txt", **AURORA_STAGE_TRANSLATOR_FILES}
    for label, fname in extra_files.items():
        fpath = os.path.join(log_dir, fname)
        if os.path.exists(fpath):
            total_epochs, res = analyze_run(fpath)
            rows.append({"label": label, "total_epochs": total_epochs,
                          "epoch_at_1pct": res[0.01], "epoch_at_0.1pct": res[0.001]})

    # report missing logs
    pending = sorted((all_ids - unscheduled_ids - AURORA_LOST_IDS) - found_ids)
    if pending:
        print(f"Aurora: no training log found for {[id_to_label[i] for i in pending]}")
    if unscheduled_ids:
        print(f"Aurora: no translator for {[id_to_label[i] for i in sorted(unscheduled_ids)]}")
    print(f"Aurora: training log not available for {AURORA_LOST_LABELS}")

    return rows


# ---------------------------
# Report
# ---------------------------
def print_and_save(title, rows, csv_path):
    print(f"\n{title}")
    print(f"{'run':<20}{'epoch @1% improv.':<20}{'epoch @0.1% improv.':<22}{'total epochs':<12}")
    for r in rows:
        e1 = r["epoch_at_1pct"] if r["epoch_at_1pct"] is not None else "not reached"
        e01 = r["epoch_at_0.1pct"] if r["epoch_at_0.1pct"] is not None else "not reached"
        print(f"{r['label']:<20}{str(e1):<20}{str(e01):<22}{r['total_epochs']:<12}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "epoch_at_1pct", "epoch_at_0.1pct", "total_epochs"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    gc_rows = collect_graphcast(GC_LOG_DIR)
    print_and_save("GraphCast translators", gc_rows, os.path.join(OUTPUT_DIR, "graphcast_convergence_epochs.csv"))

    arches_rows = collect_archesweather(ARCHES_LOG_DIR)
    print_and_save("ArchesWeather translators", arches_rows, os.path.join(OUTPUT_DIR, "arches_convergence_epochs.csv"))

    aurora_rows = collect_aurora(AURORA_LOG_DIR)
    print_and_save("Aurora translators", aurora_rows, os.path.join(OUTPUT_DIR, "aurora_convergence_epochs.csv"))
