"""Table of the convergence epochs of the skip translators (supplement).

For every training run: the first epoch at which the validation loss improves on the best
value so far by less than 1 % and 0.1 %, and the number of trained epochs. Printed and
saved as one csv file per model to results/<mode>/translator_convergence/.
"""
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
from utils.translator_logs import load_losses

GC_LOG_DIR = f"{config.TRANSLATOR_DIR}/graphcast/logs"
ARCHES_LOG_DIR = f"{config.TRANSLATOR_DIR}/archesweather/skip/logs"
AURORA_LOG_DIR = f"{config.TRANSLATOR_DIR}/aurora/skip/logs"
OUTPUT_DIR = f"{config.RESULTS_DIR}/translator_convergence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aurora translator ids (argv[1] of methods_without_data/translator_training/aurora/
# train_translators_*.py) per stage: (stage, first id, number of blocks, blocks with a
# translator). The last blocks of dec1, dec2 and dec3 (ids 31, 41, 47) are translator
# targets and have no translator.
AURORA_STAGES = [
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
# encoder output translator and the two stage translators, which have no translator id
AURORA_EXTRA_LOGS = {
    "enc": "val_loss_enc.txt",
    "stage_translator_1": "val_loss_stage_0.txt",
    "stage_translator_2": "val_loss_stage_1.txt",
}


def convergence_epoch(val_losses, threshold):
    """First epoch (counted from 1) at which the validation loss improves on the best value
    so far by less than `threshold` (relative), as in the convergence check of the training
    scripts. None if this does not happen within the recorded epochs."""
    best = float("inf")
    for epoch, val in enumerate(val_losses, start=1):
        if val < best * (1 - threshold):
            best = val
        else:
            return epoch
    return None


def table_row(label, val_loss_path):
    val_losses = load_losses(val_loss_path)
    return {"label": label, "total_epochs": len(val_losses),
            "epoch_at_1pct": convergence_epoch(val_losses, 0.01),
            "epoch_at_0.1pct": convergence_epoch(val_losses, 0.001)}


def graphcast_rows():
    files = glob.glob(os.path.join(GC_LOG_DIR, "val_loss_*.txt"))
    run_ids = sorted(int(re.search(r"val_loss_(\d+)\.txt$", path).group(1)) for path in files)
    return [table_row(f"h{run_id}", os.path.join(GC_LOG_DIR, f"val_loss_{run_id}.txt")) for run_id in run_ids]


def arches_rows():
    runs = []
    for path in glob.glob(os.path.join(ARCHES_LOG_DIR, "val_loss_layer*_block*.txt")):
        m = re.fullmatch(r"val_loss_layer(\d+)_block(\d+)\.txt", os.path.basename(path))
        if m:
            runs.append((int(m.group(1)), int(m.group(2)), path))
    return [table_row("stage translator" if layer == 5 else f"L{layer}_B{block}", path)
            for layer, block, path in sorted(runs)]


def aurora_rows():
    id_to_label = {}
    no_translator_ids = set()
    for stage, first_id, n_blocks, n_translators in AURORA_STAGES:
        for block in range(n_blocks):
            id_to_label[first_id + block] = f"{stage}_b{block}"
            if block >= n_translators:
                no_translator_ids.add(first_id + block)

    rows = []
    found_ids = set()
    for run_id in sorted(id_to_label):
        path = os.path.join(AURORA_LOG_DIR, f"val_loss_{run_id}.txt")
        if os.path.exists(path):
            found_ids.add(run_id)
            rows.append(table_row(id_to_label[run_id], path))
    for label, fname in AURORA_EXTRA_LOGS.items():
        path = os.path.join(AURORA_LOG_DIR, fname)
        if os.path.exists(path):
            rows.append(table_row(label, path))

    missing = sorted(set(id_to_label) - no_translator_ids - AURORA_LOST_IDS - found_ids)
    if missing:
        print(f"Aurora: no training log found for {[id_to_label[i] for i in missing]}")
    print(f"Aurora: no translator for {[id_to_label[i] for i in sorted(no_translator_ids)]}")
    print(f"Aurora: training log not available for {[id_to_label[i] for i in sorted(AURORA_LOST_IDS)]}")
    return rows


def print_and_save(title, rows, csv_path):
    print(f"\n{title}")
    print(f"{'run':<20}{'epoch @1% improv.':<20}{'epoch @0.1% improv.':<22}{'total epochs':<12}")
    for row in rows:
        e1 = row["epoch_at_1pct"] if row["epoch_at_1pct"] is not None else "not reached"
        e01 = row["epoch_at_0.1pct"] if row["epoch_at_0.1pct"] is not None else "not reached"
        print(f"{row['label']:<20}{str(e1):<20}{str(e01):<22}{row['total_epochs']:<12}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "epoch_at_1pct", "epoch_at_0.1pct", "total_epochs"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")


print_and_save("GraphCast translators", graphcast_rows(),
               os.path.join(OUTPUT_DIR, "graphcast_convergence_epochs.csv"))
print_and_save("ArchesWeather translators", arches_rows(),
               os.path.join(OUTPUT_DIR, "arches_convergence_epochs.csv"))
print_and_save("Aurora translators", aurora_rows(),
               os.path.join(OUTPUT_DIR, "aurora_convergence_epochs.csv"))
