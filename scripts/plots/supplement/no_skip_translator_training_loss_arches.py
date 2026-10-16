"""Training (solid) and validation loss (dashed) of the ArchesWeather no_skip translators
(supplement): the skip translators of layers 0, 1 and 4, which already map to the final
latent state, and the expand translators of layers 2 and 3."""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
from utils.translator_logs import load_run, plot_runs_with_zoom

SKIP_LOG_DIR = f"{config.TRANSLATOR_DIR}/archesweather/skip/logs"
EXPAND_LOG_DIR = f"{config.TRANSLATOR_DIR}/archesweather/no_skip/logs"

# losses[(layer, block)] = (train, val)
losses = {}

# skip translators: train_loss_layer<L>_block<B>.txt, initial loss in all_prints_*.txt
# (layer 5 is the stage translator, which is not shown)
for train_file in glob.glob(os.path.join(SKIP_LOG_DIR, "train_loss_layer*_block*.txt")):
    run_id = os.path.basename(train_file)[len("train_loss_"):-len(".txt")]
    m = re.fullmatch(r"layer(\d+)_block(\d+)", run_id)
    val_file = os.path.join(SKIP_LOG_DIR, f"val_loss_{run_id}.txt")
    if not m or m.group(1) == "5" or not os.path.exists(val_file):
        continue
    losses[(int(m.group(1)), int(m.group(2)))] = load_run(
        train_file, val_file, os.path.join(SKIP_LOG_DIR, f"all_prints_{run_id}.txt"))

# expand translators of layers 2 and 3: layer<L>_block<B>_train_loss.txt, the initial loss
# is already the first value. They replace the skip translators of these layers.
for train_file in glob.glob(os.path.join(EXPAND_LOG_DIR, "layer*_block*_train_loss.txt")):
    run_id = os.path.basename(train_file)[:-len("_train_loss.txt")]
    m = re.fullmatch(r"layer(\d+)_block(\d+)", run_id)
    val_file = os.path.join(EXPAND_LOG_DIR, f"{run_id}_val_loss.txt")
    if not m or not os.path.exists(val_file):
        continue
    losses[(int(m.group(1)), int(m.group(2)))] = load_run(train_file, val_file)

runs = [(f"L{layer}_B{block}", train, val) for (layer, block), (train, val) in sorted(losses.items())]
plot_runs_with_zoom(
    runs, "Training loss of ArchesWeather translators (solid=train, dashed=val), no skip logic",
    os.path.join(config.SUPPLEMENT_DIR, "no_skip_translator_training_loss_arches.png"))
