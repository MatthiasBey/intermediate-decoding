"""Training (solid) and validation loss (dashed) of the Aurora no_skip translators
(supplement), which upscale the latent states of the coarse stages to the final state.
Trained by methods_without_data/translator_training/aurora/train_no_skip_translators.py."""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # scripts/
import config
from utils.translator_logs import load_run, plot_runs_with_zoom

LOG_DIR = f"{config.TRANSLATOR_DIR}/aurora/no_skip/logs/"
STAGES = ["enc_1", "enc_2", "dec_0", "dec_1"]  # in network order

# losses[(stage index, block)] = (label, train, val), from train_loss_upscale_<stage>_<block>.txt
# with the initial loss in all_prints_upscale_<stage>_<block>.txt
losses = {}
for train_file in glob.glob(os.path.join(LOG_DIR, "train_loss_upscale_*.txt")):
    run_id = os.path.basename(train_file)[len("train_loss_"):-len(".txt")]
    m = re.fullmatch(r"upscale_(enc_1|enc_2|dec_0|dec_1)_(\d+)", run_id)
    val_file = os.path.join(LOG_DIR, f"val_loss_{run_id}.txt")
    if not m or not os.path.exists(val_file):
        continue
    stage, block = m.group(1), int(m.group(2))
    train, val = load_run(train_file, val_file, os.path.join(LOG_DIR, f"all_prints_{run_id}.txt"),
                          required=False)
    losses[(STAGES.index(stage), block)] = (f"{stage}_b{block}", train, val)

for i, stage in enumerate(STAGES):
    if not any(stage_index == i for stage_index, _ in losses):
        print(f"No training logs found for {stage}")

runs = [losses[key] for key in sorted(losses)]
plot_runs_with_zoom(
    runs, "Training loss of Aurora upscale-to-final translators (solid=train, dashed=val)",
    os.path.join(config.SUPPLEMENT_DIR, "no_skip_translator_training_loss_aurora.png"))
