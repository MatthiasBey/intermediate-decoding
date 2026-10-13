#!/bin/bash
# Packs data/ and results/paper/ into the archives that are published with the paper and fetched by
# download_data.sh. The archives keep the repository layout (paths start with data/ or results/), so
# they can be extracted in the repository folder. Each archive stays below 50 GB, the per-record
# limit of Zenodo.
#
# Usage (from anywhere):
#   bash scripts/utils/pack_data.sh [output folder]      # default: ../data_release next to the repository
#
# Writes the archives and md5sums.txt to the output folder. After uploading, copy the download links
# and the checksums from md5sums.txt into download_data.sh.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
OUT=$(readlink -f "${1:-../data_release}")
mkdir -p "$OUT"

pack () {   # pack <archive name> [--exclude=...] <path> [<path> ...]
    local name=$1; shift
    echo ">>> $name.tar"
    tar -cf "$OUT/$name.tar" --exclude=README.md "$@"
}

# results/paper/ without the tables, which run_paper_figures.sh writes itself
pack results_paper \
    --exclude=results/paper/translator_convergence \
    --exclude='results/paper/encoder_decoder_similarity/summary_*' \
    results/paper
pack data_base                  data/model_loss data/translators data/era5/constants
pack data_era5                  data/era5/era4aim_*.nc
pack data_latent_states         data/latent_states
pack data_decodings_graphcast   data/intermediate_decodings/tuned/graphcast
pack data_decodings_aurora      data/intermediate_decodings/tuned/aurora
pack data_decodings_other       data/intermediate_decodings/tuned/archesweather data/intermediate_decodings/no_skip

cd "$OUT"
md5sum ./*.tar | tee md5sums.txt
ls -l ./*.tar
