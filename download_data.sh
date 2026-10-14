#!/bin/bash
# Downloads the published data and extracts it into data/ and results/paper/. The data are
# published on Zenodo in three parts: doi:10.5281/zenodo.23345097, doi:10.5281/zenodo.23345099
# and doi:10.5281/zenodo.23345101.
#
# Usage:
#   bash download_data.sh results    # ~4.5 GB: everything except the four example-forecast figures
#                                    #   below and the single-run pipelines
#   bash download_data.sh all        # ~112 GB: everything
#   bash download_data.sh <archive> [<archive> ...]   # single archives, see ARCHIVES below
#
# "results" is enough for all figures averaged over the 48 forecasts, the covariance figure, the
# increment maps, the translator training-loss figures and both tables. The figures
# latent_state_development, encoder_decoder, tuned_decoding_comparison_* and latent_channel_*, and
# the single-run pipelines, need "all".
#
# Archives are downloaded to downloads/ (set DOWNLOAD_DIR to change this), checked against their
# md5 checksum, extracted and then deleted (set KEEP_ARCHIVES=1 to keep them). Archives that were
# extracted before are skipped. data/ is extracted to ID_DATA_DIR if that is set.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

#  name                       group     md5                               download link
ARCHIVES="
results_paper                 results   d3d4f8426cf099812d54934b833ce6b3  https://zenodo.org/records/23345097/files/results_paper.tar?download=1
data_base                     results   45bc59bc12eed68d507ed6a2af3fa0da  https://zenodo.org/records/23345097/files/data_base.tar?download=1
data_era5                     all       daa1bcf311883c9db3df324837ef2346  https://zenodo.org/records/23345097/files/data_era5.tar?download=1
data_latent_states            all       4f2a0a4e81d60eef23fbb06da42a9103  https://zenodo.org/records/23345099/files/data_latent_states.tar?download=1
data_decodings_graphcast      all       e230e89135b1a24f48c81a3341fc3779  https://zenodo.org/records/23345099/files/data_decodings_graphcast.tar?download=1
data_decodings_aurora         all       b1d75e84338cda501fad5836be03b978  https://zenodo.org/records/23345101/files/data_decodings_aurora.tar?download=1
data_decodings_other          all       49b934cafab3a55269018b8391ddbcdd  https://zenodo.org/records/23345097/files/data_decodings_other.tar?download=1
"

DOWNLOAD_DIR=${DOWNLOAD_DIR:-downloads}
DATA_DIR=${ID_DATA_DIR:-data}

if [ $# -eq 0 ]; then
    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

# archives to fetch: "results" selects its group, "all" every archive, otherwise names
selected=""
for arg in "$@"; do
    case $arg in
        results) selected+=" $(echo "$ARCHIVES" | awk '$2 == "results" {print $1}')" ;;
        all)     selected+=" $(echo "$ARCHIVES" | awk 'NF {print $1}')" ;;
        *)       echo "$ARCHIVES" | awk -v n="$arg" '$1 == n {f=1} END {exit !f}' \
                     || { echo "Unknown archive or option: $arg"; exit 1; }
                 selected+=" $arg" ;;
    esac
done

mkdir -p "$DOWNLOAD_DIR" "$DATA_DIR"
for name in $(echo $selected | tr ' ' '\n' | awk '!seen[$0]++'); do
    read -r md5 url <<< "$(echo "$ARCHIVES" | awk -v n="$name" '$1 == n {print $3, $4}')"
    marker="$DOWNLOAD_DIR/.$name.extracted"
    if [ -e "$marker" ]; then
        echo "=== $name: already extracted, skipping"
        continue
    fi

    file="$DOWNLOAD_DIR/$name.tar"
    echo "=== $name: downloading"
    curl -L --fail --retry 3 -C - -o "$file" "$url"   # -C - resumes an interrupted download

    echo "=== $name: checking"
    if [ "$(md5sum "$file" | cut -d' ' -f1)" != "$md5" ]; then
        echo "Checksum of $file does not match. Delete the file and run the script again."
        exit 1
    fi

    echo "=== $name: extracting"
    if [ "$name" = results_paper ]; then
        tar -xf "$file"                                             # -> results/paper/
    else
        tar -xf "$file" -C "$DATA_DIR" --strip-components=1         # data/... -> $DATA_DIR/...
    fi
    touch "$marker"
    [ "${KEEP_ARCHIVES:-0}" = 1 ] || rm "$file"
done
echo "Done."
