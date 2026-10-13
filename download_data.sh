#!/bin/bash
# Downloads the published data (DOI: [add DOI]) and extracts it into data/ and results/paper/.
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

#  name                       group     md5                                download link
ARCHIVES="
results_paper                 results   ADD_MD5   ADD_URL
data_base                     results   ADD_MD5   ADD_URL
data_era5                     all       ADD_MD5   ADD_URL
data_latent_states            all       ADD_MD5   ADD_URL
data_decodings_graphcast      all       ADD_MD5   ADD_URL
data_decodings_aurora         all       ADD_MD5   ADD_URL
data_decodings_other          all       ADD_MD5   ADD_URL
"

DOWNLOAD_DIR=${DOWNLOAD_DIR:-downloads}
DATA_DIR=${ID_DATA_DIR:-data}

if [ $# -eq 0 ]; then
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
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
    if [ "$url" = ADD_URL ]; then
        echo "The download links have not been added to this script yet."
        exit 1
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
