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
#
# ZENODO_DRAFT=1 bash download_data.sh ... downloads from the unpublished Zenodo drafts instead
# (only for the owner of the records, to test the download before publishing). It asks for a Zenodo
# token with the scope deposit:write.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# Each archive is stored on Zenodo in pieces of 1 GiB (<name>.tar.000, <name>.tar.001, ...).
#  name                       group     record    pieces  md5 of the whole archive
ARCHIVES="
results_paper                 results   23345097   2      8315e8e17193d2c7f24151cce5a18329
data_base                     results   23345097   3      45bc59bc12eed68d507ed6a2af3fa0da
data_era5                     all       23345097  16      daa1bcf311883c9db3df324837ef2346
data_latent_states            all       23345099   8      4f2a0a4e81d60eef23fbb06da42a9103
data_decodings_graphcast      all       23345099  30      e230e89135b1a24f48c81a3341fc3779
data_decodings_aurora         all       23345101  27      b1d75e84338cda501fad5836be03b978
data_decodings_other          all       23345097  21      49b934cafab3a55269018b8391ddbcdd
"

DOWNLOAD_DIR=${DOWNLOAD_DIR:-downloads}
DATA_DIR=${ID_DATA_DIR:-data}

if [ $# -eq 0 ]; then
    sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'
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

if [ "${ZENODO_DRAFT:-0}" = 1 ]; then
    read -rsp "Zenodo token (input is not shown): " ZENODO_TOKEN
    echo
fi

# download <record> <file> <output>: from the published record, or with ZENODO_DRAFT=1 from the draft.
# The token is passed to curl on stdin, so it does not appear in the process list.
download() {
    if [ "${ZENODO_DRAFT:-0}" = 1 ]; then
        printf 'header = "Authorization: Bearer %s"\n' "$ZENODO_TOKEN" |
            curl --config - -L --fail --retry 5 --progress-bar -C - -o "$3" \
                "https://zenodo.org/api/records/$1/draft/files/$2/content"
    else
        curl -L --fail --retry 5 --progress-bar -C - -o "$3" \
            "https://zenodo.org/records/$1/files/$2?download=1"           # -C - resumes a download
    fi
}

mkdir -p "$DOWNLOAD_DIR" "$DATA_DIR"
for name in $(echo $selected | tr ' ' '\n' | awk '!seen[$0]++'); do
    read -r record npieces md5 <<< "$(echo "$ARCHIVES" | awk -v n="$name" '$1 == n {print $3, $4, $5}')"
    marker="$DOWNLOAD_DIR/.$name.extracted"
    if [ -e "$marker" ]; then
        echo "=== $name: already extracted, skipping"
        continue
    fi

    echo "=== $name: downloading $npieces pieces"
    pieces=()
    for i in $(seq 0 $((npieces - 1))); do
        piece=$(printf "%s.tar.%03d" "$name" "$i")
        pieces+=("$DOWNLOAD_DIR/$piece")
        [ -e "$DOWNLOAD_DIR/.$piece.done" ] && continue
        echo "$piece"
        download "$record" "$piece" "$DOWNLOAD_DIR/$piece"
        touch "$DOWNLOAD_DIR/.$piece.done"
    done

    echo "=== $name: checking"
    if [ "$(cat "${pieces[@]}" | md5sum | cut -d' ' -f1)" != "$md5" ]; then
        echo "Checksum of $name does not match. Delete $DOWNLOAD_DIR/$name.tar.* and"
        echo "$DOWNLOAD_DIR/.$name.tar.*.done and run the script again."
        exit 1
    fi

    echo "=== $name: extracting"
    if [ "$name" = results_paper ]; then
        cat "${pieces[@]}" | tar -xf -                                      # -> results/paper/
    else
        cat "${pieces[@]}" | tar -xf - -C "$DATA_DIR" --strip-components=1 # data/... -> $DATA_DIR/...
    fi
    touch "$marker"
    if [ "${KEEP_ARCHIVES:-0}" != 1 ]; then
        for piece in "${pieces[@]}"; do
            rm "$piece" "$DOWNLOAD_DIR/.$(basename "$piece").done"
        done
    fi
done
echo "Done."
