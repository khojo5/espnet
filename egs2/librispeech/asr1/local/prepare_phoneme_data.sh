#!/usr/bin/env bash
set -e
set -u
set -o pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

textgrid_root=
phone_tier=phones
missing_textgrid=error

if [ "${1:-}" = "--help" ]; then
    echo "Usage: bash local/prepare_phoneme_data.sh --textgrid-root <dir> [--phone-tier <name>] [set ...]"
    echo "Create data/<set>_phn from prepared data/<set>, without overwriting."
    echo "Default sets: train_960 dev test_clean test_other dev_clean dev_other"
    echo "--missing-textgrid error|g2p: stop on missing TextGrids (default), or use English G2P."
    exit 0
fi

. utils/parse_options.sh
if [ -z "${textgrid_root}" ] || [ ! -d "${textgrid_root}" ]; then
    echo "Error: --textgrid-root must specify an existing TextGrid directory" >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    set -- train_960 dev test_clean test_other dev_clean dev_other
fi

declare -A seen
for dset in "$@"; do
    if ! [[ "${dset}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
        echo "Error: invalid dataset name: ${dset}" >&2
        exit 1
    fi
    if [ -n "${seen[${dset}]:-}" ]; then
        echo "Error: duplicate dataset: ${dset}" >&2
        exit 1
    fi
    seen["${dset}"]=true
    if [ -e "data/${dset}_phn" ] || [ -L "data/${dset}_phn" ]; then
        echo "Error: output already exists: data/${dset}_phn" >&2
        exit 1
    fi
    for file in text wav.scp utt2spk spk2utt; do
        if [ ! -s "data/${dset}/${file}" ]; then
            echo "Error: missing or empty input: data/${dset}/${file}" >&2
            exit 1
        fi
    done
    utils/validate_data_dir.sh --no-feats --non-print "data/${dset}"
done

. ./path.sh

work_dir=
cleanup() {
    if [ -n "${work_dir}" ]; then
        rm -r -- "${work_dir}"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

work_dir=$(mktemp -d "${PWD}/data/.phoneme.XXXXXX")
inputs=()
outputs=()
for dset in "$@"; do
    utils/copy_data_dir.sh --validate-opts --non-print \
        "data/${dset}" "${work_dir}/${dset}/data"
    inputs+=("data/${dset}/text")
    outputs+=("${work_dir}/${dset}/text.phn")
done
python local/textgrid_to_phoneme.py \
    --textgrid-root "${textgrid_root}" \
    --phone-tier "${phone_tier}" \
    --missing-textgrid "${missing_textgrid}" \
    --input "${inputs[@]}" \
    --output "${outputs[@]}"

for dset in "$@"; do
    if ! cmp -s \
        <(awk '{print $1}' "data/${dset}/text") \
        <(awk '{print $1}' "${work_dir}/${dset}/text.phn"); then
        echo "Error: phoneme conversion changed utterance IDs: ${dset}" >&2
        exit 1
    fi
    mv -- "${work_dir}/${dset}/text.phn" "${work_dir}/${dset}/data/text"
    if [ "${missing_textgrid}" = g2p ]; then
        mv -- "${work_dir}/${dset}/text.phn.g2p_fallback_utt_ids" \
            "${work_dir}/${dset}/data/g2p_fallback_utt_ids"
    fi
    utils/validate_data_dir.sh --no-feats --non-print "${work_dir}/${dset}/data"
done

for dset in "$@"; do
    # A concurrent run may have created the destination since the initial check.
    if ! mv -T -n -- "${work_dir}/${dset}/data" "data/${dset}_phn" \
        || [ -d "${work_dir}/${dset}/data" ]; then
        if [ -e "data/${dset}_phn" ] || [ -L "data/${dset}_phn" ]; then
            echo "Error: output already exists: data/${dset}_phn" >&2
        else
            echo "Error: could not publish data/${dset}_phn" >&2
        fi
        exit 1
    fi
    echo "Created data/${dset}_phn"
done
