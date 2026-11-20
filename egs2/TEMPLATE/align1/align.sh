#!/usr/bin/env bash
# Phoneme CTC forced alignment: 0=reference conversion, 1=inference, 2=scoring.
set -euo pipefail

stage=1
stop_stage=2
test_sets=test
data_root=../asr1/data
output_root=exp/alignment
reference_root=data/reference
textgrid_root=
word_tier=words
phone_tier=phones
asr_train_config=
asr_model_file=
model_tag=
phone_map=
audio_type=sound
input_fs=
device=cpu
nj=1
cmd=utils/run.pl

. utils/parse_options.sh
if [ "$#" -ne 0 ]; then
    echo "Use named options; see README.md" >&2
    exit 2
fi
if ! [[ ${stage} =~ ^[012]$ && ${stop_stage} =~ ^[012]$ ]] || \
    [ "${stage}" -gt "${stop_stage}" ]; then
    echo "Invalid stage range (0=reference, 1=infer, 2=score)" >&2
    exit 2
fi
if ! [[ ${nj} =~ ^[1-9][0-9]*$ ]]; then
    echo "--nj must be a positive integer" >&2
    exit 2
fi
if [ -z "${test_sets}" ]; then
    echo "--test-sets must not be empty" >&2
    exit 2
fi
. ./path.sh

if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    if [ -z "${textgrid_root}" ]; then
        echo "--textgrid-root is required for Stage 0" >&2
        exit 2
    fi
    if [ ! -d "${textgrid_root}" ]; then
        echo "TextGrid directory not found: ${textgrid_root}" >&2
        exit 2
    fi
    for dset in ${test_sets}; do
        python pyscripts/utils/prepare_alignment_reference.py \
            --textgrid-root "${textgrid_root}" \
            --word-tier "${word_tier}" --phone-tier "${phone_tier}" \
            --text "${data_root}/${dset}/text" \
            --output "${reference_root}/${dset}.jsonl" \
            --pronunciation-output "${reference_root}/${dset}.pronunciations.jsonl" \
            --key-output "${reference_root}/${dset}.keys"
    done
fi

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    model_opts=()
    if [ -n "${model_tag}" ]; then model_opts+=(--model-tag "${model_tag}"); fi
    if [ -n "${asr_train_config}" ]; then model_opts+=(--asr-train-config "${asr_train_config}"); fi
    if [ -n "${asr_model_file}" ]; then model_opts+=(--asr-model-file "${asr_model_file}"); fi
    if [ -n "${input_fs}" ]; then model_opts+=(--input-fs "${input_fs}"); fi

    for dset in ${test_sets}; do
        data_dir=${data_root}/${dset}
        key_file=${reference_root}/${dset}.keys
        log_dir=${output_root}/${dset}/log
        mkdir -p "${log_dir}"

        # Only utterances with a reference TextGrid (listed in the key file) are aligned.
        n_utts=$(wc -l < "${key_file}")
        if [ "${n_utts}" -eq 0 ]; then
            echo "Empty key file: ${key_file}" >&2
            exit 2
        fi
        _nj=$(( nj < n_utts ? nj : n_utts ))
        split_keys=()
        for n in $(seq "${_nj}"); do
            split_keys+=("${log_dir}/keys.${n}.scp")
        done
        utils/split_scp.pl "${key_file}" "${split_keys[@]}"

        data_opts=()
        if [ -f "${data_dir}/segments" ]; then
            data_opts+=(--segments "${data_dir}/segments")
        fi
        # shellcheck disable=SC2086
        ${cmd} JOB=1:"${_nj}" "${log_dir}/alignment.JOB.log" \
            python -m espnet2.bin.alignment_inference \
                --wav-scp "${data_dir}/wav.scp" --text "${data_dir}/text" \
                --pronunciations "${reference_root}/${dset}.pronunciations.jsonl" \
                --key-file "${log_dir}/keys.JOB.scp" \
                --audio-type "${audio_type}" --device "${device}" \
                "${model_opts[@]}" "${data_opts[@]}" \
                --output "${log_dir}/hyp.JOB.jsonl"

        for n in $(seq "${_nj}"); do
            echo "${log_dir}/hyp.${n}.jsonl"
        done > "${output_root}/${dset}/hypotheses.list"
    done
fi

if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    score_opts=()
    if [ -n "${phone_map}" ]; then score_opts+=(--phone-map "${phone_map}"); fi

    for dset in ${test_sets}; do
        mapfile -t hypotheses < "${output_root}/${dset}/hypotheses.list"
        if [ "${#hypotheses[@]}" -eq 0 ]; then
            echo "Empty hypothesis list for ${dset}" >&2
            exit 2
        fi
        python -m espnet2.bin.alignment_scoring \
            --reference "${reference_root}/${dset}.jsonl" \
            --hypothesis "${hypotheses[@]}" \
            --output-dir "${output_root}/${dset}/score" \
            "${score_opts[@]}"
    done
fi
