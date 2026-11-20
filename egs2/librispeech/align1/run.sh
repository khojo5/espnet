#!/usr/bin/env bash
set -euo pipefail

# cd egs2/librispeech/align1
# ./run.sh --stage 1 --stop-stage 2 \
#     --test-sets "test_clean test_other" \
#     --asr-train-config ../asr1/exp/asr_conformer-ctc_prior/config.yaml \
#     --asr-model-file ../asr1/exp/asr_conformer-ctc_prior/valid.cer_ctc.ave.pth \
#     --output-root exp/alignment_prior \
#     --device cuda --nj 4

stage=1
stop_stage=2
test_sets="test_clean"
data_root=../asr1/data
output_root=exp/alignment
reference_root=data/reference
textgrid_root=
asr_train_config=
asr_model_file=
device=cpu
nj=1

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
if [ -z "${test_sets}" ]; then
    echo "--test-sets must not be empty" >&2
    exit 2
fi

# Run every subset through a stage before moving on to the next stage.
# Each subset gets its own TextGrid directory (test_clean -> test-clean).
for current_stage in $(seq "${stage}" "${stop_stage}"); do
    for dset in ${test_sets}; do
        subset_textgrid_root=
        if [ -n "${textgrid_root}" ]; then
            subset_textgrid_root="${textgrid_root}/${dset//_/-}"
        fi
        bash ./align.sh \
            --stage "${current_stage}" --stop-stage "${current_stage}" \
            --test-sets "${dset}" --data-root "${data_root}" \
            --output-root "${output_root}" --reference-root "${reference_root}" \
            --textgrid-root "${subset_textgrid_root}" \
            --asr-train-config "${asr_train_config}" \
            --asr-model-file "${asr_model_file}" \
            --device "${device}" --nj "${nj}"
    done
done
