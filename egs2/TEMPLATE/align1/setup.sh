#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <egs2/corpus/align1>" >&2
    exit 2
fi
template_dir=$(cd "$(dirname "$0")" && pwd -P)
dir=$(realpath -m "$1")
if [ "$(realpath -m "${dir}/../../TEMPLATE/align1")" != "${template_dir}" ]; then
    echo "Target must be a recipe directory under the same egs2 tree" >&2
    exit 2
fi
for f in run.sh README.md align.sh path.sh scripts pyscripts utils local/path.sh; do
    if [ -e "${dir}/${f}" ] || [ -L "${dir}/${f}" ]; then
        echo "Refusing to overwrite ${dir}/${f}" >&2
        exit 2
    fi
done
mkdir -p "${dir}/local"
cp "${template_dir}/run.sh" "${template_dir}/README.md" "${dir}/"
cp "${template_dir}/local/path.sh" "${dir}/local/path.sh"
ln -s ../../TEMPLATE/align1/align.sh "${dir}/align.sh"
for f in path.sh scripts pyscripts utils; do
    ln -s "../../TEMPLATE/asr1/${f}" "${dir}/${f}"
done
echo "Created alignment recipe: ${dir}"
