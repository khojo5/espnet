#!/usr/bin/env bash
set -euo pipefail

test_sets=test
data_root=../asr1/data

bash ./align.sh --test-sets "${test_sets}" --data-root "${data_root}" "$@"
