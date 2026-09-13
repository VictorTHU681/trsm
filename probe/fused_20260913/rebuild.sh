#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd "$(dirname "$0")" && pwd)
python3 "$experiment_dir/prepare.py"
python3 "$experiment_dir/prepare_front.py"
python3 "$experiment_dir/prepare_front_validation.py"
python3 "$experiment_dir/prepare_direct.py"
