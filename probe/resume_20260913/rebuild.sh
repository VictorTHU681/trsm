#!/usr/bin/env bash
set -euo pipefail
experiment_dir=$(cd "$(dirname "$0")" && pwd)
for generator in prepare.py prepare_validation.py prepare_fullinv.py prepare_shapes.py \
    prepare_strassen.py prepare_dense_profile.py prepare_prime.py prepare_dense_layout.py \
    prepare_dense_data.py prepare_dense_tasks.py prepare_dense_color.py \
    prepare_component_check.py prepare_special_mb.py prepare_dense_clear.py; do
    python3 "$experiment_dir/$generator"
done
