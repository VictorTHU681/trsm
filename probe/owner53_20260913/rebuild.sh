#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 prepare.py
python3 prepare_implicit_only.py
python3 prepare_diag.py
python3 prepare_diag_bound.py
python3 prepare_final.py
python3 prepare_close.py
