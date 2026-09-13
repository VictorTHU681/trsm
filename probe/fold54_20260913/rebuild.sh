#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 prepare_fold.py
python3 prepare_kernels.py
python3 prepare_prefetch.py
python3 prepare_tile.py
python3 prepare_diagstore.py
python3 prepare_width.py
python3 prepare_noprof.py
python3 prepare_diag_sme.py
python3 prepare_diag_combo.py
python3 prepare_fold_last.py
python3 prepare_diag_final.py
python3 prepare_diag_close.py
python3 prepare_diag_late.py
python3 prepare_diag_fuse.py
python3 prepare_diag8.py
python3 prepare_diag8_compact.py
python3 prepare_fused_final.py
python3 prepare_fused_close.py
python3 prepare_fused_align.py
