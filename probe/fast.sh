#!/usr/bin/env bash
# 一键快速基准：把变体写入 probe/fast_variants.txt，提交到空闲节点，等待完成并打印结果。
# 用法：probe/fast.sh ["TRSM_KC=96" "TRSM_MB=256 TRSM_KC=128" ...]   （无参数 = 只跑默认）
# 典型耗时：提交 ~10 s（候选缓存）+ 运行 ~20 s + 每个变体 ~15 s
set -uo pipefail
cd "$(dirname "$0")/.."
: > probe/fast_variants.txt
for v in "$@"; do echo "$v" >> probe/fast_variants.txt; done
out=$(bash probe/submit.sh trsm_fast probe/fast_job.sh 2>&1)
jid=$(grep -oE 'Submit job <[0-9]+>' <<<"$out" | grep -oE '[0-9]+')
grep -E "node=|no idle" <<<"$out" >&2
[[ -n "$jid" ]] || { echo "$out"; exit 1; }
for i in $(seq 1 240); do
    st=$(djob 2>/dev/null | awk -v j="$jid" '$1==j{print $3}')
    [[ -z "$st" ]] && break
    sleep 5
done
f="logs/trsm_fast_${jid}.out"
[[ -f "$f" ]] || { echo "no output ($f)"; exit 1; }
grep -vE "^\[env\]" "$f"
