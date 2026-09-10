#!/usr/bin/env bash
# 在 q_pacopt 池（cn20480-cn20499）里挑一个空闲节点直接提交，避免排队。
# 用法：probe/submit.sh <作业名> <脚本> [额外 dsub 参数...]
set -uo pipefail
name="$1"; script="$2"; shift 2
cd "$(dirname "$0")/.."
busy=$(djob 2>/dev/null | awk 'NR>1 && $3=="RUNNING"{print $NF}' | tr ',' '\n' | sort -u)
node=""
# q_pacopt 池 = 带 queue_q_pacopt 标签的节点 cn20481-cn20499（cn20480 属于 q_pacapp）
for n in $(dnode 2>/dev/null | awk '$1 ~ /^cn204(8[1-9]|9[0-9])$/ && $2=="OK" && $4==608 {print $1}'); do
    if ! grep -qx "$n" <<<"$busy"; then node=$n; break; fi
done
if [[ -z "$node" ]]; then echo "no idle node, submitting without -nl" >&2
    dsub -n "$name" -x job -q q_pacopt -N 1 -R "cpu=37,mem=60000MB" -T 3600 -oo "$PWD/logs/${name}_%J.out" -eo "$PWD/logs/${name}_%J.err" "$@" "bash $PWD/$script"
else
    echo "node=$node" >&2
    dsub -n "$name" -x job -q q_pacopt -N 1 -nl "$node" -R "cpu=37,mem=60000MB" -T 3600 -oo "$PWD/logs/${name}_%J.out" -eo "$PWD/logs/${name}_%J.err" "$@" "bash $PWD/$script"
fi
