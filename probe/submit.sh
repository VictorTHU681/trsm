#!/usr/bin/env bash
# 在 q_pacopt 池（cn20481-cn20499）里挑一个空闲节点直接提交，避免排队。
# 用法：probe/submit.sh <作业名> <脚本> [额外 dsub 参数...]
# 流程：dnode 全表（~60 s）选候选 → 提交前 dnode <节点> 即时复核 → 提交后 6 s 仍 PENDING 则 dkill 换下一个候选。
set -uo pipefail
name="$1"; script="$2"; shift 2
cd "$(dirname "$0")/.."
busy=$(djob 2>/dev/null | awk 'NR>1 && ($3=="RUNNING" || $3=="PENDING"){print $NF}' | tr ',' '\n' | sort -u)
cands=()
# q_pacopt 池 = 带 queue_q_pacopt 标签的节点 cn20481-cn20499（cn20480 属于 q_pacapp）；有作业的节点显示 CLOSE_FULL / CPU_FREE=0
for n in $(dnode 2>/dev/null | awk '$1 ~ /^cn204(8[1-9]|9[0-9])$/ && $2=="OK" && $4==608 {print $1}'); do
    grep -qx "$n" <<<"$busy" || cands+=("$n")
done
submit_on() {
    local node="$1"; shift
    dsub -n "$name" -x job -q q_pacopt -N 1 -nl "$node" -R "cpu=37,mem=60000MB" -T 3600 \
         -oo "$PWD/logs/${name}_%J.out" -eo "$PWD/logs/${name}_%J.err" "$@" "bash $PWD/$script" 2>&1
}
for node in "${cands[@]}"; do
    # 即时复核（单节点查询很快）
    st=$(dnode "$node" 2>/dev/null | awk -v n="$node" '$1==n {print $2":"$4}')
    [[ "$st" == "OK:608" ]] || { echo "skip $node ($st)" >&2; continue; }
    out=$(submit_on "$node" "$@")
    jid=$(grep -oE '<[0-9]+>' <<<"$out" | tr -d '<>')
    [[ -n "$jid" ]] || { echo "$out" >&2; continue; }
    sleep 6
    state=$(djob 2>/dev/null | awk -v j="$jid" '$1==j {print $3}')
    if [[ "$state" == "PENDING" ]]; then
        echo "pending on $node, dkill $jid and retry" >&2
        dkill "$jid" >/dev/null 2>&1
        continue
    fi
    echo "node=$node job=$jid state=$state" >&2
    echo "$out"
    exit 0
done
echo "no idle node, submitting without -nl" >&2
dsub -n "$name" -x job -q q_pacopt -N 1 -R "cpu=37,mem=60000MB" -T 3600 -oo "$PWD/logs/${name}_%J.out" -eo "$PWD/logs/${name}_%J.err" "$@" "bash $PWD/$script"
