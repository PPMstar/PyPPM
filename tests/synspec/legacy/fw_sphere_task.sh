#!/bin/bash
# Stage 2, one node's share of the per-point FASTWIND run (called once per node).
#
#   fw_sphere_task.sh RUN_DIR K [POINT_LIST]
#
# Task index k (0..K-1): $FW_TASK, else $SLURM_ARRAY_TASK_ID, else $SLURM_PROCID (the srun rank
# in fw_sphere_multinode.sbatch). Task k takes every K-th point (lines k, k+K, ... of points.txt
# or POINT_LIST) and writes only to RUN_DIR/results/task_<k> (task_<listname>_<k> for a list).
# Resume with the same K (finished points in that directory are skipped); for a different split
# use a point list (e.g. RUN_DIR/missing.txt from fw_sphere_merge.py --combine).
#   RUN_DIR     output of fw_sphere_extract.py (points.txt, meta.json)
# Environment: NW (concurrent models per node, default 192), KEEP_MODEL (0/1), PNLTE_TIMEOUT (s)
#
# Design (no shared writes, one read of the shared table per node):
#   * FASTWIND data, executables and all run directories live in node memory (/dev/shm);
#   * the slice of the points table is read once; workers get "idx teff" from xargs;
#   * each worker writes only its own directory; one packer process per node moves finished
#     points into RUN_DIR/results/<tag>/part_*.tar.gz (+ .idx listing the points, written
#     after the tar is complete, both via temporary name + rename);
#   * points listed in existing .idx files are skipped, so a resubmitted task continues;
#   * SIGUSR1 (sent 15 min before the time limit) stops the workers and packs what is finished.
set -uo pipefail
RUN_DIR=$1; K=${2:-1}; LIST=${3:-}
k=${FW_TASK:-${SLURM_ARRAY_TASK_ID:-${SLURM_PROCID:-0}}}   # srun rank in a multi-node job, else array index
NW=${NW:-192}; export KEEP_MODEL=${KEEP_MODEL:-0} PNLTE_TIMEOUT=${PNLTE_TIMEOUT:-3600}
ANA=/home/ppathak/stellar-atmosphere-KU-Leuven/project/analysis
FWTOP=/scratch/ppathak/FW_10.6.4.1
if [ -n "$LIST" ]; then TAG=$(printf "task_%s_%04d" "$(basename "$LIST" .txt)" "$k"); else TAG=$(printf "task_%04d" "$k"); fi
OUT=$RUN_DIR/results/$TAG
JID=${SLURM_JOB_ID:-local$$}
LOCAL=/dev/shm/fwsphere_${JID}_$k            # node memory; ~0.4 GB install + ~7.5 GB run dirs
export FWL=$LOCAL/fw RES=$LOCAL/res
STAGE=$LOCAL/stage
mkdir -p "$OUT" "$FWL/bin" "$FWL/inicalc" "$RES" "$STAGE"
umask 007
echo "$(date) host $(hostname) task $k/$K tag $TAG NW=$NW KEEP_MODEL=$KEEP_MODEL local=$LOCAL"

# ---- node-local FASTWIND installation (read once from scratch) ------------------------
cp -r "$FWTOP"/inicalc/{DATA,OP_DATA_NEW,ATOMDAT_NEW,RaymondSmith} "$FWL/inicalc/"
cp "$FWTOP"/inicalc/HOPFPARA_ALL_{HHe,met} "$FWL/"
cp "$FWTOP"/v10.6_HHe/{pnlte_A10HHe.eo,pformalsol_A10HHe.eo,ATOM_FILE,A10HHe.dat} "$FWL/bin/"
cp "$ANA/fastwind/INDAT_M424test.DAT" "$FWL/INDAT.template"          # Mdot 1e-10; TEFF replaced per point
cp "$ANA/fastwind/FORMAL_INPUT_He3" "$FWL/FORMAL_INPUT"
echo "$(date) node-local install: $(du -sh "$FWL" | cut -f1)"

# ---- this node's points, minus those already done ----------------------------------------
awk -v K="$K" -v k="$k" '(NR - 1) % K == k' "${LIST:-$RUN_DIR/points.txt}" > "$LOCAL/mine.txt"   # one read per node
cat "$OUT"/*.idx 2>/dev/null > "$LOCAL/done.txt"
awk 'FILENAME == ARGV[1] {d[$1]; next} !($1 in d)' "$LOCAL/done.txt" "$LOCAL/mine.txt" > "$LOCAL/todo.txt"   # (NR==FNR fails for an empty done list)
echo "$(date) points: $(wc -l < "$LOCAL/mine.txt") assigned, $(wc -l < "$LOCAL/done.txt") done before, $(wc -l < "$LOCAL/todo.txt") to run"

# ---- packer: the only process that writes to $OUT ----------------------------------------
pack() {
    local n part
    mapfile -t fin < <(ls "$RES" 2>/dev/null | grep -v '\.tmp$')
    n=${#fin[@]}; [ "$n" -eq 0 ] && return
    for p in "${fin[@]}"; do mv "$RES/$p" "$STAGE/"; done
    part=part_$(date +%Y%m%d_%H%M%S)_$RANDOM
    tar -czf "$OUT/$part.tar.gz.tmp" -C "$STAGE" . && mv "$OUT/$part.tar.gz.tmp" "$OUT/$part.tar.gz"
    cat "$STAGE"/*/meta.txt > "$OUT/$part.idx.tmp" && mv "$OUT/$part.idx.tmp" "$OUT/$part.idx"
    rm -rf "${STAGE:?}"/*
    echo "$(date) packed $n points -> $part"
}
( while true; do sleep 900; pack; done ) &
PACKER=$!

# ---- run ---------------------------------------------------------------------------------
xargs -r -a "$LOCAL/todo.txt" -P "$NW" -n 2 "$ANA/fw_sphere_point.sh" &     # -r: nothing to do -> no call
XPID=$!
stop() {   # exclusive node: all these processes are ours. Workers first, so an interrupted
           # point records nothing (and is rerun later) instead of being marked failed.
    echo "$(date) SIGUSR1: stopping workers"
    kill "$XPID" 2>/dev/null
    pkill -u "$USER" -f fw_sphere_point.sh 2>/dev/null
    pkill -u "$USER" -f pnlte_A10HHe.eo 2>/dev/null
    pkill -u "$USER" -f pformalsol_A10HHe.eo 2>/dev/null
}
trap stop USR1
wait "$XPID"; wait "$XPID" 2>/dev/null
kill "$PACKER" 2>/dev/null; wait "$PACKER" 2>/dev/null
pack
ok=$(cat "$OUT"/*.idx 2>/dev/null | awk '$3 == "ok"' | wc -l)
bad=$(cat "$OUT"/*.idx 2>/dev/null | awk '$3 != "ok"' | wc -l)
echo "$(date) task $TAG finished: $ok ok, $bad failed, of $(wc -l < "$LOCAL/mine.txt") assigned"
rm -rf "$LOCAL"
