#!/bin/bash
# Stage 2 worker: one FASTWIND model (pnlte + pformalsol) for one sphere point.
#
#   fw_sphere_point.sh IDX TEFF        (called by fw_sphere_node.sbatch through xargs -P)
#
# Environment (set by the node launcher):
#   FWL         node-local FASTWIND root (in memory): inicalc/, HOPFPARA_ALL_*, bin/, INDAT.template, FORMAL_INPUT
#   RES         node-local result directory; a finished point appears there as RES/P<idx>/ (atomic rename)
#   KEEP_MODEL  1 = also keep the files needed to rerun pformalsol later (~6.6 MB per point)
#   PNLTE_TIMEOUT  seconds before a hung pnlte is killed (default 3600)
#
# The run directory FWL/P<idx> sits one level below FWL, so FASTWIND's ../inicalc and
# ../HOPFPARA_ALL_* resolve to the node-local copies. Nothing shared is written here.
idx=$1; teff=$2
name=$(printf "P%06d" "$idx")
run=$FWL/$name
rm -rf "$run" "$RES/$name.tmp"
mkdir -p "$run/$name" && cd "$run" || exit 1
ln -s "$FWL"/bin/* .
awk -v n="$name" -v t="$teff" 'NR == 1 {printf "%-47sCATALOG\n", n; next}
     NR == 4 {sub(/^[^,]*,/, sprintf("%.3f,", t))} {print}' "$FWL/INDAT.template" > INDAT.DAT
cp "$FWL/FORMAL_INPUT" .
ulimit -s unlimited 2>/dev/null

t0=$(date +%s.%N)
timeout "${PNLTE_TIMEOUT:-3600}" ./pnlte_A10HHe.eo > pnlte.log 2>&1
rc=$?
t1=$(date +%s.%N)
if grep -q "ESTO ES EL ACABOSE" pnlte.log; then
    printf "%s\n10 0.1\n0\n" "$name" | ./pformalsol_A10HHe.eo > pformalsol.log 2>&1
    nout=$(ls "$name"/OUT.* 2>/dev/null | wc -l)
    [ "$nout" -eq 3 ] && status=ok || status=formal_failed
elif [ $rc -eq 124 ]; then
    status=pnlte_timeout
else
    status=pnlte_failed
fi
t2=$(date +%s.%N)
niter=$(grep -c "ITERATION NO" pnlte.log)
tr23=$(grep "T(TAUROSS=2/3)" pnlte.log | tail -1 | awk '{print $NF}')

out=$RES/$name.tmp
mkdir -p "$out"
cp INDAT.DAT "$out/"
cp "$name"/OUT.* "$out/" 2>/dev/null
if [ "${KEEP_MODEL:-0}" = 1 ] && [ $status = ok ]; then
    for f in MODEL NLTE_POP LTE_POP ENION TAU_ROS FLUXCONT CLUMPING_OUTPUT CONT_FORMAL CONT_FORMAL_ALL; do
        cp "$name/$f" "$out/" 2>/dev/null
    done
fi
[ $status != ok ] && tail -40 pnlte.log > "$out/pnlte_tail.log"
awk -v i="$idx" -v t="$teff" -v s="$status" -v n="$niter" -v r="${tr23:-nan}" -v a="$t0" -v b="$t1" -v c="$t2" \
    'BEGIN {printf "%d %s %s %d %s %.1f %.1f\n", i, t, s, n, r, b - a, c - b}' > "$out/meta.txt"
mv "$out" "$RES/$name"                 # atomic: the packer only ever sees complete results
cd "$FWL" && rm -rf "$run"
