#!/bin/bash
# Run one FASTWIND model (pnlte + pformalsol) in its own isolated run directory.
#
#   fastwind_run.sh NAME INDAT FORMAL_INPUT [VTURB_ANSWER] [IESCAT]
#
#   NAME          run-directory name, created as $FW_ROOT/NAME
#   INDAT         INDAT.DAT to use (its first word = model catalogue name)
#   FORMAL_INPUT  line list for pformalsol
#   VTURB_ANSWER  answer to pformalsol's turbulence prompt (default "10 0.1":
#                 10 km/s rising to 0.1 vinf); IESCAT default 0
#
# FASTWIND is serial and opens its data relative to the working directory
# (../inicalc/..., ../HOPFPARA_ALL_*) and writes scratch files (control.dat, OX,
# cross_out, ...) into it, so every model needs its own run directory one level
# below FW_ROOT, which holds links to inicalc and the Hopf files. Many models
# can then run concurrently (see fastwind_scaling.sbatch).
#
# Defaults: FW_ROOT=/scratch/ppathak/fastwind_runs,
#           FW_BUILD=/scratch/ppathak/FW_10.6.4.1/v10.6_HHe (A10HHe = H+He model atom)
set -euo pipefail
NAME=$1; INDAT=$(readlink -f "$2"); FORMAL=$(readlink -f "$3")
VT=${4:-"10 0.1"}; IESCAT=${5:-0}
FW_ROOT=${FW_ROOT:-/scratch/ppathak/fastwind_runs}
FW_BUILD=${FW_BUILD:-/scratch/ppathak/FW_10.6.4.1/v10.6_HHe}
FW_TOP=$(dirname "$FW_BUILD")
ATOM=$(head -1 "$FW_BUILD/ATOM_FILE" | tr -d ' ')          # e.g. A10HHe.dat
TAG=${ATOM%.dat}

# one-time root set-up: ../inicalc and ../HOPFPARA_ALL_* as seen from every run dir
# (atomic: many concurrent launches may race here, so link to a temporary name and
#  rename over the target; a plain `ln -s` on an existing directory link fails)
mkdir -p "$FW_ROOT"
link() { [ -e "$2" ] && return 0; ln -s "$1" "$2.tmp.$$" && mv -Tf "$2.tmp.$$" "$2"; }
link "$FW_TOP/inicalc"                  "$FW_ROOT/inicalc"
link "$FW_TOP/inicalc/HOPFPARA_ALL_HHe" "$FW_ROOT/HOPFPARA_ALL_HHe"
link "$FW_TOP/inicalc/HOPFPARA_ALL_met" "$FW_ROOT/HOPFPARA_ALL_met"

RUN="$FW_ROOT/$NAME"
mkdir -p "$RUN"
cd "$RUN"
for f in ATOM_FILE "$ATOM" "pnlte_$TAG.eo" "pformalsol_$TAG.eo" "ptotout_$TAG.eo"; do
    ln -sf "$FW_BUILD/$f" .
done
cp "$INDAT" INDAT.DAT
cp "$FORMAL" FORMAL_INPUT
MODEL=$(head -1 INDAT.DAT | awk '{print $1}')
mkdir -p "$MODEL"

ulimit -s unlimited 2>/dev/null || true
t0=$(date +%s.%N)
"./pnlte_$TAG.eo" > pnlte.log 2>&1 || true
t1=$(date +%s.%N)
if ! grep -q "ESTO ES EL ACABOSE" pnlte.log; then
    echo "$NAME: pnlte FAILED ($(tail -1 pnlte.log))"; exit 1
fi
printf "%s\n%s\n%s\n" "$MODEL" "$VT" "$IESCAT" | "./pformalsol_$TAG.eo" > pformalsol.log 2>&1
t2=$(date +%s.%N)
nout=$(ls "$MODEL"/OUT.* 2>/dev/null | wc -l)
printf "%s: pnlte %.1f s, pformalsol %.1f s, %d line profiles in %s\n" \
    "$NAME" "$(echo "$t1 - $t0" | bc)" "$(echo "$t2 - $t1" | bc)" "$nout" "$RUN/$MODEL"
