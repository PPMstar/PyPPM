#!/bin/bash
# Rerun the modified pformalsol (FW_10.6.4.1/v10.6_HHe_imu, writes OUT_IMU.*) for saved models, on the host.
#   fw_imu_run.sh REPRESENTATIVES_TXT RUNS_DIR [NPAR]
# REPRESENTATIVES_TXT lines: "bin idx teff model_dir" (fw_imu_library.py --stage select). Existing outputs are skipped.
LIST=$1; RUNS=$2; NPAR=${3:-20}
FW=/scratch/ppathak/FW_10.6.4.1; BUILD=$FW/v10.6_HHe_imu
FI=/home/ppathak/stellar-atmosphere-KU-Leuven/project/analysis/fastwind/FORMAL_INPUT_He3
mkdir -p "$RUNS"
for d in inicalc inicalc/HOPFPARA_ALL_HHe inicalc/HOPFPARA_ALL_met; do
    [ -e "$RUNS/$(basename $d)" ] || ln -s "$FW/$d" "$RUNS/$(basename $d)"
done
one() {
    src=$1; name=$(basename "$src"); run=$RUNS/$name; m=$run/$name
    [ -f "$m/OUT_IMU.HEI4922_VTV010" ] && [ -f "$m/OUT_IMU.HEI4026_VTV010" ] && [ -f "$m/OUT_IMU.HEII4200_VTV010" ] && return 0
    mkdir -p "$m"
    for f in pformalsol_A10HHe.eo ATOM_FILE A10HHe.dat; do ln -sf "$BUILD/$f" "$run/$f"; done
    for f in MODEL NLTE_POP LTE_POP ENION TAU_ROS FLUXCONT CLUMPING_OUTPUT CONT_FORMAL CONT_FORMAL_ALL; do ln -sf "$src/$f" "$m/$f"; done
    cp "$src/INDAT.DAT" "$run/"; cp "$FI" "$run/FORMAL_INPUT"
    (cd "$run" && ulimit -s unlimited && printf "%s\n10 0.1\n0\n" "$name" | ./pformalsol_A10HHe.eo > pformalsol.log 2>&1)
}
export -f one; export RUNS BUILD FI
awk '{print $4}' "$LIST" | xargs -P "$NPAR" -I{} bash -c 'one "$@"' _ {}
n=$(ls "$RUNS"/P*/P*/OUT_IMU.HEI4922_VTV010 2>/dev/null | wc -l)
echo "$(date) $n of $(wc -l < "$LIST") models have OUT_IMU files"
