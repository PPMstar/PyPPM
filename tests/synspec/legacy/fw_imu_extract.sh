#!/bin/bash
# Extract selected model directories from the per-point run's archives (for the intensity library).
#   fw_imu_extract.sh PARTS_FILE OUT_DIR [NPAR]
# PARTS_FILE lines: "<part>.tar.gz P0xxxxx P0yyyyy ..." (written by the coverage search); NPAR parallel tar streams.
PARTS=$1; OUT=$2; NPAR=${3:-8}
mkdir -p "$OUT"
awk '{printf "%s", $1; for (i = 2; i <= NF; i++) printf " ./%s", $i; printf "\n"}' "$PARTS" | \
    xargs -P "$NPAR" -L 1 bash -c 'tar -xzf "$0" -C '"$OUT"' "$@" && echo "done $(basename $0): $#"'
echo "extracted $(ls -d "$OUT"/P* | wc -l) model directories into $OUT"
