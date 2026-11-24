#!/usr/bin/env bash
# Matched-budget comparison of head layouts on a slice of the Qwen3-8B vectors.
# Every run: same 12 training chunks (1.2M pairs), batch 1024, 3000 steps, seed 1234.
# Usage: ROOT=data OUT=runs bash innovations/sweep_cpu.sh
set -u
ROOT=${ROOT:-/data}; OUT=${OUT:-runs}; STEPS=${STEPS:-3000}
COMMON="--root $ROOT --steps $STEPS --batch 1024 --eval-every 1000 --gen-limit 20480 --hn-limit 20000"
run() { name=$1; shift; [ -f "$OUT/$name/result.json" ] && return; python3 -W ignore -m innovations.train $COMMON --out "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1; echo "done $name"; }
mkdir -p "$OUT"
run base_w1536_p512
run shared_p512           --tower shared
run shared_adapter_p512   --tower shared_adapter
run mrl_p1024             --proj 1024 --matryoshka 64,128,256,512,1024
run p128                  --proj 128
run p256                  --proj 256
run p1024                 --proj 1024
run w768_p512             --width 768
run depth2_p512           --depth 2
echo ALLDONE
