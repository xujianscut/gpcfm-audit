#!/bin/bash
# usage: mnsweep.sh <gpu> <worker_id> <n_workers>
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
G=$1; W=$2; NW=$3; k=0
PAIRS="5.0:0.0071 1.0:0.1745 0.5:0.5933 0.35:0.8628 0.25:0.9815"
for p in $PAIRS; do
  L=${p%%:*}; S=${p##*:}
  for arm in gpfix bump; do
    for s in 0 1 2 3 4 5 6 7 8 9; do
      if [ $((k % NW)) -eq $W ]; then
        out=res_mnsweep/${arm}_l${L}_s${s}.json
        mkdir -p res_mnsweep
        for try in 1 2 3 4; do
          [ -f $out ] && break
          CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            python mnist_derisk.py --arm $arm --kern_l $L --sigma_max $S --tag _l$L \
            --seed $s --out $out > res_mnsweep/${arm}_l${L}_s${s}.log 2>&1 || sleep 120
        done
      fi
      k=$((k+1))
    done
  done
done
echo "worker $W done"
