#!/bin/bash
# usage: mnist_worker.sh <gpu> <worker_id> <n_workers> <nseeds>   (retries OOM-killed runs)
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
GPU=$1; W=$2; NW=$3; NS=$4; k=0
for s in $(seq 0 $((NS-1))); do for arm in icfm gp gpfix indep mmdet; do
  if [ $((k % NW)) -eq $W ]; then
    out=res_mnist/${arm}_${s}.json
    for try in 1 2 3 4; do
      [ -f $out ] && break
      CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python mnist_derisk.py --arm $arm --seed $s --out $out > res_mnist/${arm}_${s}.log 2>&1 || sleep 120
    done
  fi; k=$((k+1))
done; done
echo "worker $W done"
