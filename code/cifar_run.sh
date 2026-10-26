#!/bin/bash
# usage: cifar_run.sh <gpu> <arm> <seed>   -- restarts (resuming from _last.pt) on failure, e.g. OOM from other users
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
G=$1; A=$2; S=$3
for try in $(seq 1 30); do
  CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python cifar_derisk.py --arm $A --seed $S --outdir res_cifar >> res_cifar/logs/${A}_s${S}.log 2>&1 && break
  echo "retry $try $(date)" >> res_cifar/logs/${A}_s${S}.log; sleep 600
done
