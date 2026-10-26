#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PAIRS="2.0:0.0442 1.0:0.1745 0.5:0.5933 0.35:0.8628 0.25:0.9815"
# wait for every warm file
for p in $PAIRS; do L=${p%%:*}; while [ ! -f warm2/gpfix_l$L.pt ] || [ ! -f warm2/bump_l$L.pt ]; do sleep 10; done; done
echo "warm done $(date)"
for p in $PAIRS; do
  L=${p%%:*}; S=${p##*:}
  for i in 0 1 2 3; do
    s0=$((i*10))
    OMP_NUM_THREADS=1 nohup python derisk.py --arm gpfix --kern_l $L --tag _l$L --seed0 $s0 --nseeds 10 \
       --epochs 5000 --warm warm2/gpfix_l$L.pt --out res2/gpfix_l${L}_$i.json > res2/gpfix_l${L}_$i.log 2>&1 &
  done
  for i in 0 1; do
    s0=$((i*20))
    OMP_NUM_THREADS=1 nohup python derisk.py --arm bump --kern_l $L --sigma_max $S --tag _l$L --seed0 $s0 --nseeds 20 \
       --epochs 5000 --warm warm2/bump_l$L.pt --out res2/bump_l${L}_$i.json > res2/bump_l${L}_$i.log 2>&1 &
  done
done
echo "sweep launched $(date)"
wait
echo "ALL DONE $(date)"
