#!/bin/bash
# Train one run, then evaluate every export on the same GPU.
# usage: CUDA_VISIBLE_DEVICES=0 scripts/train_and_eval.sh <sft|opsd|grpo> <run_dir> <data_root> [port]
# (OPSD with "teacher_device": "cuda:1" needs two visible GPUs.)
set -uo pipefail
METHOD=$1; RUN=$(cd "$2" && pwd); DATA=$3; PORT=${4:-8000}
REPO=$(cd "$(dirname "$0")/.." && pwd); mkdir -p $REPO/logs $RUN/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
cd $REPO && python -u -m attrib.train_$METHOD $RUN > $RUN/logs/train.log 2>&1 || { echo "training failed: $RUN/logs/train.log"; exit 1; }
for c in $(ls -d $RUN/checkpoints/step* | sort); do
  bash $REPO/scripts/serve_eval.sh $c $(basename $RUN)_$(basename $c) $DATA $PORT > $RUN/logs/eval_$(basename $c).log 2>&1 || echo "eval failed: $c"
done
