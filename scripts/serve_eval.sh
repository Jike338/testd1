#!/bin/bash
# Serve one checkpoint with vLLM on the GPU(s) in CUDA_VISIBLE_DEVICES, evaluate it on the held-out ids, stop the server.
# usage: scripts/serve_eval.sh <checkpoint_or_hf_id> <tag> <data_root> [port] [extra attrib.eval_vllm args]
# Writes results/<tag>/. Adapt freely to your scheduler (Slurm, k8s, ...); only the two commands matter.
set -uo pipefail
CKPT=$1; TAG=$2; DATA=$3; PORT=${4:-8000}; shift $(( $# < 4 ? $# : 4 ))
REPO=$(cd "$(dirname "$0")/.." && pwd); mkdir -p $REPO/logs
cat $REPO/data/splits/id_test_ids.txt $REPO/data/splits/ood_test_ids.txt > /tmp/wwp_eval_ids_$PORT.txt
vllm serve "$CKPT" --served-model-name "$TAG" --port $PORT --max-model-len 40960 --gpu-memory-utilization ${VLLM_MEM:-0.90} \
  --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-seqs 64 > $REPO/logs/vllm_$TAG.log 2>&1 &
VP=$!; trap 'kill $VP 2>/dev/null; wait $VP 2>/dev/null' EXIT
for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $VP 2>/dev/null || { echo VLLM_DIED; tail -20 $REPO/logs/vllm_$TAG.log; exit 1; }; sleep 10; done
cd $REPO && python -m attrib.eval_vllm --model "$TAG" --base http://127.0.0.1:$PORT/v1 --data-root "$DATA" --out results/$TAG --ids /tmp/wwp_eval_ids_$PORT.txt "$@"
