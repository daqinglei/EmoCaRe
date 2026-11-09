#!/bin/bash
set -euo pipefail
ROOT="${EMOZOOM_ROOT:-$(pwd)}"
PY="${PYTHON:-python}"
BASE=$ROOT/work_dir/m3ed_qwen_at_v1
PROTO=$BASE/protocol_v1/protocol.json
RUN=$BASE/emitter_v1/formal
EVAL=$BASE/evaluation_v1
LOG=$ROOT/logs/m3ed
cd "$ROOT"; mkdir -p "$EVAL/validation" "$EVAL/test" "$LOG"
while [ ! -f "$BASE/emitter_v1/TRAINING_COMPLETE" ]; do sleep 60; done
for ckpt in "$RUN"/checkpoint-*; do
  step=${ckpt##*-}; out="$EVAL/validation/checkpoint-${step}.jsonl"
  if [ ! -f "$out" ]; then
    $PY experiments/run_meld_qwen_sft_selection_eval.py --protocol "$PROTO" \
      --arm seven_state_sft --adapter-path "$ckpt/adapter" --output-cache "$out" \
      --work-dir "$EVAL/validation/work-${step}" --num-gpus 4 --gpu-ids 0,1,2,3 --resume \
      > "$LOG/validation_checkpoint-${step}.log" 2>&1
  fi
done
$PY experiments/paper2_m3ed_select_checkpoint_20260922.py \
  --cache "$EVAL"/validation/checkpoint-*.jsonl --out "$EVAL/checkpoint_selection.json" \
  > "$LOG/checkpoint_selection.log" 2>&1
$PY - <<'PY'
import hashlib,json
from pathlib import Path
base=Path('work_dir/m3ed_qwen_at_v1'); source=base/'protocol_v1/protocol.json'; d=json.loads(source.read_text())
d['derived_from_protocol_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
d['data']['selection_split_name']='test'; d['data']['selection_samples']=d['splits']['test']['utterances']
d['artifacts']['selection_manifest']=str((base/'protocol_v1/test_manifest.jsonl').resolve())
d['artifacts']['selection_manifest_sha256']=d['splits']['test']['sha256']
d['evaluation']={'split':'M3ED official test','access':'exactly once after validation checkpoint selection'}
(base/'evaluation_v1/test_protocol.json').write_text(json.dumps(d,indent=2)+'\n')
sel=json.loads((base/'evaluation_v1/checkpoint_selection.json').read_text())
(base/'evaluation_v1/selected_step.txt').write_text(str(sel['selected']['step'])+'\n')
PY
step=$(cat "$EVAL/selected_step.txt")
$PY experiments/run_meld_qwen_sft_selection_eval.py --protocol "$EVAL/test_protocol.json" \
  --arm seven_state_sft --adapter-path "$RUN/checkpoint-$step/adapter" \
  --output-cache "$EVAL/test/selected_checkpoint-${step}.jsonl" \
  --work-dir "$EVAL/test/work-${step}" --num-gpus 4 --gpu-ids 0,1,2,3 --resume \
  > "$LOG/test_selected_checkpoint-${step}.log" 2>&1
$PY experiments/paper2_m3ed_decode_20260922.py \
  --train-manifest "$BASE/protocol_v1/train_manifest.jsonl" \
  --validation-manifest "$BASE/protocol_v1/validation_manifest.jsonl" \
  --test-manifest "$BASE/protocol_v1/test_manifest.jsonl" \
  --validation-cache "$EVAL/validation/checkpoint-${step}.jsonl" \
  --test-cache "$EVAL/test/selected_checkpoint-${step}.jsonl" \
  --protocol "$PROTO" --out "$EVAL/final_macd_rrtd_results.json" \
  > "$LOG/final_decode.log" 2>&1
touch "$EVAL/EVALUATION_COMPLETE"
