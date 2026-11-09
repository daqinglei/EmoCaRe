#!/bin/bash
# Zero-shot Qwen2.5-Omni-3B (SSOT 10.18.214): MELD dev -> stop-rule check -> MELD test -> M3ED validation/test -> decode MELD and M3ED.
# Runs on GPUs 2,3 next to the FairHIST scoring (memory is ample; the bottleneck is CPU video decoding).  Resumable: finished caches are skipped.
set -euo pipefail
ROOT=${EMOZOOM_ROOT:-$(pwd)}; PY=${PYTHON:-python}; Z=$ROOT/work_dir/qwen_zeroshot_v1; LOG=$ROOT/logs/zeroshot; GPUS=2,3
mkdir -p "$Z/meld/dev" "$Z/meld/test" "$Z/m3ed" "$LOG"; cd "$ROOT"
ts(){ TZ=Asia/Shanghai date '+%F %T'; }
score(){ # protocol out work
  $PY experiments/run_meld_qwen_sft_selection_eval.py --protocol "$1" --arm zero_shot --output-cache "$2" --work-dir "$3" \
    --num-gpus 2 --gpu-ids $GPUS --cpu-threads 4 --resume >> "$LOG/$(basename "$2" .jsonl).log" 2>&1
}
MD=$ROOT/work_dir/meld_etr_20260914_v2/protocol_official_dev.json; MT=$ROOT/work_dir/meld_emp_devselect_test_20260916/protocol_official_test.json
[ -f "$Z/meld/dev/zeroshot.jsonl" ] || { echo "[$(ts)] MELD dev"; score $MD "$Z/meld/dev/zeroshot.jsonl" "$Z/meld/dev/work"; }
if [ ! -f "$Z/meld/agreement.json" ]; then
  echo "[$(ts)] stop-rule check"; CUDA_VISIBLE_DEVICES=2 $PY experiments/paper2_zeroshot_agreement_20260930.py "$Z/meld/dev/zeroshot.jsonl" 300 > "$LOG/agreement.log" 2>&1 || true
fi
grep -q '"passed": true' "$Z/meld/agreement.json" || { echo "[$(ts)] STOP RULE FAILED or check errored; see $LOG/agreement.log"; exit 1; }
[ -f "$Z/meld/test/zeroshot.jsonl" ] || { echo "[$(ts)] MELD test"; score $MT "$Z/meld/test/zeroshot.jsonl" "$Z/meld/test/work"; }
B=$ROOT/work_dir/m3ed_qwen_at_v1
[ -f "$Z/m3ed/validation.jsonl" ] || { echo "[$(ts)] M3ED validation"; score $B/protocol_v1/protocol.json "$Z/m3ed/validation.jsonl" "$Z/m3ed/work_val"; }
[ -f "$Z/m3ed/test.jsonl" ] || { echo "[$(ts)] M3ED test"; score $B/evaluation_v1/test_protocol.json "$Z/m3ed/test.jsonl" "$Z/m3ed/work_test"; }
for t in meld m3ed; do
  ls "$ROOT"/work_dir/QWEN_ZEROSHOT_${t}_*.json >/dev/null 2>&1 && continue
  echo "[$(ts)] decode $t"; $PY experiments/paper2_zeroshot_decode_20260930.py $t > "$ROOT/work_dir/qwen_zeroshot_decode_$t.log" 2>&1
done
touch "$Z/MELD_M3ED_COMPLETE"; echo "[$(ts)] zero-shot MELD/M3ED complete"
