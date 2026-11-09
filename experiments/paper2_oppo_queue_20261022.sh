#!/bin/bash
# OPPO baseline (SSOT 10.18.351): the zero-shot Qwen2.5-Omni scoring procedure (SSOT 10.18.214 / 10.18.300 / 10.18.328) with only
# the model path changed to model/oppo-7b.  MELD dev -> stop-rule check -> MELD test -> IEMOCAP-4 and IEMOCAP-6 (clips_v2, five folds,
# validation and test) -> M3ED validation/test.  Resumable: finished caches are skipped.  Usage: paper2_oppo_queue_20261022.sh "G0,G1"
set -euo pipefail
GPUS=$1; G0=${GPUS%%,*}; G1=${GPUS##*,}
ROOT=${EMOZOOM_ROOT:-$(pwd)}; PY=${PYTHON:-python}; Z=$ROOT/work_dir/oppo_v1; LOG=$ROOT/logs/oppo; Q=$ROOT/work_dir/iemocap_qwen_v1
mkdir -p "$Z/protocols" "$Z/meld/dev" "$Z/meld/test" "$Z/m3ed" "$LOG"; cd "$ROOT"
ts(){ TZ=Asia/Shanghai date '+%F %T'; }
OPPO=$ROOT/model/oppo-7b
$PY - "$Z" "$OPPO" <<'PY'
import json, sys
from pathlib import Path
Z, M = Path(sys.argv[1]), sys.argv[2]; W = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "/work_dir"))
def fix(src, dst):
    d = json.loads(Path(src).read_text()); n = 0
    for t in (d.get("training"), d.get("f0", {}).get("training")):
        if t: t["model"] = M; n += 1
    assert n >= 1, src
    d["oppo_baseline"] = {"ssot": "10.18.351", "derived_from": str(src), "changed": ["training.model -> model/oppo-7b"]}
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists(): dst.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
fix(W / "meld_etr_20260914_v2/protocol_official_dev.json", Z / "protocols/meld_dev.json")
fix(W / "meld_emp_devselect_test_20260916/protocol_official_test.json", Z / "protocols/meld_test.json")
fix(W / "m3ed_qwen_at_v1/protocol_v1/protocol.json", Z / "protocols/m3ed_validation.json")
fix(W / "m3ed_qwen_at_v1/evaluation_v1/test_protocol.json", Z / "protocols/m3ed_test.json")
for run, out in (("SESSION_FIVEFOLD_CLIPFIX_20261010", "SESSION_FIVEFOLD_OPPO_CLIPFIX_20261022"), ("SESSION6_FIVEFOLD_CLIPFIX_20261010", "SESSION6_FIVEFOLD_OPPO_CLIPFIX_20261022")):
    for k in range(1, 6): fix(W / f"iemocap_qwen_v1/{run}/fold{k}/protocol.json", W / f"iemocap_qwen_v1/{out}/fold{k}/protocol.json")
print("protocols ready")
PY
score(){ # protocol out work
  $PY experiments/run_meld_qwen_sft_selection_eval.py --protocol "$1" --arm zero_shot --output-cache "$2" --work-dir "$3" \
    --num-gpus 2 --gpu-ids $GPUS --cpu-threads 4 --resume >> "$LOG/$(basename "$(dirname "$2")")_$(basename "$2" .jsonl).log" 2>&1
}
[ -f "$Z/meld/dev/oppo.jsonl.done" ] || { echo "[$(ts)] MELD dev"; score $Z/protocols/meld_dev.json "$Z/meld/dev/oppo.jsonl" "$Z/meld/dev/work"; touch "$Z/meld/dev/oppo.jsonl.done"; }
if [ ! -f "$Z/meld/agreement.json" ]; then
  echo "[$(ts)] stop-rule check"; CUDA_VISIBLE_DEVICES=$G0 $PY experiments/paper2_oppo_agreement_20261022.py $Z/protocols/meld_dev.json "$Z/meld/dev/oppo.jsonl" "$Z/meld/agreement.json" 300 > "$LOG/agreement.log" 2>&1 || true
fi
grep -q '"passed": true' "$Z/meld/agreement.json" || { echo "[$(ts)] STOP RULE FAILED or check errored; see $LOG/agreement.log"; exit 1; }
# OPPO_DEFER_MELD_TEST=1 skips MELD test for now (two very long videos OOM on one GPU; SSOT 10.18.353)
[ -f "$Z/meld/test/oppo.jsonl.done" ] || [ -n "${OPPO_DEFER_MELD_TEST:-}" ] || { echo "[$(ts)] MELD test"; score $Z/protocols/meld_test.json "$Z/meld/test/oppo.jsonl" "$Z/meld/test/work"; touch "$Z/meld/test/oppo.jsonl.done"; }
for TARGET in iemocap4 iemocap6; do
  if [ "$TARGET" = iemocap4 ]; then RUN=$Q/SESSION_FIVEFOLD_CLIPFIX_20261010; OUT=$Q/SESSION_FIVEFOLD_OPPO_CLIPFIX_20261022; LABELS=angry,happy,neutral,sad
  else RUN=$Q/SESSION6_FIVEFOLD_CLIPFIX_20261010; OUT=$Q/SESSION6_FIVEFOLD_OPPO_CLIPFIX_20261022; LABELS=angry,happy,excited,neutral,sad,frustrated; fi
  for fold in 5 1 2 3 4; do
    D=$RUN/fold$fold; P=$OUT/fold$fold/protocol.json; O=$OUT/fold$fold/control; mkdir -p "$O"
    for split in validation test; do
      final=$O/$([ $split = test ] && echo evaluation.jsonl || echo validation_predictions.jsonl); [ -f "$final" ] && continue
      T=$O/${split}_shards; mkdir -p "$T"   # resumable: eval_c6_states skips successful rows
      echo "[$(ts)] $TARGET fold $fold $split"
      i=0
      for g in $G0 $G1; do
        CUDA_VISIBLE_DEVICES=$g $PY $D/source/eval_c6_states.py --protocol $P --manifest $D/${split}_manifest.jsonl --base-only \
          --labels $LABELS --states A,V,T,AV,AT,VT,AVT --shard $i --num-shards 2 --output $T/shard-$i.jsonl \
          >> "$LOG/${TARGET}_fold${fold}_${split}_$i.log" 2>&1 &
        i=$((i + 1)); sleep 120   # stagger 7B model loading (31 GB RAM)
      done
      wait
      $PY - "$T" "$D/${split}_manifest.jsonl" "$final" <<'PY'
import json, sys
from pathlib import Path
T, man, final = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = [json.loads(l) for f in sorted(T.glob("shard-*.jsonl")) for l in open(f) if l.strip()]
ok = [r for r in rows if r.get("status") == "success"]
exp = {json.loads(l)["sample_id"] for l in open(man) if l.strip()}
keys = [(r["sample_id"], s) for r in ok for s in r.get("states", {})]
assert len(keys) == len(set(keys)) == 7 * len(exp) and {k[0] for k in keys} == exp, f"key gate failed: {len(keys)} keys for {len(exp)} samples"
with open(final, "x", encoding="utf-8") as f:
    for r in sorted(ok, key=lambda r: r["sample_id"]): f.write(json.dumps(r) + "\n")
print("merged", final, len(ok))
PY
    done
  done
  touch "$OUT/COMPLETE"
done
[ -f "$Z/m3ed/validation.jsonl.done" ] || { echo "[$(ts)] M3ED validation"; score $Z/protocols/m3ed_validation.json "$Z/m3ed/validation.jsonl" "$Z/m3ed/work_val"; touch "$Z/m3ed/validation.jsonl.done"; }
[ -f "$Z/m3ed/test.jsonl.done" ] || { echo "[$(ts)] M3ED test"; score $Z/protocols/m3ed_test.json "$Z/m3ed/test.jsonl" "$Z/m3ed/work_test"; touch "$Z/m3ed/test.jsonl.done"; }
[ -f "$Z/meld/test/oppo.jsonl.done" ] || { echo "[$(ts)] all scored except the deferred MELD test"; exit 0; }
touch "$Z/SCORING_COMPLETE"; echo "[$(ts)] OPPO scoring complete (decoding is run separately)"
