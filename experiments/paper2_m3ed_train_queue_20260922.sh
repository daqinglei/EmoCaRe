#!/bin/bash
set -euo pipefail
ROOT="${EMOZOOM_ROOT:-$(pwd)}"
PY="${PYTHON:-python}"
TORCHRUN="$PY -m torch.distributed.run"
PROTO=$ROOT/work_dir/m3ed_qwen_at_v1/protocol_v1/protocol.json
RUN=$ROOT/work_dir/m3ed_qwen_at_v1/emitter_v1
LOG=$ROOT/logs/m3ed
cd "$ROOT"
mkdir -p "$RUN" "$LOG"
while [ ! -f data/M3ED/media/AUDIO_DOWNLOAD_COMPLETE ]; do sleep 30; done
$PY experiments/paper2_build_m3ed_at_protocol_20260922.py > "$LOG/protocol_build_before_train.log" 2>&1
$PY - <<'PY'
import json
from pathlib import Path
p=Path('work_dir/m3ed_qwen_at_v1/protocol_v1/protocol.json')
d=json.loads(p.read_text())
for split in ('train','validation','test'):
 if d['splits'][split]['audio_members_resolved'] != d['splits'][split]['utterances']:
  raise SystemExit(f"unresolved audio in {split}: {d['splits'][split]}")
 for line in Path(f'work_dir/m3ed_qwen_at_v1/protocol_v1/{split}_manifest.jsonl').open():
  r=json.loads(line)
  if not Path(r['video_path']).is_file(): raise SystemExit(f"missing media: {r['video_path']}")
print('M3ED_MEDIA_PREFLIGHT_PASS')
PY
# Avoid colliding with any job that entered the four GPUs while the archive downloaded.
while /usr/bin/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000{bad=1} END{exit bad?0:1}'; do sleep 60; done
rm -rf "$RUN/smoke"
CUDA_VISIBLE_DEVICES=0 $PY experiments/run_meld_qwen_lora_sft.py \
  --protocol "$PROTO" --arm seven_state_sft --output-dir "$RUN/smoke" \
  --smoke --max-microsteps 2 --attn-implementation flash_attention_2 \
  > "$LOG/emitter_smoke.log" 2>&1
rm -rf "$RUN/formal"
CUDA_VISIBLE_DEVICES=0,1,2,3 $TORCHRUN --standalone --nproc_per_node=4 --no-python \
  "$PY" experiments/run_meld_qwen_lora_sft.py --protocol "$PROTO" \
  --arm seven_state_sft --output-dir "$RUN/formal" \
  --checkpoint-every 250 --log-every 25 --attn-implementation flash_attention_2 \
  > "$LOG/emitter_formal.log" 2>&1
touch "$RUN/TRAINING_COMPLETE"
