#!/usr/bin/env python3
"""Per-state evaluation for C6 arms.

Answers three things from one forward per (sample, state):

  1. official metric  -- argmax over full-label candidate sequence scores, the
     same scoring the frozen evaluator uses;
  2. surrogate alignment -- argmax over the 7 label-initial-token logits, i.e.
     exactly the quantity the C6 training loss optimises.  If these two disagree
     often, the training surrogate is not the evaluated decision;
  3. absolute GT margin m(x) = z_y - max_{c!=y} z_c, per state, so a contribution
     arm that "wins" by depressing the incomplete-state margin is visible.
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
LABELS = ["happy", "sad", "neutral", "angry", "surprise", "disgust", "fear"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, default=None,
                    help="omit to evaluate the frozen base F0 adapter in the protocol")
    ap.add_argument("--base-only", action="store_true",
                    help="load the base checkpoint with no PEFT at all (true Raw Qwen); used "
                         "when the LoRA-stripped wrapper fails the equivalence gate")
    ap.add_argument("--states", default="VT,AT,AV,AVT")
    ap.add_argument("--labels", default="",
                    help="comma-separated label set; empty keeps MELD's seven emotions. "
                         "IEMOCAP-4 uses angry,happy,neutral,sad")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--sample-ids", default="",
                    help="comma-separated exact sample IDs to repair; original manifest is retained")
    ap.add_argument("--record-provenance", action="store_true",
                    help="require a new output and save input/adapter/code hashes before inference")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--surrogate-only", action="store_true",
                    help="skip the official candidate scorer; emit only the differentiable "
                         "first-token surrogate margin (used to build frozen-F0 anchors)")
    args = ap.parse_args()
    if args.record_provenance:
        if args.output.exists():
            raise SystemExit("provenance-recorded evaluation requires a new output path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        provenance = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      "protocol_sha256": sha256_file(args.protocol),
                      "manifest_sha256": sha256_file(args.manifest),
                      "evaluator_sha256": sha256_file(Path(__file__)),
                      "adapter_files": {str(p): sha256_file(p) for p in sorted(args.adapter.glob("*"))
                                        if p.is_file()} if args.adapter else {},
                      "started_unix": time.time()}
        with Path(str(args.output) + ".started.json").open("x") as stream:
            json.dump(provenance, stream, indent=2)

    from run_meld_qwen_lora_sft import (patch_torch_compat, prepare_training_input, move_inputs,
                                        decode_audio, decode_video)
    from run_meld_qwen_sft_selection_eval import prepare_view
    from run_meld_c1_oof import load_model
    from run_planb_meld_qwen_pilot import cached_sequence_scores
    patch_torch_compat()

    global LABELS
    if args.labels:
        from label_set import apply_label_set
        LABELS, touched = apply_label_set([x for x in args.labels.split(",") if x])
        print(f"[eval] label set = {LABELS}  (rebound: {len(touched)} bindings)", flush=True)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    runtime = dict(protocol)
    runtime["training"] = protocol["f0"]["training"]
    adapter = args.adapter if args.adapter else \
        Path("work_dir/meld_c1_raw_asr_gate_v1/f0/adapter")
    if args.base_only:
        model, processor, _ = load_model(runtime, "zero_shot", None)
    else:
        model, processor, _ = load_model(runtime, "f0", Path(adapter).resolve())
    model.eval()
    device = next(model.parameters()).device
    tok = processor.tokenizer
    first_ids = torch.tensor([tok(l, add_special_tokens=False).input_ids[0] for l in LABELS],
                             device=device)
    cand_ids = [tok(l, add_special_tokens=False).input_ids for l in LABELS]

    source_protocol = json.loads(
        Path(protocol["source"]["platform_protocol"]).read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    if args.sample_ids:
        selected = set(args.sample_ids.split(","))
        if selected - {r["sample_id"] for r in rows}:
            raise SystemExit("requested sample not in the original manifest")
        rows = [r for r in rows if r["sample_id"] in selected]
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    if args.max_samples:
        rows = rows[: args.max_samples]
    states = [s for s in args.states.split(",") if s]
    vp = int(runtime["training"]["video_total_pixels"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        # resume must skip only SUCCESSFUL rows; an error row (e.g. a transient
        # CUDA OOM) previously counted as "done" and silently left holes in the
        # anchor file, which then crashed anchored training on the first miss.
        kept = []
        for line in args.output.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "success":
                done.add(r["sample_id"]); kept.append(line)
        args.output.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    started = time.time()
    with args.output.open("a", encoding="utf-8", buffering=1) as out:
        for n, r in enumerate(rows, 1):
            if r["sample_id"] in done:
                continue
            rec = {"sample_id": r["sample_id"], "split": r.get("split"),
                   "dialogue_id": int(r["dialogue_id"]), "gt": r["gt"], "states": {}}
            try:
                y = LABELS.index(r["gt"])
                video_path = Path(r["video_path"])
                waveform = decode_audio(video_path)
                video = decode_video(video_path, vp)
                for st in states:
                    # --- path 1: the training surrogate (first-token class logits) ---
                    inputs, _ = prepare_training_input(processor, dict(r), st, vp)
                    inputs = move_inputs(inputs, device)
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        o = model(**inputs, use_cache=False, use_audio_in_video=False)
                    logits = o.logits.float()
                    sup = (inputs["labels"][0] != -100).nonzero(as_tuple=True)[0]
                    pos = int(sup[0].item())
                    z = logits[0, pos - 1, :].index_select(0, first_ids)
                    other = z.clone(); other[y] = -1e30
                    margin = float(z[y] - other.max())
                    surrogate_pred = LABELS[int(z.argmax())]
                    del o, inputs, logits

                    if args.surrogate_only:
                        rec["states"][st] = {"margin": margin, "surrogate_pred": surrogate_pred,
                                             "class_logits": [float(v) for v in z]}
                        continue
                    # --- path 2: the exact frozen evaluator (autoregressive candidate sum) ---
                    prepared, prov = prepare_view(source_protocol, processor, dict(r), st,
                                                  waveform, video)
                    scores_d, _ = cached_sequence_scores(model, prepared,
                                                         prov["candidate_token_ids"])
                    official_pred = max(LABELS, key=scores_d.__getitem__)
                    official_margin = scores_d[LABELS[y]] - max(
                        scores_d[l] for l in LABELS if l != LABELS[y])
                    rec["states"][st] = {"margin": margin, "surrogate_pred": surrogate_pred,
                                         "official_pred": official_pred,
                                         "official_margin": float(official_margin),
                                         "class_logits": [float(v) for v in z],
                                         "candidate_scores": {k: float(v) for k, v in scores_d.items()}}
                rec["status"] = "success"
            except Exception as exc:  # noqa: BLE001
                rec = {"sample_id": r["sample_id"], "status": "error", "error": repr(exc)}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n % 25 == 0:
                print(f"[eval shard {args.shard}] {n}/{len(rows)} {time.time()-started:.0f}s", flush=True)
    print(f"[eval shard {args.shard}] DONE", flush=True)


if __name__ == "__main__":
    main()
