#!/usr/bin/env python3
"""Do the two speedups change the numbers?

Decoding a sample once per unit instead of once per state, and skipping
activation checkpointing, should both be numerically inert. Neither is worth
having if it silently shifts a loss, so this compares the tensors a training
step actually consumes and produces, on the same samples in the same order.

Checkpointing recomputes activations in a different order, so bf16 backward can
differ in the last bits; the loss comparison therefore allows a small tolerance
while the input comparison must be exact.
"""
import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

STATES = ["A", "V", "T", "AV", "AT", "VT", "AVT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--labels", default="")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--loss-tol", type=float, default=2e-2)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from run_meld_qwen_lora_sft import (patch_torch_compat, prepare_training_input,
                                        move_inputs, build_model, decode_audio, decode_video)
    patch_torch_compat()
    if args.labels:
        from label_set import apply_label_set
        apply_label_set([x for x in args.labels.split(",") if x])

    protocol = json.loads(Path(args.protocol).read_text())
    runtime = dict(protocol)
    if "training" not in runtime:
        runtime["training"] = protocol["f0"]["training"]
    vp = int(runtime["training"]["video_total_pixels"])
    rows = [json.loads(l) for l in open(args.manifest) if l.strip()][: args.n]
    device = torch.device("cuda", 0)

    report = {"n_samples": len(rows), "states": STATES,
              "input_mismatches": [], "loss_deltas": []}

    # ---- part 1: decode reuse must produce byte-identical model inputs -----
    _, processor, _n, _t, _tot = build_model(runtime, device, "flash_attention_2")
    for r in rows:
        smp = dict(r)
        media = Path(r["video_path"])
        decoded = (decode_audio(media), decode_video(media, vp))
        for st in STATES:
            a, _ = prepare_training_input(processor, smp, st, vp)                 # decode inside
            b, _ = prepare_training_input(processor, smp, st, vp, decoded=decoded)  # decode reused
            if set(a) != set(b):
                report["input_mismatches"].append([r["sample_id"], st, "different keys"])
                continue
            for k in a:
                x, y = a[k], b[k]
                if isinstance(x, torch.Tensor):
                    same = x.shape == y.shape and torch.equal(x, y)
                else:
                    same = x == y
                if not same:
                    report["input_mismatches"].append([r["sample_id"], st, k])
    report["decode_reuse_identical"] = not report["input_mismatches"]
    print(f"decode reuse: {'byte-identical inputs' if report['decode_reuse_identical'] else 'MISMATCH'}"
          f" over {len(rows)} samples x {len(STATES)} states", flush=True)

    # ---- part 2: checkpointing on vs off, same forward, same loss ----------
    losses = {}
    for ckpt in (True, False):
        model, processor, _n, _t, _tot = build_model(runtime, device, "flash_attention_2",
                                                     gradient_checkpointing=ckpt)
        if args.adapter:
            from safetensors.torch import load_file
            import glob as _g
            sd = {}
            for f in _g.glob(str(Path(args.adapter) / "*.safetensors")):
                sd.update(load_file(f))
            own = dict(model.named_parameters())
            for k, v in sd.items():
                cand = k if k in own else k.replace(".weight", ".default.weight")
                if cand in own and own[cand].shape == v.shape:
                    own[cand].data.copy_(v.to(own[cand].dtype).to(own[cand].device))
        model.train()
        vals = []
        for r in rows:
            smp = dict(r)
            media = Path(r["video_path"])
            decoded = (decode_audio(media), decode_video(media, vp))
            for st in STATES:
                inputs, _ = prepare_training_input(processor, smp, st, vp, decoded=decoded)
                inputs = move_inputs(inputs, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(**inputs, use_cache=False, use_audio_in_video=False)
                vals.append(float(out.loss))
                del out, inputs
        losses[ckpt] = vals
        del model
        torch.cuda.empty_cache()
    d = [abs(x - y) for x, y in zip(losses[True], losses[False])]
    rel = [abs(x - y) / max(abs(x), 1e-6) for x, y in zip(losses[True], losses[False])]
    report["loss_deltas"] = {"n": len(d), "max_abs": max(d), "mean_abs": sum(d) / len(d),
                             "max_rel": max(rel), "tolerance": args.loss_tol}
    report["checkpointing_equivalent"] = max(rel) <= args.loss_tol
    print(f"checkpointing on vs off: max |dloss| = {max(d):.3e}, max relative = {max(rel):.3e}"
          f"  (tolerance {args.loss_tol})")

    report["verdict"] = ("BOTH SPEEDUPS SAFE" if report["decode_reuse_identical"]
                         and report["checkpointing_equivalent"] else "NOT EQUIVALENT")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"\nVERDICT: {report['verdict']}\nwrote {args.output}")
    sys.exit(0 if report["verdict"] == "BOTH SPEEDUPS SAFE" else 3)


if __name__ == "__main__":
    main()
