#!/usr/bin/env python3
"""C6-v0: Balanced Counterfactual Evidence Optimization.

Diagnosis being acted on: on the frozen F0, adding real extra modality evidence
raises WAR but degrades UAR/Macro-F1 -- an evidence-induced majority attraction.
Instead of predicting after the fact whether evidence will hurt (C1), this trains
the model so that adding evidence does not reduce the ground-truth decision margin.

For a paired unit (x-, x+) where x+ has one more true modality than x-:

    m(x) = z_y(x) - max_{c != y} z_c(x)              (7-way class logits)
    L_contrib = relu( m(x-) - m(x+) + delta )
    L = L_task + lam_cb * L_cb + lam_contrib * L_contrib

z is read differentiably from the logit position that predicts the first answer
token, gathered over the 7 label-initial token ids (verified unique).
Ground truth is used only as training supervision; inference is unchanged.

Arms (matched compute, identical pair order/initialisation):
    task_only | cb_only | contrib_only | cb_contrib(proposed)
"""
from __future__ import annotations

import argparse, hashlib, json, math, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

LABELS = ["happy", "sad", "neutral", "angry", "surprise", "disgust", "fear"]
PAIRS = [("VT", "AVT"), ("AT", "AVT"), ("AV", "AVT")]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def adapter_digest(model) -> str:
    h = hashlib.sha256()
    for n, p in sorted((n, p) for n, p in model.named_parameters() if p.requires_grad):
        h.update(n.encode()); h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def class_logits(logits: torch.Tensor, labels: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
    """Differentiable 7-way logits at the first supervised answer position."""
    sup = (labels[0] != -100).nonzero(as_tuple=True)[0]
    pos = int(sup[0].item())
    return logits[0, pos - 1, :].index_select(0, label_ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--arm", required=True,
                    choices=["task_only", "cb_only", "contrib_only", "cb_contrib"])
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--anchor", default="symmetric", choices=["symmetric", "stopgrad", "frozen"],
                    help="symmetric: L=relu(m_neg-m_pos+delta) -- gradient also lowers m_neg "
                         "(shortcut); stopgrad: detach m_neg; frozen: anchor on the frozen-F0 "
                         "incomplete margin supplied by --frozen-margins")
    ap.add_argument("--frozen-margins", type=Path, default=None)
    ap.add_argument("--lam-cb", type=float, default=1.0)
    ap.add_argument("--lam-contrib", type=float, default=1.0)
    ap.add_argument("--lam-kd", type=float, default=0.0,
                    help="C6-v2 option A: KL(p_F0(.|x-) || p_theta(.|x-)) on the incomplete "
                         "state, a stronger preservation signal than the margin hinge because "
                         "it constrains the whole distribution, not just the top-2 gap")
    ap.add_argument("--frozen-logits", type=Path, default=None,
                    help="jsonl with {sample_id, state, class_logits} from the frozen F0")
    ap.add_argument("--lam-preserve", type=float, default=0.0,
                    help="weight of L_preserve = relu(m_neg_F0 - m_neg_theta): explicitly "
                         "forbids buying evidence-added margin by degrading incomplete competence")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--state-balanced", type=int, default=1,
                    help="1 = one unit per sample covering ALL FOUR states with equal task "
                         "weight (VT:AT:AV:AVT = 1:1:1:1) and the three contribution / preserve "
                         "constraints averaged.  0 = legacy single-pair units, which give AVT "
                         "3x the task-gradient exposure of each incomplete state.")
    ap.add_argument("--incomplete-states", default="VT,AT,AV",
                    help="incomplete states carried by every unit.  The four-state C6 used "
                         "VT,AT,AV; the seven-state version adds the singletons A,V,T, which "
                         "the missing-rate audit showed are where severe-missing performance "
                         "is lost because they were never trained.")
    ap.add_argument("--contrib-states", default="",
                    help="subset of --incomplete-states used as x- anchors in L_contrib "
                         "(default: all of them).  Kept separate so singletons can enter "
                         "task/preserve/KD without forcing a monotonic contribution on every "
                         "lattice edge -- that full-lattice form is SimMLM MoFe, which is "
                         "under TERMINAL STOP.")
    ap.add_argument("--pos-weight", type=float, default=0.0,
                    help="task-loss weight of the complete state AVT.  0 = uniform 1/len(states) "
                         "(pure state balance).  A positive value pins AVT at that weight and "
                         "spreads the remainder over the incomplete states, which holds AVT's "
                         "task gradient identical to the four-state run so a seven-state gain "
                         "cannot be attributed to AVT being down-weighted.")
    ap.add_argument("--cw-on-incomplete", type=int, default=0,
                    help="1 = class-weight the CE on incomplete states only, leaving the "
                         "complete state unweighted.  With --pos-weight 0.5 and "
                         "--incomplete-states AV,A this reproduces the stage-1 F0 loss "
                         "0.5*CE(AVT) + 0.5*class-weighted CE(missing), so a second corpus "
                         "can build its own F0 through this same trainer.")
    ap.add_argument("--report-memory", action="store_true",
                    help="log torch.cuda.max_memory_allocated / reserved per step. nvidia-smi "
                         "shows the caching allocator's footprint, not what the step actually "
                         "needed, which is what a checkpointing decision turns on.")
    ap.add_argument("--reuse-decoded-media", action="store_true",
                    help="decode each sample's audio and video once per unit instead of once "
                         "per state. Off by default so the default path stays identical to "
                         "every arm trained before 2026-09-08; enable only for a whole new "
                         "comparison, never mid-curve.")
    ap.add_argument("--no-gradient-checkpointing", action="store_true",
                    help="skip activation checkpointing. It costs about a third of the step "
                         "time and buys memory that is not scarce here; results should be "
                         "equivalent, so a run that uses it must pass the equivalence check "
                         "in experiments/test_speedup_equivalence.py first.")
    ap.add_argument("--init-adapter", default="",
                    help="start stage 2 from these LoRA weights instead of a fresh adapter. "
                         "Without it the student begins at the base checkpoint (lora_B is "
                         "zero, so delta-W is zero) and only meets F0 through the anchor and "
                         "KD terms -- which is what every run before 2026-09-07 did.")
    ap.add_argument("--require-init-sha", default="",
                    help="abort unless the initial adapter digest equals this. Paired arms "
                         "pass each other's digest so a matched comparison cannot silently "
                         "differ by its random initialisation as well as by its loss.")
    ap.add_argument("--init-gate-reference", type=Path,
                    help="audited F0 seven-state scores on first 12 training samples; checks before updates")
    ap.add_argument("--preflight-only", action="store_true",
                    help="run actual initialization and step-0 gate, then exit without optimizer updates")
    ap.add_argument("--save-init-only", action="store_true",
                    help="save the seeded fresh/shared LoRA initialization and exit before reading "
                         "teacher caches or constructing an optimizer")
    ap.add_argument("--labels", default="",
                    help="comma-separated label set; empty keeps MELD's seven emotions")
    ap.add_argument("--max-units", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="save LoRA checkpoints every N optimizer steps; 0 keeps legacy final-only behavior")
    ap.add_argument("--wandb-mode", default="disabled", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb-project", default="EmoZoom-Paper2")
    ap.add_argument("--wandb-entity", default="")
    ap.add_argument("--wandb-name", default="")
    args = ap.parse_args()
    if args.preflight_only and not args.init_gate_reference:
        raise SystemExit("--preflight-only requires --init-gate-reference")
    if args.init_gate_reference and args.output_dir.exists():
        raise SystemExit("audited runs require a new output directory; existing artifacts are immutable")

    global LABELS
    if args.labels:
        from label_set import apply_label_set
        LABELS, _ = apply_label_set([x for x in args.labels.split(",") if x])
        print(f"[train] label set = {LABELS}", flush=True)

    incomplete_states = [x for x in args.incomplete_states.split(",") if x]
    contrib_states = [x for x in args.contrib_states.split(",") if x] or list(incomplete_states)
    unknown = set(contrib_states) - set(incomplete_states)
    if unknown:
        raise SystemExit(f"--contrib-states {sorted(unknown)} are not in --incomplete-states")

    from run_meld_qwen_lora_sft import (patch_torch_compat, prepare_training_input, move_inputs,
                                        build_model, decode_audio, decode_video)
    patch_torch_compat()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    runtime = dict(protocol)
    if "training" not in runtime:                      # C1 protocol nests it under f0
        runtime["training"] = protocol["f0"]["training"]
    device = torch.device("cuda", 0)   # CUDA_VISIBLE_DEVICES pins the physical GPU
    # seed torch BEFORE the adapter is created: lora_A is drawn randomly inside
    # get_peft_model, so without this every arm starts from a different point and a
    # "matched" pair differs by its initialisation as well as by its loss
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, processor, names, trainable, total = build_model(
        runtime, device, "flash_attention_2",
        gradient_checkpointing=not args.no_gradient_checkpointing)

    if args.init_adapter:
        from safetensors.torch import load_file
        src = Path(args.init_adapter)
        sd = {}
        for f in sorted(src.glob("*.safetensors")):
            sd.update(load_file(f))
        own = dict(model.named_parameters())
        loaded = 0
        for k, v in sd.items():
            cand = k if k in own else k.replace(".weight", ".default.weight")
            if cand in own and own[cand].shape == v.shape:
                own[cand].data.copy_(v.to(own[cand].dtype).to(own[cand].device))
                loaded += 1
        if loaded != len(sd) or loaded != len(names):
            raise SystemExit(f"--init-adapter {src}: incomplete load {loaded}/{len(sd)}; expected {len(names)} tensors")
        print(f"[init] loaded {loaded}/{len(sd)} tensors from {src}", flush=True)

    initial = adapter_digest(model)
    print(f"[init] initial_adapter_sha256 = {initial}", flush=True)
    if args.require_init_sha and args.require_init_sha != initial:
        raise SystemExit(f"initial adapter digest {initial} != required {args.require_init_sha}; "
                         "the paired arms would not share an initialisation")
    # persist the exact starting weights so a paired arm can reuse them byte for byte
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir / "initial_adapter", safe_serialization=True)
    if args.save_init_only:
        init_manifest = {
            "status": "INITIALIZATION_ONLY",
            "seed": args.seed,
            "initial_adapter_sha256": initial,
            "base_function_expected": True,
            "optimizer_updates": 0,
            "protocol_sha256": sha256_file(args.protocol),
            "manifest_sha256": sha256_file(args.manifest),
            "trainer_sha256": sha256_file(Path(__file__)),
        }
        (args.output_dir / "initialization_manifest.json").write_text(
            json.dumps(init_manifest, indent=2, ensure_ascii=False)
        )
        print(json.dumps(init_manifest, ensure_ascii=False), flush=True)
        return
    tok = processor.tokenizer
    label_ids = torch.tensor([tok(l, add_special_tokens=False).input_ids[0] for l in LABELS],
                             device=device)
    assert len(set(label_ids.tolist())) == len(LABELS), "label first tokens not unique"
    if args.init_gate_reference:
        from c6_training_preflight import preflight
        preflight(args, model, processor, runtime, initial, LABELS)
        if args.preflight_only:
            print("[preflight] completed with zero optimizer updates", flush=True)
            return

    frozen_logits: dict[str, list[float]] = {}
    if args.lam_kd > 0:
        if not args.frozen_logits or not args.frozen_logits.exists():
            raise SystemExit("--lam-kd requires --frozen-logits")
        for line in open(args.frozen_logits, encoding="utf-8"):
            rec = json.loads(line)
            frozen_logits[f"{rec['sample_id']}|{rec['state']}"] = rec["class_logits"]

    frozen_margins: dict[str, float] = {}
    if args.anchor == "frozen":
        if not args.frozen_margins or not args.frozen_margins.exists():
            raise SystemExit("--anchor frozen requires --frozen-margins")
        for line in open(args.frozen_margins, encoding="utf-8"):
            rec = json.loads(line)
            frozen_margins[f"{rec['sample_id']}|{rec['state']}"] = float(rec["margin"])

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    rng = np.random.default_rng(args.seed)
    # one deterministic pair type per sample -> balanced coverage, bounded compute
    units = []
    for i, r in enumerate(rows):
        if args.state_balanced:
            # every sample carries all incomplete states once, so no state is over-trained
            units.append({"row": r, "negs": list(incomplete_states), "pos": "AVT"})
        else:
            neg, pos = PAIRS[i % len(PAIRS)]
            units.append({"row": r, "negs": [neg], "pos": pos})
    order = rng.permutation(len(units)).tolist() * args.epochs
    if args.max_units:
        order = order[: args.max_units]
    order = order[: (len(order) // args.grad_accum) * args.grad_accum] or order
    steps_expected = math.ceil(len(order) / args.grad_accum)

    counts = np.zeros(len(LABELS))
    for u in units:
        counts[LABELS.index(u["row"]["gt"])] += 1
    cw = torch.tensor(np.where(counts > 0, counts.sum() / (len(LABELS) * np.maximum(counts, 1)), 0.0),
                      dtype=torch.float32, device=device)

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / max(1, steps_expected)))
    model.train(); opt.zero_grad(set_to_none=True)
    ctr = {"units": 0, "forward_count": 0, "optimizer_steps": 0, "supervised_tokens": 0}
    hist = {"task": [], "cb": [], "contrib": [], "margin_gain": [], "m_neg": [], "m_pos": [], "preserve": [], "kd": []}
    started = time.time()
    vp = int(runtime["training"]["video_total_pixels"])
    history_file = (args.output_dir / "history.jsonl").open("x", buffering=1)
    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb
        wandb_dir = args.output_dir / "wandb"
        wandb_dir.mkdir()
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_name or f"{args.arm}-{args.output_dir.name}",
            dir=str(wandb_dir),
            mode=args.wandb_mode,
            config={
                "arm": args.arm, "seed": args.seed, "epochs": args.epochs,
                "steps": steps_expected, "lr": args.lr, "grad_accum": args.grad_accum,
                "anchor": args.anchor, "delta": args.delta,
                "lam_contrib": args.lam_contrib, "lam_kd": args.lam_kd,
                "lam_preserve": args.lam_preserve, "incomplete_states": incomplete_states,
                "contrib_states": contrib_states, "pos_weight": args.pos_weight,
                "checkpoint_every": args.checkpoint_every,
                "initial_adapter_sha256": initial,
            },
            tags=["Paper2", "C6", args.arm, "fresh-student"],
        )
        (args.output_dir / "wandb_run.json").write_text(json.dumps({
            "id": wandb_run.id, "name": wandb_run.name,
            "project": wandb_run.project, "entity": wandb_run.entity,
            "url": wandb_run.url, "mode": args.wandb_mode,
        }, indent=2))

    for position, ui in enumerate(order):
        u = units[int(ui)]
        r = u["row"]; y = LABELS.index(r["gt"])
        yt = torch.tensor([y], device=device)
        # Memory-safe formulation.  With a frozen anchor every loss term is
        # separable per state -- contribution depends only on the AVT graph (the
        # anchor is a constant), KD / preserve only on their own incomplete-state
        # graph -- so each state is backwarded and freed immediately.  Gradients
        # are identical to accumulating all four graphs, but peak memory is one
        # forward instead of four, which is what OOMed the first v2 attempt.
        # A symmetric or stop-grad anchor needs both margins live in one graph, which
        # the sequential backward cannot provide.  With every anchored term switched
        # off there is no such term, so a pure task-loss stage (the stage-1 F0) is
        # allowed through.
        anchored = args.lam_contrib > 0 or args.lam_preserve > 0 or args.lam_kd > 0
        if args.anchor != "frozen" and anchored:
            raise SystemExit("sequential backward requires --anchor frozen "
                             "whenever a contribution / preserve / KD term is active")
        n_neg = len(u["negs"])
        state_list = [(f"neg:{st}", st) for st in u["negs"]] + [("pos", u["pos"])]
        if args.pos_weight > 0:
            w_pos = args.pos_weight
            w_neg = (1.0 - w_pos) / n_neg
        else:
            w_pos = w_neg = 1.0 / len(state_list)
        margins, parts = {}, {"task": 0.0, "cb": 0.0, "contrib": 0.0,
                              "preserve": 0.0, "kd": 0.0}
        # `r` is the manifest row; `smp` is only built inside the loop below
        decoded = None
        if args.reuse_decoded_media:
            media_path = Path(r["video_path"])
            decoded = (decode_audio(media_path), decode_video(media_path, vp))
        for tag, state in state_list:
            smp = dict(r); smp["transcript"] = r["transcript"]
            inputs, stats = prepare_training_input(processor, smp, state, vp, decoded=decoded)
            inputs = move_inputs(inputs, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**inputs, use_cache=False, use_audio_in_video=False)
            if out.loss is None or not torch.isfinite(out.loss):
                raise RuntimeError(f"non-finite loss {r['sample_id']} {state}")
            z = class_logits(out.logits.float(), inputs["labels"], label_ids)
            other = z.clone(); other[y] = -1e30
            m = z[y] - other.max()
            margins[tag] = float(m.detach())

            w_state = w_pos if tag == "pos" else w_neg
            if args.cw_on_incomplete and tag != "pos":
                # answer-only CE is already in out.loss; reweight it by the class weight
                # of this sample so incomplete states carry the F0 stage-1 balancing
                loss_s = w_state * cw[y] * out.loss
            else:
                loss_s = w_state * out.loss
            parts["task"] += float((w_state * out.loss).detach())
            if args.arm in ("cb_only", "cb_contrib"):
                cb_s = w_state * torch.nn.functional.cross_entropy(z[None, :], yt, weight=cw)
                loss_s = loss_s + args.lam_cb * cb_s
                parts["cb"] += float(cb_s.detach())

            if tag == "pos":
                if args.arm in ("contrib_only", "cb_contrib"):
                    c = torch.zeros((), device=device)
                    c_states = [st for st in u["negs"] if st in contrib_states]
                    for st2 in c_states:
                        key = f"{r['sample_id']}|{st2}"
                        if key not in frozen_margins:
                            raise KeyError(f"no frozen anchor margin for {key}")
                        a0 = torch.tensor(frozen_margins[key], device=device, dtype=m.dtype)
                        c = c + torch.relu(a0 - m + args.delta) / len(c_states)
                    loss_s = loss_s + args.lam_contrib * c
                    parts["contrib"] += float(c.detach())
            else:
                st2 = state
                if args.lam_preserve > 0:
                    key = f"{r['sample_id']}|{st2}"
                    if key not in frozen_margins:
                        raise KeyError(f"no frozen margin for preserve: {key}")
                    m0 = torch.tensor(frozen_margins[key], device=device, dtype=m.dtype)
                    pr_s = torch.relu(m0 - m) / n_neg
                    loss_s = loss_s + args.lam_preserve * pr_s
                    parts["preserve"] += float(pr_s.detach())
                if args.lam_kd > 0:
                    key = f"{r['sample_id']}|{st2}"
                    if key not in frozen_logits:
                        raise KeyError(f"no frozen logits for KD: {key}")
                    z0 = torch.tensor(frozen_logits[key], device=device, dtype=torch.float32)
                    kd_s = cw[y] * torch.nn.functional.kl_div(
                        torch.log_softmax(z.float(), dim=-1),
                        torch.softmax(z0, dim=-1), reduction="sum") / n_neg
                    loss_s = loss_s + args.lam_kd * kd_s
                    parts["kd"] += float(kd_s.detach())

            (loss_s / args.grad_accum).backward()
            ctr["forward_count"] += 1
            ctr["supervised_tokens"] += int(stats["supervised_tokens"])
            del out, inputs, z, other, m, loss_s
            torch.cuda.empty_cache() if (position % 200 == 0) else None

        if args.report_memory:
            ctr["peak_alloc_mib"] = max(ctr.get("peak_alloc_mib", 0),
                                        int(torch.cuda.max_memory_allocated() / 2**20))
            ctr["peak_reserved_mib"] = max(ctr.get("peak_reserved_mib", 0),
                                           int(torch.cuda.max_memory_reserved() / 2**20))
        mneg_mean = sum(margins[f"neg:{st}"] for st in u["negs"]) / n_neg
        hist["kd"].append(parts["kd"])
        hist["preserve"].append(parts["preserve"])
        hist["contrib"].append(parts["contrib"])
        hist["cb"].append(parts["cb"])
        hist["margin_gain"].append(margins["pos"] - mneg_mean)
        hist["m_neg"].append(mneg_mean)
        hist["m_pos"].append(margins["pos"])
        task_loss_val = parts["task"]
        hist["task"].append(task_loss_val)
        ctr["units"] += 1
        if ((position + 1) % args.grad_accum == 0) or (position + 1 == len(order)):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            ctr["optimizer_steps"] += 1
            window = min(args.grad_accum, len(hist["task"]))
            entry = {
                "arm": args.arm,
                "step": ctr["optimizer_steps"],
                "steps": steps_expected,
                "task": float(np.mean(hist["task"][-window:])),
                "cb": float(np.mean(hist["cb"][-window:])),
                "contrib": float(np.mean(hist["contrib"][-window:])),
                "kd": float(np.mean(hist["kd"][-window:])),
                "preserve": float(np.mean(hist["preserve"][-window:])),
                "margin_gain": float(np.mean(hist["margin_gain"][-window:])),
                "grad_norm": float(grad_norm),
                "learning_rate": opt.param_groups[0]["lr"],
                "elapsed": time.time() - started,
                "peak_alloc_MiB": torch.cuda.max_memory_allocated() / 2**20,
            }
            history_file.write(json.dumps(entry) + "\n")
            if wandb_run is not None:
                wandb_run.log({
                    "train/task_loss": entry["task"],
                    "train/contribution_loss": entry["contrib"],
                    "train/kd_loss": entry["kd"],
                    "train/preserve_loss": entry["preserve"],
                    "train/margin_gain": entry["margin_gain"],
                    "train/grad_norm": entry["grad_norm"],
                    "train/learning_rate": entry["learning_rate"],
                    "train/elapsed_seconds": entry["elapsed"],
                    "system/peak_alloc_MiB": entry["peak_alloc_MiB"],
                }, step=ctr["optimizer_steps"])
            if args.checkpoint_every and (
                    ctr["optimizer_steps"] % args.checkpoint_every == 0
                    or ctr["optimizer_steps"] == steps_expected):
                checkpoint = args.output_dir / f"checkpoint-{ctr['optimizer_steps']}"
                model.save_pretrained(checkpoint, safe_serialization=True)
                (args.output_dir / f"checkpoint-{ctr['optimizer_steps']}.json").write_text(
                    json.dumps({
                        "step": ctr["optimizer_steps"],
                        "adapter_sha256": adapter_digest(model),
                        "initial_adapter_sha256": initial,
                    }, indent=2)
                )
            if ctr["optimizer_steps"] % args.log_every == 0:
                print(json.dumps(entry), flush=True)

    history_file.close()

    outdir = args.output_dir.resolve(); outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(outdir / "adapter", safe_serialization=True)
    processor.save_pretrained(outdir / "processor")
    manifest = {"status": "COMPLETE", "protocol": "meld_c6_balanced_evidence_v0", "arm": args.arm,
                "seed": args.seed, "epochs": args.epochs, "delta": args.delta,
                "lam_cb": args.lam_cb, "lam_contrib": args.lam_contrib,
                "initial_adapter_sha256": initial, "adapter_sha256": adapter_digest(model),
                "trainable_parameter_count": trainable, "counters": ctr,
                "mean_task": float(np.mean(hist["task"])), "mean_cb": float(np.mean(hist["cb"])),
                "mean_contrib": float(np.mean(hist["contrib"])),
                "mean_preserve": float(np.mean(hist["preserve"])),
                "lam_preserve": args.lam_preserve,
                "gradient_checkpointing": not args.no_gradient_checkpointing,
                "decode_reuse": args.reuse_decoded_media,
                "init_adapter": args.init_adapter or "fresh (base checkpoint)",
                "torch_seed": args.seed,
                "incomplete_states": incomplete_states,
                "contrib_states": contrib_states,
                "pos_weight": args.pos_weight,
                "lam_kd": args.lam_kd,
                "mean_kd": float(np.mean(hist["kd"])) if hist["kd"] else 0.0,
                "anchor_mode": args.anchor,
                "checkpoint_every": args.checkpoint_every,
                "state_balanced": bool(args.state_balanced),
                "states_per_unit": len(incomplete_states) + 1 if args.state_balanced else 2,
                "mean_margin_gain_start": float(np.mean(hist["margin_gain"][:200])),
                "mean_margin_gain_end": float(np.mean(hist["margin_gain"][-200:])),
                "m_neg_start": float(np.mean(hist["m_neg"][:200])),
                "m_neg_end": float(np.mean(hist["m_neg"][-200:])),
                "m_pos_start": float(np.mean(hist["m_pos"][:200])),
                "m_pos_end": float(np.mean(hist["m_pos"][-200:])),
                "elapsed_seconds": time.time() - started,
                "official_dev_evaluations": 0, "official_test_evaluations": 0}
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    if wandb_run is not None:
        wandb_run.summary.update({
            "status": "COMPLETE",
            "final_adapter_sha256": manifest["adapter_sha256"],
            "optimizer_steps": ctr["optimizer_steps"],
        })
        wandb_run.finish()
    print(json.dumps({k: manifest[k] for k in
                      ("status", "arm", "counters", "mean_margin_gain_start", "mean_margin_gain_end")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
