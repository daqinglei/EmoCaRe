#!/usr/bin/env python3
"""Matched multi-GPU F0 training for the Paper2 Qwen scale gate.

This is a data-parallel execution of the existing IEMOCAP F0 objective:

    .50 CE(AVT) + .25 class_weight[y] CE(AV)
                  + .25 class_weight[y] CE(A)

The global batch remains 16.  Each process sees one physical GPU, processes a
strided part of every global batch, and SUM-reduces LoRA gradients.  Thus the
loss scaling matches the historical single-GPU grad-accumulation-16 runner.
Only LoRA parameters are trainable; all backbone/tower parameters stay frozen.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np


GLOBAL_BATCH = 16
DEFAULT_SEED = 20260906
STATES = ("AV", "A", "AVT")


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_digest(model) -> str:
    digest = hashlib.sha256()
    tensors = sorted(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    for name, parameter in tensors:
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def init_distributed(args):
    import torch
    import torch.distributed as dist

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Each rank must see exactly one GPU through CUDA_VISIBLE_DEVICES"
        )
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=args.rank,
        world_size=args.world_size,
    )
    return torch, dist


def assert_replicas_equal(model, dist) -> str:
    digest = adapter_digest(model)
    values = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(values, digest)
    if len(set(values)) != 1:
        raise RuntimeError(f"LoRA replicas diverged: {values}")
    return digest


def reduce_gradients(model, torch, dist) -> None:
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", default="angry,happy,neutral,sad")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-project", default="EmoZoom-Paper2")
    parser.add_argument("--wandb-entity", default="2265836229-tzu")
    parser.add_argument("--wandb-name", default="")
    args = parser.parse_args()

    if args.world_size < 1 or GLOBAL_BATCH % args.world_size:
        raise SystemExit("world size must divide the frozen global batch 16")

    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    from label_set import apply_label_set
    from run_meld_qwen_lora_sft import (
        build_model,
        move_inputs,
        patch_torch_compat,
        prepare_training_input,
    )

    labels = [value for value in args.labels.split(",") if value]
    apply_label_set(labels)
    patch_torch_compat()
    torch, dist = init_distributed(args)
    rank = args.rank
    started = time.time()
    output = args.output_dir.resolve()
    protocol_path = args.protocol.resolve()
    manifest_path = args.manifest.resolve()

    try:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=False)
        dist.barrier()

        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        runtime = dict(protocol)
        if "training" not in runtime:
            runtime["training"] = protocol["f0"]["training"]

        # All ranks must construct the identical fresh LoRA before dropout
        # streams are separated for distributed training.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model, processor, trainable_names, trainable, total = build_model(
            runtime,
            torch.device("cuda", 0),
            "flash_attention_2",
            gradient_checkpointing=True,
        )
        initial_digest = assert_replicas_equal(model, dist)
        if rank == 0:
            model.save_pretrained(output / "initial_adapter", safe_serialization=True)
            atomic_json(output / "step0_gate.json", {
                "status": "PASS",
                "initial_adapter_sha256": initial_digest,
                "replicas": args.world_size,
                "trainable_parameter_count": trainable,
                "trainable_tensor_count": len(trainable_names),
                "base_function_expected": True,
                "optimizer_updates": 0,
            })
        dist.barrier()

        rows = read_rows(manifest_path)
        if len(rows) != len({row["sample_id"] for row in rows}):
            raise RuntimeError("duplicate samples in F0 manifest")
        order = np.random.default_rng(args.seed).permutation(len(rows)).tolist()
        order = (order * args.epochs)
        order = order[: len(order) // GLOBAL_BATCH * GLOBAL_BATCH]
        budget_steps = len(order) // GLOBAL_BATCH
        steps = args.max_steps or budget_steps
        if steps > budget_steps:
            raise RuntimeError("requested max steps exceeds frozen epoch budget")
        if rank == 0:
            atomic_json(output / "unit_order.json", order)

        counts = np.zeros(len(labels), dtype=np.float64)
        for row in rows:
            counts[labels.index(row["gt"])] += 1
        class_weights = torch.tensor(
            counts.sum() / (len(labels) * np.maximum(counts, 1)),
            dtype=torch.float32,
            device="cuda",
        )

        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
        model.train()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0.0, 1.0 - step / max(1, budget_steps))
        )
        optimizer.zero_grad(set_to_none=True)

        wandb_run = None
        if rank == 0 and args.wandb_project:
            import wandb

            wandb_dir = output / "wandb"
            wandb_dir.mkdir()
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=args.wandb_name or f"F0-{output.parent.name}",
                dir=str(wandb_dir),
                mode=args.wandb_mode,
                config={
                    "stage": "7B Fold2 F0",
                    "objective": ".5*CE(AVT)+.25*cw*CE(AV)+.25*cw*CE(A)",
                    "global_batch": GLOBAL_BATCH,
                    "world_size": args.world_size,
                    "epochs": args.epochs,
                    "steps": steps,
                    "seed": args.seed,
                    "lr": args.lr,
                    "model": runtime["training"]["model"],
                    "protocol_sha256": sha256(protocol_path),
                    "manifest_sha256": sha256(manifest_path),
                },
                tags=["Paper2", "IEMOCAP", "Fold2", "Qwen7B", "F0"],
            )
            atomic_json(output / "wandb_run.json", {
                "id": wandb_run.id,
                "name": wandb_run.name,
                "url": wandb_run.url,
                "mode": args.wandb_mode,
            })

        history_context = (
            (output / "history.jsonl").open("x", encoding="utf-8", buffering=1)
            if rank == 0 else contextlib.nullcontext(None)
        )
        completed = 0
        global_forward_count = 0
        global_supervised_tokens = 0
        with history_context as history:
            for step_zero in range(steps):
                block = order[
                    step_zero * GLOBAL_BATCH:(step_zero + 1) * GLOBAL_BATCH
                ]
                local_indices = block[rank::args.world_size]
                local_loss_sum = 0.0
                local_forward_count = 0
                local_supervised_tokens = 0
                for index in local_indices:
                    row = rows[index]
                    target = labels.index(row["gt"])
                    for state in STATES:
                        inputs, statistics = prepare_training_input(
                            processor,
                            dict(row),
                            state,
                            int(runtime["training"]["video_total_pixels"]),
                        )
                        inputs = move_inputs(inputs, torch.device("cuda", 0))
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            result = model(
                                **inputs,
                                use_cache=False,
                                use_audio_in_video=False,
                            )
                        if result.loss is None or not torch.isfinite(result.loss):
                            raise RuntimeError(
                                f"non-finite F0 loss at {row['sample_id']} {state}"
                            )
                        if state == "AVT":
                            weighted = 0.5 * result.loss
                        else:
                            weighted = 0.25 * class_weights[target] * result.loss
                        (weighted / GLOBAL_BATCH).backward()
                        local_loss_sum += float(weighted.detach())
                        local_forward_count += 1
                        local_supervised_tokens += int(statistics["supervised_tokens"])
                        del result, inputs, weighted

                reduce_gradients(model, torch, dist)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                totals = torch.tensor(
                    [local_loss_sum, local_forward_count, local_supervised_tokens],
                    dtype=torch.float64,
                    device="cuda",
                )
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                completed = step_zero + 1
                global_forward_count += int(totals[1].item())
                global_supervised_tokens += int(totals[2].item())
                if rank == 0:
                    entry = {
                        "step": completed,
                        "steps": steps,
                        "task_loss": float(totals[0].item()) / GLOBAL_BATCH,
                        "grad_norm": float(gradient_norm),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "elapsed": time.time() - started,
                        "peak_alloc_MiB": torch.cuda.max_memory_allocated() / 2**20,
                        "world_size": args.world_size,
                        "rank_batch_size": len(local_indices),
                    }
                    history.write(json.dumps(entry) + "\n")
                    print(json.dumps(entry), flush=True)
                    if wandb_run is not None:
                        wandb_run.log({
                            "train/task_loss": entry["task_loss"],
                            "train/grad_norm": entry["grad_norm"],
                            "train/learning_rate": entry["learning_rate"],
                            "train/elapsed_seconds": entry["elapsed"],
                            "system/peak_alloc_MiB": entry["peak_alloc_MiB"],
                        }, step=completed)

                checkpoint_due = (
                    completed % args.checkpoint_every == 0 or completed == steps
                )
                if checkpoint_due:
                    digest = assert_replicas_equal(model, dist)
                    if rank == 0:
                        model.save_pretrained(
                            output / f"checkpoint-{completed}",
                            safe_serialization=True,
                        )
                        atomic_json(output / f"checkpoint-{completed}.json", {
                            "step": completed,
                            "adapter_sha256": digest,
                            "world_size": args.world_size,
                        })
                    dist.barrier()

        final_digest = assert_replicas_equal(model, dist)
        if rank == 0:
            model.save_pretrained(output / "adapter", safe_serialization=True)
            manifest = {
                "status": "COMPLETE" if completed == budget_steps else "SMOKE_COMPLETE",
                "stage": "IEMOCAP Fold2 F0",
                "arm": "task_only",
                "architecture": protocol["f0"]["architecture"],
                "seed": args.seed,
                "epochs": args.epochs,
                "world_size": args.world_size,
                "global_batch": GLOBAL_BATCH,
                "rank_batch_size": GLOBAL_BATCH // args.world_size,
                "optimizer_steps": completed,
                "budget_steps": budget_steps,
                "train_samples": len(rows),
                "train_dialogues": len({row["dialogue_id"] for row in rows}),
                "forward_count": global_forward_count,
                "supervised_tokens": global_supervised_tokens,
                "objective": ".5*CE(AVT)+.25*cw[y]*CE(AV)+.25*cw[y]*CE(A)",
                "initial_adapter_sha256": initial_digest,
                "adapter_sha256": final_digest,
                "trainable_parameter_count": trainable,
                "total_parameter_count_with_adapter": total,
                "protocol_sha256": sha256(protocol_path),
                "manifest_sha256": sha256(manifest_path),
                "trainer_sha256": sha256(Path(__file__).resolve()),
                "hostname": socket.gethostname(),
                "gradient_checkpointing": True,
                "distributed_dropout_streams": [args.seed + value for value in range(args.world_size)],
                "official_dev_evaluations": 0,
                "official_test_evaluations": 0,
                "elapsed_seconds": time.time() - started,
            }
            atomic_json(output / "manifest.json", manifest)
            if wandb_run is not None:
                wandb_run.summary.update({
                    "status": manifest["status"],
                    "final_adapter_sha256": final_digest,
                    "optimizer_steps": completed,
                })
                wandb_run.finish()
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
