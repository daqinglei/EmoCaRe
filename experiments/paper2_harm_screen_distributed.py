#!/usr/bin/env python3
"""Multi-process data-parallel trainer for the frozen Paper2 harm screen.

Each rank sees exactly one physical GPU through CUDA_VISIBLE_DEVICES.  A global
batch keeps the preregistered 16 examples: ranks take strided slices of each
16-example block, then LoRA gradients are SUM-reduced before the optimizer
step.  Since every local loss is divided by 16, the reduced gradient is the
same global-batch scaling as the single-GPU implementation.

This is an execution amendment, not a new scientific arm.  Distributed
dropout streams are recorded because they cannot be bit-identical to the
already completed single-GPU Control/A trajectories.
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


ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + ""))
DEFAULT_SCREEN = ROOT / "work_dir/iemocap_qwen_v1/harm_screen_20260908_v1"
GLOBAL_BATCH = 16
SEED = 20260906
N2_LAMBDA_DEFAULT = 0.1


def script_sha256() -> str:
    h = hashlib.sha256()
    with open(__file__, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
    os.replace(temporary, path)


def import_frozen(screen: Path):
    sys.path.insert(0, str(screen / "source"))
    import paper2_harm_screen_v1 as frozen
    frozen.SHA = json.loads((screen / 'preregistration.json').read_text())['initial_tensor_sha256']
    return frozen


def make_training_order(count, epoch_budget=0):
    base = np.random.default_rng(SEED).permutation(count).tolist()
    if epoch_budget:
        order = base * epoch_budget
        order = order[:len(order) // GLOBAL_BATCH * GLOBAL_BATCH]
    else:
        order = (base * ((8880 + count - 1) // count))[:8880]
    if not order or len(order) % GLOBAL_BATCH:
        raise RuntimeError('Invalid training order')
    return order


def validation_key(record):
    return (record['selection_score'], record['state_mean_WAR_UAR_MF1'][2], -record['step'])


def init_distributed(args):
    import torch
    import torch.distributed as dist

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Each rank must receive exactly one physical GPU via CUDA_VISIBLE_DEVICES"
        )
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=args.rank,
        world_size=args.world_size,
    )
    return torch, dist


def reduce_gradients(model, torch, dist) -> None:
    """SUM gradients, matching per-sample losses scaled by GLOBAL_BATCH."""
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


def assert_replicas_equal(model, adapter_digest, dist) -> str:
    digest = adapter_digest(model)
    replicas = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(replicas, digest)
    if len(set(replicas)) != 1:
        raise RuntimeError(f"Distributed adapters diverged: {replicas}")
    return digest


def n2_recoverable_target(references, target, state, mode, torch):
    """Return the offline correct-subset target for one child state, or None."""
    current_set = set(state)
    proper = [name for name in references if set(name) < current_set]
    if not proper or int(references[state].argmax()) == target:
        return None
    good = [name for name in proper if int(references[name].argmax()) == target]
    if not good:
        return None
    value = torch.stack([torch.softmax(references[name], dim=-1) for name in good]).mean(0)
    if mode == "cm":
        confidence = value[target]
        value = torch.ones_like(value) * ((1.0 - confidence) / (len(value) - 1))
        value[target] = confidence
    return value.detach()


def n2_support(cached, labels, states) -> dict:
    import torch

    events = 0
    by_state = {state: 0 for state in states}
    for record in cached.values():
        target = labels.index(record["gt"])
        references = {
            state: torch.tensor(record["states"][state], dtype=torch.float32)
            for state in states
        }
        for state in states:
            if n2_recoverable_target(references, target, state, "true", torch) is not None:
                events += 1
                by_state[state] += 1
    total = len(cached) * len(states)
    return {"events": events, "total_sample_states": total,
            "rho": events / total if total else 0.0, "events_by_state": by_state}


def validation_metrics(model, processor, runtime, validation_rows, rank, world,
                       frozen, torch, dist, selection_metric):
    """Exact candidate-score validation, sharded over ranks, with seven states."""
    from run_meld_qwen_sft_selection_eval import prepare_view
    from run_planb_meld_qwen_pilot import cached_sequence_scores

    source_protocol = json.loads(
        Path(runtime["source"]["platform_protocol"]).read_text(encoding="utf-8")
    )
    video_pixels = int(runtime["training"]["video_total_pixels"])
    confusion = torch.zeros(
        (len(frozen.STATES), len(frozen.LABELS), len(frozen.LABELS)),
        dtype=torch.int64,
        device="cuda",
    )
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for row in validation_rows[rank::world]:
            target = frozen.LABELS.index(row["gt"])
            waveform, video = frozen.decoded_row(row, video_pixels)
            for state_index, state in enumerate(frozen.STATES):
                prepared, provenance = prepare_view(
                    source_protocol, processor, dict(row), state, waveform, video
                )
                values, _ = cached_sequence_scores(
                    model, prepared, provenance["candidate_token_ids"]
                )
                prediction = max(frozen.LABELS, key=values.__getitem__)
                confusion[state_index, target, frozen.LABELS.index(prediction)] += 1
    dist.all_reduce(confusion, op=dist.ReduceOp.SUM)
    if was_training:
        model.train()

    matrix = confusion.double()
    diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
    row_sum = matrix.sum(-1)
    column_sum = matrix.sum(-2)
    war = diagonal.sum(-1) / matrix.sum((-2, -1)).clamp_min(1) * 100
    recall = diagonal / row_sum.clamp_min(1)
    f1 = 2 * diagonal / (row_sum + column_sum).clamp_min(1)
    uar = recall.mean(-1) * 100
    macro_f1 = f1.mean(-1) * 100
    per_state = torch.stack([war, uar, macro_f1], -1)
    state_mean = per_state.mean(0)
    if selection_metric == "war_uar_mean":
        selection_score = float((state_mean[0] + state_mean[1]) / 2)
        selection_name = "mean of seven-state WAR and UAR"
    else:
        selection_score = float(state_mean[1])
        selection_name = "mean seven-state UAR"
    return {
        "selection_score": selection_score,
        "selection_metric": selection_name,
        "state_mean_WAR_UAR_MF1": state_mean.cpu().tolist(),
        "per_state_WAR_UAR_MF1": {
            state: per_state[index].cpu().tolist()
            for index, state in enumerate(frozen.STATES)
        },
        "samples": int(confusion.sum().item() // len(frozen.STATES)),
        "dialogues": len({row["dialogue_id"] for row in validation_rows}),
    }


def train(args) -> None:
    torch, dist = init_distributed(args)
    screen = args.directory.resolve()
    frozen = import_frozen(screen)
    from run_meld_c6_balanced_evidence import adapter_digest

    rank, world = args.rank, args.world_size
    destination = screen / args.output_arm
    started = time.time()
    try:
        frozen.verify(screen)
        cache_complete = json.loads((screen / "cache_complete.json").read_text())
        if frozen.sha(screen / "f0_sequences.jsonl") != cache_complete["sha256"]:
            raise RuntimeError("Frozen F0 cache digest mismatch")
        if args.arm not in {"control", "A", "B", "N2_control", "N2_true", "N2_cm"}:
            raise RuntimeError("Unknown training arm")

        train_manifest = (args.train_manifest.resolve() if args.train_manifest
                          else frozen.training_manifest(screen))
        validation_manifest = (args.validation_manifest.resolve()
                               if args.validation_manifest else None)
        if bool(validation_manifest) != bool(args.validate_every):
            raise RuntimeError("validation-manifest and validate-every must be set together")
        order = make_training_order(len(frozen.rows(train_manifest)), args.epoch_budget)
        budget_steps = len(order) // GLOBAL_BATCH
        if args.max_steps > budget_steps:
            raise RuntimeError('Requested steps exceed frozen epoch budget')
        if rank == 0:
            destination.mkdir()
            prereg = json.loads((screen / "preregistration.json").read_text())
            atomic_json(destination / "config.json", {
                **prereg,
                "arm": args.arm,
                "output_arm": args.output_arm,
                "execution_amendment": "manual synchronous gradient SUM data parallel",
                "world_size": world,
                "global_batch": GLOBAL_BATCH,
                "rank_batch_sizes": [len(range(r, GLOBAL_BATCH, world)) for r in range(world)],
                "rank_seeds": [SEED + r for r in range(world)],
                "distributed_dropout_not_bit_identical_to_single_gpu_controls": True,
                "trainer_sha256": script_sha256(),
                "train_manifest": str(train_manifest),
                "train_manifest_sha256": frozen.sha(train_manifest),
                "validation_manifest": str(validation_manifest) if validation_manifest else None,
                "validation_manifest_sha256": frozen.sha(validation_manifest) if validation_manifest else None,
                "validate_every": args.validate_every,
                "selection_metric": args.selection_metric if validation_manifest else None,
                "early_stop_patience": args.early_stop_patience,
                "max_steps": args.max_steps, "scheduler_budget_steps": budget_steps,
                "epoch_budget": args.epoch_budget,
                "wandb": {"project": args.wandb_project, "entity": args.wandb_entity,
                          "mode": args.wandb_mode},
            })
        dist.barrier()

        model, processor, runtime, label_ids = frozen.load_model(screen)
        initial_digest = assert_replicas_equal(model, adapter_digest, dist)
        if initial_digest != frozen.SHA:
            raise RuntimeError(f"Wrong initial adapter: {initial_digest}")

        # Run the original step-0 scorer gate on rank 0.  Other replicas were
        # independently digest-checked above and wait at the barrier.
        gate = []
        if rank == 0:
            model.save_pretrained(destination / "initial_adapter", safe_serialization=True)
            training_rows = frozen.rows(train_manifest)
            cached_gate = {
                row["sample_id"]: row
                for row in frozen.rows(screen / "f0_sequences.jsonl")
            }
            video_pixels = int(runtime["training"]["video_total_pixels"])
            model.train()
            for row in training_rows[:3]:
                decoded = frozen.decoded_row(row, video_pixels)
                for state in frozen.STATES:
                    with torch.no_grad(), frozen.decision_mode(model):
                        output, inputs = frozen.forward(
                            model, processor, row, state, video_pixels, decoded, True
                        )
                        value = frozen.scores(output, inputs, label_ids)
                    reference = torch.tensor(
                        cached_gate[row["sample_id"]]["states"][state], device=value.device
                    )
                    error = float((value - reference).abs().max())
                    gate.append(error)
                    if error > .02 or int(value.argmax()) != int(reference.argmax()):
                        raise RuntimeError((row["sample_id"], state, error))
                    del output, inputs, value
            atomic_json(destination / "step0_gate.json", {
                "status": "PASS",
                "pairs": len(gate),
                "max_abs_score_error": max(gate),
                "initial_tensor_sha256": initial_digest,
                "replicas": world,
            })
            atomic_json(destination / "manifest.pretrain.json", {
                "initial_tensor_sha256": initial_digest,
                "frozen_files_sha256": frozen.sha(screen / "frozen_files.json"),
                "cache_sha256": frozen.sha(screen / "f0_sequences.jsonl"),
                "harm_statistics_sha256": frozen.sha(screen / "harm_statistics.json"),
                "trainer_sha256": script_sha256(),
                "hostname": socket.gethostname(),
                "world_size": world,
            })
        dist.barrier()

        training_rows = frozen.rows(train_manifest)
        validation_rows = frozen.rows(validation_manifest) if validation_manifest else []
        if validation_rows:
            train_ids = {row["sample_id"] for row in training_rows}
            validation_ids = {row["sample_id"] for row in validation_rows}
            train_dialogues = {row["dialogue_id"] for row in training_rows}
            validation_dialogues = {row["dialogue_id"] for row in validation_rows}
            if train_ids & validation_ids or train_dialogues & validation_dialogues:
                raise RuntimeError("Train/validation isolation gate failed")
        cached = {
            row["sample_id"]: row
            for row in frozen.rows(screen / "f0_sequences.jsonl")
        }
        support = n2_support(cached, frozen.LABELS, frozen.STATES)
        if args.arm.startswith("N2_") and support["rho"] <= 0:
            raise RuntimeError("N2 has no recoverable training events")
        if rank == 0:
            atomic_json(destination / "n2_support.json", support)
            config = json.loads((destination / "config.json").read_text())
            config["n2"] = {
                "mode": args.arm.removeprefix("N2_"),
                "lambda": args.n2_lambda,
                "support": support,
                "target": "mean frozen-F0 posterior over all correct proper subsets",
                "confidence_matched": "preserve GT probability and uniformize non-target mass",
            }
            atomic_json(destination / "config.json", config)
        dist.barrier()
        harm = json.loads((screen / "harm_statistics.json").read_text())
        weights = np.asarray(harm["weights"])
        video_pixels = int(runtime["training"]["video_total_pixels"])
        if rank == 0:
            atomic_json(destination / "unit_order.json", order)

        # Rank-specific streams are the standard distributed-training choice;
        # record them rather than pretending bit identity with single GPU.
        torch.manual_seed(SEED + rank)
        torch.cuda.manual_seed_all(SEED + rank)
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
            betas=(.9, .999),
            eps=1e-8,
            weight_decay=0,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0.0, 1 - step / budget_steps)
        )
        optimizer.zero_grad(set_to_none=True)
        wandb_run = None
        if rank == 0 and args.wandb_project:
            import wandb
            wandb_directory = destination / "wandb"
            wandb_directory.mkdir()
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity or None,
                name=args.wandb_name or f"{args.output_arm}-{screen.name}",
                dir=str(wandb_directory),
                mode=args.wandb_mode,
                config=json.loads((destination / "config.json").read_text()),
                tags=["Paper2", "IEMOCAP", screen.name, args.arm, "missing-modality"],
            )
            atomic_json(destination / "wandb_run.json", {
                "id": wandb_run.id,
                "name": wandb_run.name,
                "project": wandb_run.project,
                "entity": wandb_run.entity,
                "url": wandb_run.url,
                "mode": args.wandb_mode,
            })
        history_context = (
            (destination / "history.jsonl").open("x", buffering=1)
            if rank == 0 else contextlib.nullcontext(None)
        )
        best_validation = None
        bad_validation_count = 0
        stopped_early = False
        completed_steps = 0
        validation_stream = ((destination / "validation_history.jsonl").open("x", buffering=1)
                             if rank == 0 and validation_rows else contextlib.nullcontext(None))
        with history_context as history, validation_stream as validation_history:
            for step_zero, block_start in enumerate(range(0, len(order), GLOBAL_BATCH)):
                if step_zero >= args.max_steps:
                    break
                global_indices = order[block_start:block_start + GLOBAL_BATCH]
                local_indices = global_indices[rank::world]
                local_task = 0.0
                local_harm = 0.0
                local_n2 = 0.0
                local_n2_events = 0
                for index in local_indices:
                    row = training_rows[index]
                    target = frozen.LABELS.index(row["gt"])
                    decoded = frozen.decoded_row(row, video_pixels)
                    references = {
                        state: torch.tensor(
                            cached[row["sample_id"]]["states"][state], device="cuda"
                        )
                        for state in frozen.STATES
                    }
                    references = {
                        state: value - value.mean() for state, value in references.items()
                    }
                    current = {}
                    active = ([
                        state for state in frozen.EDGES
                        if int(references[state].argmax()) == target
                        and int(references["AVT"].argmax()) != target
                    ] if args.arm == "B" else [])
                    for state_index, state in enumerate(frozen.STATES):
                        n2_mode = ("true" if args.arm == "N2_true" else
                                   ("cm" if args.arm == "N2_cm" else None))
                        n2_target = (n2_recoverable_target(
                            references, target, state, n2_mode, torch
                        ) if n2_mode else None)
                        requires_harm_grad = state == "AVT" and bool(active)
                        requires_decision_grad = requires_harm_grad or n2_target is not None
                        with torch.set_grad_enabled(requires_decision_grad), frozen.decision_mode(model):
                            output, inputs = frozen.forward(
                                model, processor, row, state, video_pixels, decoded, True
                            )
                            score = frozen.scores(output, inputs, label_ids)
                            score = score - score.mean()
                            current[state] = score.detach()
                            if requires_harm_grad:
                                competitor = int(references["AVT"].argmax())
                                direction = torch.zeros_like(score)
                                direction[target] = 1
                                direction[competitor] = -1
                                losses = []
                                for negative in active:
                                    beta = torch.relu(
                                        -((references[state] - references[negative]) * direction).sum()
                                    ) / 2
                                    residual = (
                                        (score - references[state])
                                        - (current[negative] - references[negative]).detach()
                                        - beta * direction
                                    )
                                    losses.append(torch.nn.functional.huber_loss(
                                        residual,
                                        torch.zeros_like(residual),
                                        reduction="sum",
                                        delta=1.0,
                                    ))
                                harm_loss = torch.stack(losses).mean()
                                if not torch.isfinite(harm_loss):
                                    raise RuntimeError("Non-finite B harm loss")
                                (.1 * harm_loss / GLOBAL_BATCH).backward()
                                local_harm += float(harm_loss.detach())
                                del harm_loss, losses, residual
                            if n2_target is not None:
                                n2_loss = torch.nn.functional.kl_div(
                                    torch.log_softmax(score, dim=-1),
                                    n2_target,
                                    reduction="sum",
                                )
                                scaled_n2 = (
                                    args.n2_lambda
                                    / (len(frozen.STATES) * support["rho"])
                                    * n2_loss
                                )
                                if not torch.isfinite(scaled_n2):
                                    raise RuntimeError("Non-finite N2 loss")
                                (scaled_n2 / GLOBAL_BATCH).backward()
                                local_n2 += float(n2_loss.detach())
                                local_n2_events += 1
                                del n2_loss, scaled_n2
                        del output, inputs, score

                        output, inputs = frozen.forward(
                            model, processor, row, state, video_pixels, decoded, False
                        )
                        task_loss = output.loss / 7
                        if args.arm == "A":
                            task_loss = task_loss * float(weights[state_index, target])
                        if not torch.isfinite(task_loss):
                            raise RuntimeError("Non-finite task loss")
                        (task_loss / GLOBAL_BATCH).backward()
                        local_task += float(task_loss.detach())
                        del output, inputs, task_loss

                reduce_gradients(model, torch, dist)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                totals = torch.tensor(
                    [local_task, local_harm, local_n2, local_n2_events],
                    device="cuda", dtype=torch.float64,
                )
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                step = step_zero + 1
                completed_steps = step
                if rank == 0:
                    entry = {
                        "arm": args.arm,
                        "output_arm": args.output_arm,
                        "step": step,
                        "steps": args.max_steps,
                        "task": float(totals[0]) / GLOBAL_BATCH,
                        "harm": float(totals[1]) / GLOBAL_BATCH,
                        "n2": float(totals[2]) / max(1.0, float(totals[3])),
                        "n2_events": int(totals[3]),
                        "grad_norm": float(grad_norm),
                        "elapsed": time.time() - started,
                        "world_size": world,
                        "rank_batch_sizes": [len(range(r, GLOBAL_BATCH, world)) for r in range(world)],
                        "peak_alloc_MiB": torch.cuda.max_memory_allocated() / 2**20,
                    }
                    history.write(json.dumps(entry) + "\n")
                    print(json.dumps(entry), flush=True)
                    if wandb_run is not None:
                        wandb_run.log({
                            "train/task_loss": entry["task"],
                            "train/harm_loss": entry["harm"],
                            "train/n2_loss_selected_mean": entry["n2"],
                            "train/n2_events": entry["n2_events"],
                            "train/grad_norm": entry["grad_norm"],
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "train/elapsed_seconds": entry["elapsed"],
                        }, step=step)

                checkpoint_due = (step % args.checkpoint_every == 0 or step == args.max_steps)
                if checkpoint_due:
                    digest = assert_replicas_equal(model, adapter_digest, dist)
                    if rank == 0:
                        frozen.verify(screen)
                        model.save_pretrained(
                            destination / f"checkpoint-{step}", safe_serialization=True
                        )
                        atomic_json(destination / f"checkpoint-{step}.json", {
                            "step": step,
                            "adapter_sha256": digest,
                            "world_size": world,
                        })
                    dist.barrier()

                validation_due = bool(validation_rows) and (
                    step % args.validate_every == 0 or step == args.max_steps
                )
                if validation_due:
                    measured = validation_metrics(
                        model, processor, runtime, validation_rows, rank, world,
                        frozen, torch, dist, args.selection_metric,
                    )
                    measured.update({"step": step, "adapter_sha256": digest})
                    if rank == 0:
                        validation_history.write(json.dumps(measured) + "\n")
                        print(json.dumps({"arm": args.arm, "validation": measured}), flush=True)
                        if wandb_run is not None:
                            values = measured["state_mean_WAR_UAR_MF1"]
                            wandb_run.log({
                                "validation/state_mean_WAR": values[0],
                                "validation/state_mean_UAR": values[1],
                                "validation/state_mean_MF1": values[2],
                                "validation/selection_score": measured["selection_score"],
                            }, step=step)
                        if (best_validation is None or
                                validation_key(measured) > validation_key(best_validation)):
                            best_validation = measured
                            bad_validation_count = 0
                            atomic_json(destination / "best_checkpoint.json", best_validation)
                        else:
                            bad_validation_count += 1
                        if wandb_run is not None:
                            wandb_run.log({
                                "validation/no_improvement_count": bad_validation_count,
                            }, step=step)
                    stop_flag = torch.zeros(1, dtype=torch.int32, device="cuda")
                    if (rank == 0 and args.early_stop_patience > 0 and
                            bad_validation_count >= args.early_stop_patience and
                            step < args.max_steps):
                        stop_flag.fill_(1)
                        atomic_json(destination / "early_stop.json", {
                            "status": "EARLY_STOPPED",
                            "step": step,
                            "patience": args.early_stop_patience,
                            "selection_metric": measured["selection_metric"],
                            "last_selection_score": measured["selection_score"],
                            "best_checkpoint": best_validation,
                        })
                    dist.broadcast(stop_flag, src=0)
                    dist.barrier()
                    if int(stop_flag.item()):
                        stopped_early = True
                        break

        final_digest = assert_replicas_equal(model, adapter_digest, dist)
        if rank == 0:
            model.save_pretrained(destination / "adapter", safe_serialization=True)
            atomic_json(destination / "manifest.json", {
                "status": ("EARLY_STOPPED" if stopped_early else
                           ("COMPLETE" if args.max_steps == budget_steps else "SMOKE_COMPLETE")),
                "arm": args.arm,
                "output_arm": args.output_arm,
                "initial_adapter_sha256": frozen.SHA,
                "final_adapter_sha256": final_digest,
                "steps": completed_steps,
                "max_steps_requested": args.max_steps,
                "units": min(len(order), completed_steps * GLOBAL_BATCH),
                "train_samples": len(training_rows),
                "train_dialogues": len({row["dialogue_id"] for row in training_rows}),
                "elapsed_seconds": time.time() - started,
                "world_size": world,
                "global_batch": GLOBAL_BATCH,
                "rank_batch_sizes": [len(range(r, GLOBAL_BATCH, world)) for r in range(world)],
                "trainer_sha256": script_sha256(),
                "train_manifest": str(train_manifest),
                "train_samples_effective": len(training_rows),
                "validation_manifest": str(validation_manifest) if validation_manifest else None,
                "validation_samples": len(validation_rows),
                "best_checkpoint": best_validation,
                "selection_metric": args.selection_metric if validation_rows else None,
                "early_stop_patience": args.early_stop_patience,
                "n2_lambda": args.n2_lambda,
                "n2_support": support,
            })
            if wandb_run is not None:
                wandb_run.summary["best_step"] = (best_validation or {}).get("step")
                wandb_run.summary["best_validation_score"] = (best_validation or {}).get("selection_score")
                wandb_run.summary["selection_metric"] = args.selection_metric
                wandb_run.summary["stopped_early"] = stopped_early
                wandb_run.finish()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=DEFAULT_SCREEN)
    parser.add_argument("--arm", default="B")
    parser.add_argument("--output-arm", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=555)
    parser.add_argument("--epoch-budget", type=int, default=0,
                        help="0 keeps historical 8880 exposure budget; crossfold uses 2 epochs")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--validate-every", type=int, default=0)
    parser.add_argument("--selection-metric", choices=["uar", "war_uar_mean"],
                        default="war_uar_mean")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="number of consecutive non-improving validations; 0 disables")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--n2-lambda", type=float, default=N2_LAMBDA_DEFAULT)
    arguments = parser.parse_args()
    if arguments.world_size < 2:
        raise SystemExit("Distributed trainer requires at least two ranks")
    if not 0 <= arguments.rank < arguments.world_size:
        raise SystemExit("rank must be in [0, world_size)")
    if arguments.max_steps < 1 or arguments.epoch_budget < 0:
        raise SystemExit("max-steps must be positive, epoch-budget nonnegative")
    if arguments.checkpoint_every < 1 or arguments.validate_every < 0:
        raise SystemExit('Invalid checkpoint or validation interval')
    if arguments.early_stop_patience < 0:
        raise SystemExit("early-stop-patience must be >= 0")
    if arguments.n2_lambda < 0:
        raise SystemExit("n2-lambda must be nonnegative")
    if arguments.early_stop_patience and not arguments.validation_manifest:
        raise SystemExit("early stopping requires validation-manifest")
    if (arguments.validation_manifest and
            arguments.validate_every % arguments.checkpoint_every != 0):
        raise SystemExit("validate-every must be a multiple of checkpoint-every")
    train(arguments)


if __name__ == "__main__":
    main()
