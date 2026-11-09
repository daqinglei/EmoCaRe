#!/usr/bin/env python3
"""Matched answer-only LoRA-SFT for the frozen MELD × Qwen platform pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260830
STATES = ("AVT", "AV", "AT", "VT", "A", "V", "T")
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
# The label set is a module-level global so a second corpus (IEMOCAP-4) can
# narrow it without forking the prompt or the training path.  set_label_set()
# rewrites both, and MELD's seven-way default is what every existing run used.
LABEL_SET = ["happy", "sad", "neutral", "angry", "surprise", "disgust", "fear"]


def _task_prompt(labels):
    return ("Classify the emotion conveyed by the supplied evidence. "
            "Answer with exactly one lowercase label: " + ", ".join(labels) + ".")


TASK_PROMPT = _task_prompt(LABEL_SET)


def set_label_set(labels):
    """Narrow the classifier to another corpus's label set."""
    global LABEL_SET, TASK_PROMPT
    LABEL_SET = list(labels)
    TASK_PROMPT = _task_prompt(LABEL_SET)
    return LABEL_SET
CANDIDATE_GROUP = 1   # candidates per cached-prefix step group; 1 keeps the
                      # prefix cache un-replicated, which matters because the
                      # model has no num_logits_to_keep and already holds a full
                      # [1, prompt, vocab] logits tensor in the graph
STATE_SPEC = {
    "AVT": {"audio": True, "video": True, "text": True},
    "AV": {"audio": True, "video": True, "text": False},
    "AT": {"audio": True, "video": False, "text": True},
    "VT": {"audio": False, "video": True, "text": True},
    "A": {"audio": True, "video": False, "text": False},
    "V": {"audio": False, "video": True, "text": False},
    "T": {"audio": False, "video": False, "text": True},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_files(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def save_adapter_checkpoint(train_model, processor, output_dir: Path, step: int) -> None:
    """Save a frozen LoRA snapshot for validation-based checkpoint selection."""
    unwrapped = train_model.module if isinstance(train_model, DistributedDataParallel) else train_model
    checkpoint_dir = output_dir / f"checkpoint-{int(step)}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped.save_pretrained(checkpoint_dir / "adapter", safe_serialization=True)
    processor.save_pretrained(checkpoint_dir / "processor")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def patch_torch_compat() -> None:
    if not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda *args, **kwargs: False


def build_model(protocol: dict[str, Any], device: torch.device, attn_impl: str,
                gradient_checkpointing: bool = True):
    from peft import LoraConfig, get_peft_model
    from peft.utils.save_and_load import set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    model_path = protocol["training"]["model"]
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": device.index},
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_impl,
    }
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    model.config.use_cache = False
    # checkpointing trades ~30-40% of the step time for memory this project has
    # spare: the seven-state arms sit at 15-22 GB of 46 GB per card
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    spec = protocol["training"]["lora"]
    lora = LoraConfig(
        r=int(spec["rank"]),
        lora_alpha=int(spec["alpha"]),
        lora_dropout=float(spec["dropout"]),
        bias=str(spec["bias"]),
        target_modules=str(spec["target_regex"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    initial_adapter = protocol["training"].get("initial_adapter", "")
    if initial_adapter:
        adapter_dir = Path(initial_adapter).expanduser().resolve()
        weights_path = adapter_dir / "adapter_model.safetensors"
        if not weights_path.is_file():
            raise FileNotFoundError(f"initial adapter weights missing: {weights_path}")
        state = load_file(str(weights_path), device="cpu")
        set_peft_model_state_dict(model, state, adapter_name="default")
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_path, local_files_only=True, use_fast=False
    )

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    invalid = [
        name for name in trainable_names
        if "lora_" not in name
        or ".model.layers." not in name
        or ".self_attn." not in name
        or any(token in name for token in ("audio_tower", "visual", "talker"))
    ]
    if invalid:
        raise RuntimeError(f"LoRA escaped the frozen Thinker LM scope: {invalid[:20]}")
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    if trainable <= 0:
        raise RuntimeError("no trainable LoRA parameters")
    return model, processor, trainable_names, trainable, total


def decode_audio(video_path: Path) -> np.ndarray:
    from run_planb_meld_qwen_pilot import decode_audio_mono_16k
    # A few released M3ED WAVs are empty or shorter than the processor's
    # minimum usable window. Keep those rows in the frozen schedule and provide
    # a deterministic silent waveform so Qwen still receives an audio slot.
    minimum_samples = 1600  # 100 ms at 16 kHz
    try:
        waveform = decode_audio_mono_16k(video_path)
    except RuntimeError as exc:
        if "decoded audio is empty" not in str(exc):
            raise
        waveform = np.zeros(minimum_samples, dtype=np.float32)
    if waveform.size < minimum_samples:
        waveform = np.pad(waveform, (0, minimum_samples - waveform.size))
    return np.asarray(waveform, dtype=np.float32)


def decode_video(video_path: Path, total_pixels: int):
    from qwen_omni_utils import process_mm_info

    conversation = [{
        "role": "user",
        "content": [{
            "type": "video",
            "video": str(video_path),
            "fps": 2.0,
            "total_pixels": int(total_pixels),
        }],
    }]
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    if audios is not None or images is not None or videos is None or len(videos) != 1:
        raise RuntimeError("capped video decode did not isolate exactly one video")
    return videos


def conversations(sample: dict[str, Any], state: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec = STATE_SPEC[state]
    content: list[dict[str, Any]] = []
    # History-in-prompt baseline (SSOT 10.18.170): only when the manifest row carries it; absent -> unchanged prompt.
    if sample.get("history_text"):
        # MatchHIST (SSOT 10.18.238) carries its own header; every other manifest keeps the original one.
        header = sample.get("history_header", "Previous turns in this conversation:")
        content.append({"type": "text", "text": header + "\n" + sample["history_text"]})
    if spec["video"]:
        content.append({"type": "video", "video": sample["video_path"], "fps": 2.0})
    if spec["audio"]:
        content.append({"type": "audio", "audio": "in_memory_waveform_16khz"})
    if spec["text"]:
        content.append({"type": "text", "text": f"Transcript: {sample['transcript']}"})
    content.append({"type": "text", "text": TASK_PROMPT})  # reflects the active LABEL_SET
    prompt = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    full = prompt + [
        {"role": "assistant", "content": [{"type": "text", "text": sample["gt"]}]}
    ]
    return prompt, full


def candidate_scores(model, prompt_inputs, sequences):
    """Exact, differentiable candidate scores: one cached-prefix pass, own prefix per candidate.

    This replaces an earlier teacher-forced version that read every candidate's token j at the
    position the ANCHOR label occupies, so a candidate's second token was scored under the
    anchor's first token rather than its own.  On MELD that mis-scored three of seven labels
    (angry, surprise, fear under disgust) and on IEMOCAP-6 two of six, which made the margin the
    objective optimised different from the margin the official decision rule ranks.  IEMOCAP-4
    was unaffected: every label but the anchor is a single token there.

    The structure is the evaluator's own: the prompt runs once with use_cache, the first token of
    each candidate is read from that last position, and each further token is stepped with the
    cache batch-repeated so that candidate i is always conditioned on candidate i's own prefix.
    Gradients are kept throughout, and the number of module calls is a constant of the label set,
    so every rank issues the same collectives.
    """
    import torch

    prefix_length = int(prompt_inputs["input_ids"].shape[1])
    # Gradient checkpointing forces use_cache off inside the model, which would leave
    # past_key_values empty and make the cached-prefix scoring impossible.  It is turned off for
    # this one forward and restored immediately; the base training pass keeps it.
    inner = model.module if isinstance(model, DistributedDataParallel) else model
    had_checkpointing = bool(getattr(inner, "is_gradient_checkpointing", False))
    if had_checkpointing:
        inner.gradient_checkpointing_disable()
    try:
        out = model(**prompt_inputs, use_cache=True, return_dict=True, use_audio_in_video=False)
        if out.past_key_values is None:
            raise RuntimeError("the prompt forward returned no cache; candidate scoring needs it")
        return _score_from_cache(model, out, prompt_inputs, sequences, prefix_length)
    finally:
        if had_checkpointing:
            # Restore with the SAME kwargs it was enabled with.  The bare call defaults to
            # use_reentrant=True, which re-runs the forward inside backward and makes DDP mark a
            # parameter ready twice; that is what broke every microstep after the first.
            inner.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})


def _score_from_cache(model, out, prompt_inputs, sequences, prefix_length):
    import torch
    from transformers.cache_utils import DynamicCache
    # Only the last row of the prompt logits is ever read, but the model has no option to
    # return fewer, so the full [1, prompt, vocab] tensor is produced.  Everything needed is
    # taken out of `out` here and `out` is dropped, which lets that tensor be freed instead of
    # sitting in the graph for the whole microstep.
    first = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)[0].clone()
    scores = [first[seq[0]] for seq in sequences]
    longest = max(len(seq) for seq in sequences)
    legacy = out.past_key_values.to_legacy_cache() if longest > 1 else None
    rope_source = getattr(out, "rope_deltas", None)
    del out
    if longest > 1:
        inner = model.module if isinstance(model, DistributedDataParallel) else model
        pad = int(getattr(inner.config, "pad_token_id", 0) or 0)
        device = prompt_inputs["input_ids"].device
        # The prefix cache has to be repeated once per candidate carried through the step, and
        # repeating all seven at once exhausted the card at optimizer step 200 on 2026-09-17.
        # Candidates are stepped in small groups instead: identical arithmetic, lower peak.
        group = CANDIDATE_GROUP
        multi = [i for i, seq in enumerate(sequences) if len(seq) > 1]
        for start in range(0, len(multi), group):
            block = multi[start:start + group]
            cache = DynamicCache.from_legacy_cache(legacy)
            cache.batch_repeat_interleave(len(block))
            rope = rope_source
            if rope is not None and rope.shape[0] == 1:
                rope = rope.repeat(len(block), *([1] * (rope.ndim - 1)))
            for position in range(1, max(len(sequences[i]) for i in block)):
                previous = [sequences[i][position - 1] if len(sequences[i]) >= position else pad
                            for i in block]
                step = model(
                    input_ids=torch.tensor(previous, dtype=torch.long, device=device).unsqueeze(1),
                    attention_mask=torch.ones((len(block), prefix_length + position),
                                              dtype=torch.long, device=device),
                    past_key_values=cache,
                    cache_position=torch.tensor([prefix_length + position - 1],
                                                dtype=torch.long, device=device),
                    rope_deltas=rope, use_cache=True, return_dict=True, use_audio_in_video=False)
                cache = step.past_key_values
                log_prob = torch.log_softmax(step.logits[:, -1, :].float(), dim=-1)
                for slot, index in enumerate(block):
                    if len(sequences[index]) > position:
                        scores[index] = scores[index] + log_prob[slot, sequences[index][position]]
                del step, log_prob
            del cache
    del legacy
    return torch.stack(scores)


def margin_of(scores_vector, target):
    """p(y) - max_{c != y} p(c) over the label-set softmax."""
    import torch
    p = torch.softmax(scores_vector, dim=-1)
    other = p.clone()
    other = torch.cat([other[:target], other[target + 1:]])
    return p[target] - other.max()


def prepare_training_input(
    processor,
    sample: dict[str, Any],
    state: str,
    video_total_pixels: int,
    decoded: tuple[Any, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """`decoded` is an optional (waveform, videos) pair already read from disk.

    A seven-state unit asks for this sample four times with audio and four times
    with video; decoding inside the function repeats that work every call. The
    evaluator has always decoded once per sample and reused it, and passing the
    same pair in here makes training do the same. Omitting it keeps the original
    behaviour exactly.
    """
    video_path = Path(sample["video_path"])
    spec = STATE_SPEC[state]
    if decoded is not None:
        cached_audio, cached_video = decoded
        waveform = cached_audio if spec["audio"] else None
        videos = cached_video if spec["video"] else None
    else:
        waveform = decode_audio(video_path) if spec["audio"] else None
        videos = decode_video(video_path, video_total_pixels) if spec["video"] else None
    prompt_conv, full_conv = conversations(sample, state)
    rendered_prompt = processor.apply_chat_template(
        prompt_conv, add_generation_prompt=True, tokenize=False
    )
    rendered_full = processor.apply_chat_template(
        full_conv, add_generation_prompt=False, tokenize=False
    )
    prompt_ids = processor.tokenizer(rendered_prompt, add_special_tokens=False).input_ids
    full_ids = processor.tokenizer(rendered_full, add_special_tokens=False).input_ids
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(f"assistant boundary retokenized prompt for {sample['sample_id']}")
    if sample["gt"] not in rendered_full[len(rendered_prompt):]:
        raise RuntimeError("ground-truth answer missing from supervised suffix")

    inputs = processor(
        text=rendered_full,
        audio=[waveform] if waveform is not None else None,
        images=None,
        videos=videos,
        return_tensors="pt",
        padding=False,
        use_audio_in_video=False,
    )
    input_ids = inputs["input_ids"]
    suffix_ids = full_ids[len(prompt_ids) :]
    if input_ids.shape[0] != 1 or len(suffix_ids) <= 0:
        raise RuntimeError("invalid processor batch or empty supervised suffix")
    if input_ids.shape[1] < len(suffix_ids):
        raise RuntimeError("processor sequence is shorter than supervised suffix")
    # Qwen expands audio/video placeholders inside the prompt.  The assistant
    # suffix must nevertheless remain an exact token-for-token sequence tail.
    if input_ids[0, -len(suffix_ids) :].tolist() != suffix_ids:
        raise RuntimeError("processor changed the supervised assistant suffix")
    labels = input_ids.clone()
    labels[:, : input_ids.shape[1] - len(suffix_ids)] = -100
    supervised = int(labels.ne(-100).sum())
    if supervised <= 0:
        raise RuntimeError("no answer tokens are supervised")
    inputs["labels"] = labels

    audio_token_id = 151646
    video_token_id = 151656
    audio_tokens = int((input_ids == audio_token_id).sum())
    video_tokens = int((input_ids == video_token_id).sum())
    if (audio_tokens > 0) != spec["audio"]:
        raise RuntimeError(f"{state}: audio token isolation failed")
    if (video_tokens > 0) != spec["video"]:
        raise RuntimeError(f"{state}: video token isolation failed")
    transcript_present = f"Transcript: {sample['transcript']}" in rendered_prompt
    if transcript_present != spec["text"]:
        raise RuntimeError(f"{state}: transcript isolation failed")

    stats = {
        "input_tokens": int(input_ids.numel()),
        "supervised_tokens": supervised,
        "audio_tokens": audio_tokens,
        "video_tokens": video_tokens,
        "audio_samples": int(len(waveform)) if waveform is not None else 0,
    }
    return dict(inputs), stats


def move_inputs(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(torch.bfloat16)
        output[key] = value
    return output


def all_reduce_sum(value: int, device: torch.device) -> int:
    tensor = torch.tensor([value], dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def run(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    # Protocols for the IEMOCAP-4/6 MatchHIST runs carry their label set.  Keep
    # the historical MELD default when the field is absent.
    if protocol.get("labels"):
        set_label_set(protocol["labels"])
    if protocol["status"] != "FROZEN_BEFORE_SMOKE":
        raise RuntimeError("unexpected protocol status")
    if args.arm not in {"avt_only_sft", "seven_state_sft"}:
        raise ValueError(args.arm)

    rank, local_rank, world_size, device = setup_distributed()
    seed_everything(int(protocol["seed"]), rank)
    patch_torch_compat()
    if not args.smoke and world_size != int(protocol["training"]["world_size"]):
        raise RuntimeError(f"formal run requires world_size={protocol['training']['world_size']}")

    output_dir = Path(args.output_dir).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    try:
        model, processor, trainable_names, trainable_count, total_count = build_model(
            protocol, device, args.attn_implementation
        )
        actual_attn = args.attn_implementation
    except Exception:
        if not args.smoke or args.attn_implementation == "sdpa":
            raise
        torch.cuda.empty_cache()
        model, processor, trainable_names, trainable_count, total_count = build_model(
            protocol, device, "sdpa"
        )
        actual_attn = "sdpa"

    initial_adapter_state = {
        name: param.detach().float().cpu().numpy().tobytes()
        for name, param in model.named_parameters() if param.requires_grad
    }
    initial_digest = hashlib.sha256()
    for name in sorted(initial_adapter_state):
        initial_digest.update(name.encode("utf-8"))
        initial_digest.update(initial_adapter_state[name])
    initial_adapter_sha256 = initial_digest.hexdigest()
    expected_initial_sha = protocol["training"].get("initial_adapter_state_sha256", "")
    if expected_initial_sha and initial_adapter_sha256 != expected_initial_sha:
        raise RuntimeError(
            "initial adapter digest mismatch: "
            f"expected {expected_initial_sha}, got {initial_adapter_sha256}"
        )

    artifact_spec = protocol.get("artifacts", {})
    train_manifest = Path(artifact_spec.get("train_manifest", protocol["data"]["train_manifest"]))
    expected_manifest_sha = artifact_spec.get("train_manifest_sha256")
    if expected_manifest_sha and sha256_file(train_manifest) != expected_manifest_sha:
        raise RuntimeError("train manifest hash mismatch")
    rows = load_jsonl(train_manifest)
    weights = None
    weights_sha256 = None
    if args.weights_json:
        weights_path = Path(args.weights_json).resolve()
        weights = json.loads(weights_path.read_text(encoding="utf-8"))
        if weights.get("status") != "FROZEN":
            raise RuntimeError("weights artifact is not FROZEN")
        weights = weights["state_class_weight"]
        weights_sha256 = sha256_file(weights_path)
    rng = np.random.default_rng(int(protocol["seed"]))
    # The original IEMOCAP emitter repeats the same training split for the
    # declared number of epochs.  The generic runner previously ignored this
    # field, which silently reduced MatchHIST to one epoch.  Expand the order
    # before distributed sharding so optimizer-step counts are auditable.
    epochs = int(protocol["training"].get("epochs", 1))
    order = rng.permutation(len(rows)).tolist() * epochs
    grad_accum = 1 if args.smoke else int(protocol["training"]["gradient_accumulation"])
    global_unit = world_size * grad_accum
    usable = (len(order) // global_unit) * global_unit
    order = order[:usable]
    local_indices = order[rank:usable:world_size]
    if args.smoke:
        if args.smoke_sample_id:
            matches = [index for index, row in enumerate(rows) if row["sample_id"] == args.smoke_sample_id]
            if len(matches) != 1:
                raise RuntimeError(f"smoke sample not uniquely found: {args.smoke_sample_id}")
            local_indices = matches if rank == 0 else []
        else:
            local_indices = local_indices[: max(1, args.max_microsteps)]
    # IEMOCAP emitter recipe (SSOT 10.18.258): every sample contributes the answer CE of ALL seven
    # states, each with weight 1/7 (paper2_harm_screen_v1.py: ce = out.loss / 7 per state).  The
    # sample is expanded into seven consecutive microsteps and the accumulation is scaled by seven,
    # so one optimizer step still averages over the same samples and the step count is unchanged.
    all_states = protocol["training"].get("state_loss") == "all_states_mean"
    if all_states:
        local_items = [(index, state) for index in local_indices for state in STATES]
        grad_accum *= len(STATES)
    else:
        local_items = [(index, None) for index in local_indices]
    expected_microsteps = len(local_items)
    total_optimizer_steps = math.ceil(expected_microsteps / grad_accum)
    expected_steps = protocol["training"].get("optimizer_steps")
    if not args.smoke and expected_steps is not None and total_optimizer_steps != int(expected_steps):
        raise RuntimeError(
            "optimizer-step budget mismatch: "
            f"expected {int(expected_steps)}, got {total_optimizer_steps}"
        )

    train_model: torch.nn.Module = model
    if world_size > 1:
        train_model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )

    optimizer = torch.optim.AdamW(
        (param for param in train_model.parameters() if param.requires_grad),
        lr=float(protocol["training"]["learning_rate"]),
        betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: max(0.0, 1.0 - step / max(1, total_optimizer_steps)),
    )
    train_model.train()
    optimizer.zero_grad(set_to_none=True)

    counters = {
        "microsteps": 0,
        "optimizer_steps": 0,
        "input_tokens": 0,
        "supervised_tokens": 0,
        "audio_tokens": 0,
        "video_tokens": 0,
        "audio_samples": 0,
    }
    # ---- relational edge objective -------------------------------------------------------
    EDGE_LIST = [("A", "AV"), ("V", "AV"), ("A", "AT"), ("T", "AT"), ("V", "VT"), ("T", "VT"),
                 ("AV", "AVT"), ("AT", "AVT"), ("VT", "AVT")]
    edge_w, edge_parents, edge_gate, edge_sbar = None, {}, {}, None
    if args.relational_edge:
        if weights is None:
            raise RuntimeError("--relational-edge needs --weights-json")
        spec = json.loads(weights_path.read_text())
        if "edge_class_weight" not in spec:
            raise RuntimeError(f"{weights_path} carries no edge_class_weight")
        edge_w = spec["edge_class_weight"]
        # The base objective is uniform for this arm: the relation is supervised directly, so
        # nothing rescales the per-state cross-entropy.
        weights = None
        for parent, child in EDGE_LIST:
            edge_parents.setdefault(child, []).append(parent)
        # g = 1 iff the frozen diagnostic predictor is already correct at the parent state.
        for path in spec["scores"]:
            for line in Path(path).open(encoding="utf-8"):
                record = json.loads(line)
                if record.get("status") != "success":
                    continue
                gt = record["gt"]
                edge_gate[record["sample_id"]] = {
                    st: (value["official_pred"] == gt) for st, value in record["states"].items()}
        total, seen = 0.0, 0
        for sample in rows:
            gates = edge_gate.get(sample["sample_id"])
            if gates is None:
                continue
            seen += 1
            for state in STATES:                       # a microstep draws one state per sample
                total += sum(edge_w[f"{parent}->{state}|{sample['gt']}"]
                             for parent in edge_parents.get(state, []) if gates.get(parent))
        edge_sbar = total / max(seen * len(STATES), 1)
        if edge_sbar <= 0:
            raise RuntimeError("no protected parent evidence for the relational objective")
        if rank == 0:
            print(f"relational edge objective: {seen} gated samples, s_bar = {edge_sbar:.4f}",
                  flush=True)

    # ---- Edge-aware Margin Preservation -------------------------------------------------
    emp = None
    if args.edge_margin:
        if weights is None or not args.fc_cache:
            raise RuntimeError("--edge-margin needs --weights-json and --fc-cache")
        spec = json.loads(weights_path.read_text())
        if "edge_class_weight" not in spec:
            raise RuntimeError(f"{weights_path} carries no edge_class_weight")
        EDGE_LIST = [("A", "AV"), ("V", "AV"), ("A", "AT"), ("T", "AT"), ("V", "VT"), ("T", "VT"),
                     ("AV", "AVT"), ("AT", "AVT"), ("VT", "AVT")]
        parents_of = {}
        for parent, child in EDGE_LIST:
            parents_of.setdefault(child, []).append(parent)
        # The base objective stays uniform: the margin constraint carries the intervention.
        weights = None
        label_index = {c: i for i, c in enumerate(LABEL_SET)}
        parent_margin = {}
        for line in Path(args.fc_cache).open(encoding="utf-8"):
            record = json.loads(line)
            if record.get("status") not in (None, "success"):
                continue
            y = label_index[record["gt"]]
            per_state = {}
            for st, value in record["states"].items():
                v = np.array([value["candidate_scores"][c] for c in LABEL_SET], dtype=np.float64)
                v = v - v.max()
                q = np.exp(v); q /= q.sum()
                t = q[y]; q[y] = -1.0
                per_state[st] = float(t - q.max())      # > 0 exactly when the state is correct
            parent_margin[record["sample_id"]] = per_state
        # rho: the share of (sample, state) pairs that have at least one gated parent, so the
        # per-microstep term averages to the weighted mean over gated pairs.
        active = total = 0
        for sample in rows:
            margins = parent_margin.get(sample["sample_id"])
            if margins is None:
                continue
            for st in STATES:
                total += 1
                if any(margins.get(par, -1.0) > 0 and
                       (not args.edge_margin_harm_only or margins.get(st, 0.0) <= 0)
                       for par in parents_of.get(st, [])):
                    active += 1
        rho = active / max(total, 1)
        if rho <= 0:
            raise RuntimeError("no gated edges: the diagnostic predictor is never right at a parent")
        from run_meld_qwen_sft_selection_eval import prepare_view  # noqa: F401
        label_ids = [processor.tokenizer(c, add_special_tokens=False).input_ids for c in LABEL_SET]
        source_protocol = json.loads(
            Path(protocol["source"]["platform_protocol"]).read_text(encoding="utf-8"))
        emp = {"w": spec["edge_class_weight"], "parents": parents_of, "margin": parent_margin,
               "rho": rho, "ids": label_ids, "index": label_index,
               "source_protocol": source_protocol, "checked": False,
               "harm_only": bool(args.edge_margin_harm_only)}
        if rank == 0:
            print(json.dumps({"emp_gated_state_share": round(rho, 4),
                              "emp_scoring": "exact cached-prefix candidate scoring",
                              "emp_gate": ("parent-correct" if not args.edge_margin_harm_only
                                           else "frozen-parent-correct-and-child-wrong"),
                              "emp_label_token_counts": {c: len(label_ids[i])
                                                         for i, c in enumerate(LABEL_SET)},
                              "emp_samples_with_parent_margins": len(parent_margin)}), flush=True)
    emp_stats = {"microsteps": 0, "active": 0, "gated": 0, "loss": 0.0, "violation": 0.0}

    base_already_done = False
    edge_stats = {"microsteps": 0, "gated": 0, "active": 0,
                  "edge_loss": 0.0, "gap": 0.0, "multiplier": 0.0}
    state_counts = {state: 0 for state in STATES}
    losses: list[float] = []
    started = time.time()
    full_media: dict[int, tuple] = {}
    for local_position, (row_index, forced_state) in enumerate(local_items):
        sample = rows[int(row_index)]
        state = forced_state or (sample["avt_only_state"] if args.arm == "avt_only_sft" else sample["seven_state"])
        shared_decoded = None
        if forced_state is not None:
            # decode the media once per sample and hand each state only its observed part
            if int(row_index) not in full_media:
                full_media.clear(); media = Path(sample["video_path"])
                full_media[int(row_index)] = (decode_audio(media),
                                              decode_video(media, int(protocol["training"]["video_total_pixels"])))
            audio_full, video_full = full_media[int(row_index)]
            shared_decoded = (audio_full if STATE_SPEC[state]["audio"] else None,
                              video_full if STATE_SPEC[state]["video"] else None)
        if emp is not None:
            # The scoring pass needs the same media as the base pass; decode once.
            spec_here = STATE_SPEC[state]
            media = Path(sample["video_path"])
            shared_decoded = (
                decode_audio(media) if spec_here["audio"] else None,
                decode_video(media, int(protocol["training"]["video_total_pixels"]))
                if spec_here["video"] else None)
        if edge_w is not None and edge_parents.get(state):
            gates_here = edge_gate.get(sample["sample_id"], {})
            if any(gates_here.get(parent) for parent in edge_parents[state]):
                # A child's modality set is a superset of each parent's, so decoding what the
                # child needs covers every parent forward and nothing is decoded twice.
                spec_here = STATE_SPEC[state]
                media = Path(sample["video_path"])
                shared_decoded = (
                    decode_audio(media) if spec_here["audio"] else None,
                    decode_video(media, int(protocol["training"]["video_total_pixels"]))
                    if spec_here["video"] else None)
        inputs, stats = prepare_training_input(
            processor,
            sample,
            state,
            int(protocol["training"]["video_total_pixels"]),
            decoded=shared_decoded,
        )
        inputs = move_inputs(inputs, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = train_model(**inputs, use_cache=False, use_audio_in_video=False)
            loss = output.loss
        if weights is not None:
            key = f"{state}|{sample['gt']}"
            if key not in weights:
                raise RuntimeError(f"missing ETR weight: {key}")
            loss = loss * float(weights[key])
        if edge_w is not None:
            # DDP broadcasts module buffers on every forward.  The number of parent forwards
            # differs from rank to rank, so calling the WRAPPER here desynchronises the
            # collectives and the run hangs on a BROADCAST (observed 2026-09-16 10:49).  The
            # parent pass needs no gradient and no synchronisation, so it goes to the module.
            base_model = (train_model.module
                          if isinstance(train_model, DistributedDataParallel) else train_model)
            # The seven states of a sample are separate microsteps here, so the parent loss is
            # not already in hand as it is on IEMOCAP: it is recomputed under no_grad, which is
            # the same quantity sg() would have produced.  Single-modality states have no
            # parent and add nothing.
            gates = edge_gate.get(sample["sample_id"], {})
            terms, coefficients = [], []
            n_gated = n_active = 0
            edge_value = gap_value = active_coefficient = 0.0
            for parent in edge_parents.get(state, []):
                if not gates.get(parent):
                    continue
                parent_inputs, _ = prepare_training_input(
                    processor, sample, parent,
                    int(protocol["training"]["video_total_pixels"]),
                    decoded=shared_decoded)
                parent_inputs = move_inputs(parent_inputs, device)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    parent_loss = base_model(**parent_inputs, use_cache=False,
                                             use_audio_in_video=False).loss.detach()
                del parent_inputs
                coefficient = float(edge_w[f"{parent}->{state}|{sample['gt']}"]) / edge_sbar
                n_gated += 1
                coefficients.append(coefficient)
                terms.append(coefficient * torch.relu(output.loss - parent_loss))
            n_active = sum(1 for t in terms if float(t) > 0.0)
            active_coefficient = sum(c for c, t in zip(coefficients, terms) if float(t) > 0.0)
            gap_value = sum(float(t) / c for c, t in zip(coefficients, terms) if c > 0)
            if terms:
                edge_loss = torch.stack(terms).sum()
                if not torch.isfinite(edge_loss):
                    raise RuntimeError(f"non-finite edge loss on {sample['sample_id']}")
                loss = loss + edge_loss
                edge_value = float(edge_loss.detach())
        if emp is not None and args.split_backward:
            # Free the base graph before the scoring graph is built.
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite base loss on {sample['sample_id']}")
            (loss / grad_accum).backward()
            losses.append(float(loss.detach().cpu()))
            base_already_done = True
            loss = None
        else:
            base_already_done = False
        if emp is not None:
            # The anchor-forced pass runs on every microstep, gated or not, so each rank issues
            # exactly the same number of DDP forwards and the collectives stay in step.
            # Prompt-only inputs, then the evaluator's own cached-prefix scoring, so each
            # candidate is conditioned on its own tokens.  The module is called a fixed number
            # of times per microstep (1 + longest label - 1), identical on every rank.
            prompt_inputs, _ = prepare_view(emp["source_protocol"], processor, dict(sample),
                                            state, shared_decoded[0], shared_decoded[1])
            prompt_inputs = move_inputs(prompt_inputs, device)
            target = emp["index"][sample["gt"]]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scored = candidate_scores(train_model, prompt_inputs, emp["ids"])
                child_margin = margin_of(scored, target)
            if not emp["checked"]:
                # One-time correctness gate: the scores this objective optimises must be the
                # scores the official decision rule ranks.  The previous implementation read
                # three of seven MELD candidates under the wrong prefix and nothing caught it,
                # so the two are compared once, on the first gated microstep of every rank.
                emp["checked"] = True
                from run_planb_meld_qwen_pilot import cached_sequence_scores
                inner_model = (train_model.module
                               if isinstance(train_model, DistributedDataParallel) else train_model)
                # The reference scorer also needs the cache, so checkpointing is off for it too.
                was_on = bool(getattr(inner_model, "is_gradient_checkpointing", False))
                if was_on:
                    inner_model.gradient_checkpointing_disable()
                try:
                    exact, _ = cached_sequence_scores(
                        inner_model, prompt_inputs,
                        {c: emp["ids"][i] for i, c in enumerate(LABEL_SET)})
                finally:
                    if was_on:
                        inner_model.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={"use_reentrant": False})
                mine = scored.detach().float().tolist()
                worst = max(abs(mine[i] - exact[c]) for i, c in enumerate(LABEL_SET))
                if worst > 0.05:
                    raise RuntimeError(
                        f"candidate scoring disagrees with the official scorer by {worst:.4f}: "
                        f"mine={[round(v, 4) for v in mine]} "
                        f"exact={[round(exact[c], 4) for c in LABEL_SET]}")
                if rank == 0:
                    print(json.dumps({"emp_scoring_gate": "PASS",
                                      "max_abs_difference_vs_official_scorer": round(worst, 6)}),
                          flush=True)
            del prompt_inputs, scored
            margins = emp["margin"].get(sample["sample_id"], {})
            numer = child_margin * 0.0
            denom = 0.0
            gated = 0
            for parent in emp["parents"].get(state, []):
                m_parent = margins.get(parent, -1.0)
                if m_parent <= 0.0:                      # the parent is not reliable evidence
                    continue
                if emp["harm_only"] and margins.get(state, 0.0) > 0.0:
                    continue
                coefficient = float(emp["w"][f"{parent}->{state}|{sample['gt']}"])
                hinge = (torch.relu(m_parent - child_margin)
                         if not emp["harm_only"] else torch.relu(-child_margin))
                numer = numer + coefficient * hinge
                denom += coefficient
                gated += 1
            emp_stats["microsteps"] += 1
            emp_stats["gated"] += gated
            if denom > 0:
                # weighted MEAN over incoming edges, so AVT's three parents do not outweigh the
                # two of AV/AT/VT, then divided by rho to average over gated pairs only
                term = numer / denom / emp["rho"]
                if not torch.isfinite(term):
                    raise RuntimeError(f"non-finite EMP term on {sample['sample_id']}")
                loss = term if base_already_done else loss + term
                emp_stats["loss"] += float(term.detach())
                emp_stats["active"] += int(float(term.detach()) > 0)
                emp_stats["violation"] += float(term.detach() > 0)
            else:
                zero = numer * 0.0                       # keeps every parameter used under DDP
                loss = zero if base_already_done else loss + zero
            if rank == 0 and emp_stats["microsteps"] in (1, 5, 20, 100, 500, 2000):
                n = emp_stats["microsteps"]
                print(json.dumps({"emp_audit_microsteps": n,
                                  "mean_base_loss": sum(losses[-n:]) / max(len(losses[-n:]), 1),
                                  "mean_emp_term": emp_stats["loss"] / n,
                                  "gated_state_rate": emp_stats["gated"] / n,
                                  "margin_violation_rate": emp_stats["active"] / max(n, 1)}),
                      flush=True)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss on {sample['sample_id']}: {loss}")
        if edge_w is not None:
            edge_stats["microsteps"] += 1
            edge_stats["gated"] += n_gated
            edge_stats["active"] += n_active
            edge_stats["edge_loss"] += edge_value
            edge_stats["gap"] += gap_value
            edge_stats["multiplier"] += 1.0 + active_coefficient
            if rank == 0 and edge_stats["microsteps"] in (1, 5, 20, 100, 500, 2000):
                n = edge_stats["microsteps"]
                print(json.dumps({"edge_audit_microsteps": n,
                                  "mean_base_loss": sum(losses[-n:]) / max(len(losses[-n:]), 1),
                                  "mean_edge_loss": edge_stats["edge_loss"] / n,
                                  "active_hinge_rate": edge_stats["active"] / max(edge_stats["gated"], 1),
                                  "mean_gradient_multiplier": edge_stats["multiplier"] / n}), flush=True)
        (loss / grad_accum).backward()
        if not base_already_done:
            losses.append(float(loss.detach().cpu()))
        counters["microsteps"] += 1
        state_counts[state] += 1
        for key, value in stats.items():
            counters[key] += int(value)

        boundary = ((local_position + 1) % grad_accum == 0) or (local_position + 1 == len(local_items))
        if boundary:
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), float(protocol["training"]["max_grad_norm"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            counters["optimizer_steps"] += 1
            if rank == 0 and (args.smoke or counters["optimizer_steps"] % args.log_every == 0):
                print(json.dumps({
                    "arm": args.arm,
                    "optimizer_step": counters["optimizer_steps"],
                    "total_optimizer_steps": total_optimizer_steps,
                    "loss": losses[-1],
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.time() - started,
                }), flush=True)
            checkpoint_due = (
                int(args.checkpoint_every) > 0
                and (counters["optimizer_steps"] % int(args.checkpoint_every) == 0
                     or counters["optimizer_steps"] == total_optimizer_steps)
            )
            if checkpoint_due:
                if rank == 0:
                    save_adapter_checkpoint(train_model, processor, output_dir, counters["optimizer_steps"])
                    print(json.dumps({
                        "checkpoint_saved": counters["optimizer_steps"],
                        "output": str(output_dir / f"checkpoint-{counters['optimizer_steps']}")
                    }), flush=True)
                if dist.is_initialized():
                    dist.barrier()

    reduced_counters = {key: all_reduce_sum(value, device) for key, value in counters.items()}
    # Optimizer steps are synchronized across DDP ranks and must not be summed.
    reduced_counters["optimizer_steps"] = counters["optimizer_steps"]
    reduced_states = {key: all_reduce_sum(value, device) for key, value in state_counts.items()}
    local_loss_sum = torch.tensor([sum(losses), len(losses)], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)

    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        unwrapped = train_model.module if isinstance(train_model, DistributedDataParallel) else train_model
        adapter_dir = output_dir / "adapter"
        unwrapped.save_pretrained(adapter_dir, safe_serialization=True)
        processor.save_pretrained(output_dir / "processor")
        manifest = {
            "status": "SMOKE_COMPLETE" if args.smoke else "COMPLETE",
            "arm": args.arm,
            "seed": int(protocol["seed"]),
            "protocol": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "world_size": world_size,
            "gradient_accumulation": grad_accum,
            "attention_implementation": actual_attn,
            "trainable_parameter_count": trainable_count,
            "total_parameter_count_with_adapter": total_count,
            "trainable_fraction": trainable_count / total_count,
            "trainable_parameter_names": trainable_names,
            "initial_adapter_sha256": initial_adapter_sha256,
            "adapter_dir": str(adapter_dir),
            "adapter_sha256": sha256_files(adapter_dir),
            "weights_json": str(Path(args.weights_json).resolve()) if args.weights_json else None,
            "edge_margin": ({"objective": ("L_uniform + mean over gated (i,S) of the weighted "
                                          "mean over incoming edges of [m_parent - m_child]_+"
                                          if not args.edge_margin_harm_only else
                                          "L_uniform + mean over frozen harmful (i,S) of the "
                                          "weighted mean over incoming edges of [-m_child]_+"),
                             "rho": emp["rho"], "delta": 0.0, "lambda_emp": 1.0,
                             "harm_only": bool(args.edge_margin_harm_only),
                             "parent_margin_source": args.fc_cache,
                             "base_cross_entropy": "uniform; no state or class weights"}
                            if args.edge_margin else None),
            "relational_edge": ({"objective": "L_uniform + mean over microsteps of "
                                              "sum_e g*w*[l_child - sg(l_parent)]_+ / s_bar",
                                 "s_bar": edge_sbar, "margin": 0.0, "lambda_edge": 1.0,
                                 "base_cross_entropy": "uniform; no state or class weights"}
                                if args.relational_edge else None),
            "weights_sha256": weights_sha256,
            "counters_global": reduced_counters,
            "state_counts_global": reduced_states,
            "mean_microstep_loss": float(local_loss_sum[0].item() / max(1.0, local_loss_sum[1].item())),
            "elapsed_seconds": time.time() - started,
            "official_dev_evaluations": 0,
            "official_test_evaluations": 0,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "status": manifest["status"],
            "arm": args.arm,
            "trainable_parameter_count": trainable_count,
            "counters_global": reduced_counters,
            "state_counts_global": reduced_states,
            "mean_microstep_loss": manifest["mean_microstep_loss"],
            "output": str(output_dir),
        }, ensure_ascii=False, indent=2), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default=str(ROOT / "work_dir" / "meld_qwen_sft_platform_v1" / "protocol" / "protocol.json"),
    )
    parser.add_argument("--arm", required=True, choices=["avt_only_sft", "seven_state_sft"])
    parser.add_argument("--split-backward", action="store_true",
                        help="Back-propagate the base loss before building the margin term's "
                             "graph, so the two are never held at once.  Peak memory becomes the "
                             "larger of the two passes instead of their sum; gradients accumulate "
                             "so the result is unchanged, at the cost of a second all-reduce per "
                             "microstep.")
    parser.add_argument("--edge-margin", action="store_true",
                        help="Train Edge-aware Margin Preservation (SSOT 10.18.122): the base "
                             "cross-entropy stays uniform and a one-sided hinge on the DECISION "
                             "MARGIN, [m_parent - m_child]_+, is added on the gated edges. The "
                             "parent margin comes from the frozen diagnostic cache, so no parent "
                             "forward is needed. Requires --weights-json with edge_class_weight "
                             "and --fc-cache.")
    parser.add_argument("--edge-margin-harm-only", action="store_true",
                        help="EMP-v2: apply the margin barrier only to frozen harmful edges "
                             "(parent correct and child wrong), using [-m_child]_+ rather "
                             "than preserving every positive parent margin.")
    parser.add_argument("--fc-cache", default="",
                        help="eval_c6_states output for the frozen diagnostic predictor on this "
                             "fold's training manifest: supplies the parent margins and the gate.")
    parser.add_argument("--relational-edge", action="store_true",
                        help="Train the relational ETR objective (SSOT 10.18.120): the base "
                             "cross-entropy stays exactly uniform and a one-sided hinge "
                             "[l_child - sg(l_parent)]_+ is added on the gated edges. Requires "
                             "--weights-json to carry edge_class_weight.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-microsteps", type=int, default=1)
    parser.add_argument("--smoke-sample-id", default="")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--checkpoint-every", type=int, default=0,
        help=("Save adapter/processor snapshots every N optimizer steps for "
              "validation-based checkpoint selection; 0 saves only the final adapter.")
    )
    parser.add_argument("--weights-json", default="", help="Frozen state-class weight artifact for ETR")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
