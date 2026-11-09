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
    frozen.bind_labels(screen)   # a six-class run directory is served without forking the pipeline
    return frozen


def make_training_order(count, epoch_budget=0, seed=SEED):
    base = np.random.default_rng(seed).permutation(count).tolist()
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


class _StopVisual(Exception):
    """Raised inside the vision-tower hook to end a forward once the visual feature is read."""


VISUAL_GRAB = {"z": None, "stop": False}


def install_visual_hook(vision_tower):
    """Persistent hook: every vision-tower forward stores its mean-pooled output (with its
    autograd graph intact) in VISUAL_GRAB['z']; when VISUAL_GRAB['stop'] is set it also ends
    the forward, so the stop-gradient bank pass never runs the language model."""
    def hook(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if hasattr(t, "last_hidden_state"):
            t = t.last_hidden_state
        VISUAL_GRAB["z"] = t.reshape(-1, t.shape[-1]).float().mean(0)
        if VISUAL_GRAB["stop"]:
            raise _StopVisual()
    return vision_tower.register_forward_hook(hook)


def capture_visual(model, vision_tower, frozen, processor, row, video_pixels, decoded, torch,
                   grad):
    """Stop-gradient visual feature of one utterance (the CD-SupCon bank).  Dropout is zeroed
    through frozen.decision_mode, as for every scoring pass in this trainer."""
    VISUAL_GRAB["z"] = None
    VISUAL_GRAB["stop"] = True
    try:
        with torch.set_grad_enabled(grad), frozen.decision_mode(model):
            try:
                frozen.forward(model, processor, row, "V", video_pixels, decoded, False)
            except _StopVisual:
                pass
    finally:
        VISUAL_GRAB["stop"] = False
    if VISUAL_GRAB["z"] is None:
        raise RuntimeError(f"vision tower did not run for {row['sample_id']}")
    return VISUAL_GRAB["z"].detach()


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
    """Return the offline correct-immediate-parent target for one child state, or None.

    Parents are restricted to |S \\ P| = 1, i.e. exactly the edges whose
    correct-to-wrong events define the harm counts behind A's weights.
    """
    current_set = set(state)
    proper = [name for name in references
              if set(name) < current_set and len(current_set - set(name)) == 1]
    if not proper or int(references[state].argmax()) == target:
        return None
    good = [name for name in proper if int(references[name].argmax()) == target]
    if not good:
        return None
    value = torch.stack([torch.softmax(references[name], dim=-1) for name in good]).mean(0)
    if mode == "hard":
        # N1 control: same recoverable event, ground-truth one-hot instead of the
        # correct-parent posterior.  KL(one-hot || p) equals candidate-normalized CE.
        value = torch.zeros_like(value)
        value[target] = 1.0
    if mode == "cm":
        confidence = value[target]
        value = torch.ones_like(value) * ((1.0 - confidence) / (len(value) - 1))
        value[target] = confidence
    return value.detach()


def lora_grad_vector(model, torch):
    parts = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            parts.append(torch.zeros(parameter.numel(), device="cuda", dtype=torch.float32)
                         if parameter.grad is None
                         else parameter.grad.detach().float().reshape(-1))
    return torch.cat(parts)


DRO_EDGES = [("A", "AV"), ("V", "AV"), ("A", "AT"), ("T", "AT"), ("V", "VT"), ("T", "VT"),
             ("AV", "AVT"), ("AT", "AVT"), ("VT", "AVT")]


class DroState:
    """Prior-guided KL-regularized transition group DRO (and ablation variants).

    Groups: TDRO/FULL g = (P->S, c) over the nine immediate edges; SDRO g = (S, c).
    Risk: TDRO/FULL joint rate of P correct and S wrong among class-c samples;
          SDRO error rate of S on class c.  Online risk uses the current model's
          no-grad predictions, exponentially decayed counts (beta) and alpha-pseudo-count
          shrinkage toward the pooled rate; z = standardized risk clipped to [-3, 3].
    Prior: FULL uses the frozen-F0 edge x class harm rates of the training cache
           (log pi = kappa * z_F0); TDRO/SDRO use a uniform prior.
    Update (after warm-up, every step): log q <- (1-rho) log q + rho * target,
           target = s * log pi + (1-s) * kappa * z with prior share s = 0.5 (FULL) / 0
           (TDRO, SDRO); the fixed point log q* = target keeps a single-kappa scale and is
           KL-regularized group DRO with reference distribution pi.  During warm-up q = pi.
    q is normalized to mean 1 over groups and clipped to [q_min, q_max]; child-state
    weight = mean of its incoming-edge q (singletons 1) for TDRO/FULL, q itself for SDRO.
    Weight normalization (weight_normalization):
      "global" rescales the whole table by one class-frequency-weighted scalar.  The mean
        weight over the training exposure is 1, but nothing pins the per-class mean, so a
        class whose transitions are riskier than average also receives more total supervision.
        That is a class re-weighting, which the method never claimed to perform.
      "class" is class-mass preserving: each class is divided by its own Z_c = (1/7) sum_S
        w[S,c], so the mean weight over the seven states is exactly 1 for every class.  Every
        utterance is presented in all seven states, so class exposure is uniform over states
        and this pins the class prior of the objective to the control's, leaving only the
        within-class redistribution across modality states.
    Weights are then clipped to [w_min, w_max].
    """

    def __init__(self, arm, states, labels, class_freq, args, prior_counts=None):
        self.arm, self.states, self.labels = arm, list(states), list(labels)
        self.norm_mode = getattr(args, "weight_normalization", "global")
        if self.norm_mode not in ("global", "class"):
            raise ValueError(f"unknown weight normalization {self.norm_mode!r}")
        self.K = len(labels)
        self.freq = np.asarray(class_freq, dtype=float)
        self.units = list(DRO_EDGES) if arm in {"TDRO", "FULL"} else [(st, st) for st in self.states]
        self.G = len(self.units) * self.K
        self.beta, self.alpha, self.kappa, self.rho = args.dro_beta, args.dro_alpha, args.dro_kappa, args.dro_rho
        self.warmup = args.dro_warmup
        self.q_min, self.q_max, self.w_min, self.w_max = args.dro_qmin, args.dro_qmax, args.dro_wmin, args.dro_wmax
        self.B = np.zeros(self.G)
        self.D = np.zeros(self.G)
        self.H = np.zeros(self.G)
        if arm == "FULL":
            self.prior_H = self.shrunk(prior_counts[:self.G], prior_counts[self.G:])
            self.log_prior = self.kappa * self.z(self.prior_H)
        else:
            self.prior_H = None
            self.log_prior = np.zeros(self.G)
        # FULL: the fixed point averages the F0 prior and the online evidence in log space
        self.prior_share = 0.5 if arm == "FULL" else 0.0
        self.log_q = self.log_prior.copy()
        self.active = False
        self.refresh()

    def shrunk(self, b, d):
        pooled = b.sum() / max(d.sum(), 1e-9)
        return (b + self.alpha * pooled) / (d + self.alpha)

    @staticmethod
    def z(h):
        return np.clip((h - h.mean()) / (h.std() + 1e-8), -3.0, 3.0)

    def local_counts(self, pred, target):
        out = np.zeros(2 * self.G)
        for u, (a, b) in enumerate(self.units):
            g = u * self.K + target
            out[self.G + g] += 1.0
            if self.arm in {"TDRO", "FULL"}:
                out[g] += float(pred[a] == target and pred[b] != target)
            else:
                out[g] += float(pred[a] != target)
        return out

    def refresh(self):
        q = np.exp(self.log_q)
        q = np.clip(q / q.mean(), self.q_min, self.q_max)
        self.q = q
        qm = q.reshape(len(self.units), self.K)
        w = np.ones((len(self.states), self.K))
        if self.arm in {"TDRO", "FULL"}:
            for si, st in enumerate(self.states):
                rows = [u for u, (p, c) in enumerate(self.units) if c == st]
                if rows:
                    w[si] = qm[rows].mean(0)
        else:
            w = qm.copy()
        if self.norm_mode == "class":
            mass = w.mean(0, keepdims=True)
            if not np.all(mass > 0):
                raise RuntimeError("non-positive class mass in the weight table")
            w = w / mass
        else:
            w = w * (1.0 / (w * self.freq[None, :]).sum(1).mean())
        self.w = np.clip(w, self.w_min, self.w_max)
        self.clipped = int(((w <= self.w_min) | (w >= self.w_max)).sum())
        if self.norm_mode == "class":
            # The clip must not silently reintroduce the drift the mode exists to remove.  Any
            # clipped entry moves its class mean, so this also certifies that clipping was inert;
            # the paper may only claim strict class-mass preservation while both hold.
            drift = np.abs(self.w.mean(0) - 1.0).max()
            if self.clipped or drift > 1e-9:
                raise RuntimeError(
                    f"class mass not preserved: {self.clipped} weights hit the clip "
                    f"[{self.w_min}, {self.w_max}], max drift {drift:.3e}")

    def update(self, counts, step):
        b, d = counts[:self.G], counts[self.G:]
        self.B = self.beta * self.B + b
        self.D = self.beta * self.D + d
        self.H = self.shrunk(self.B, self.D)
        if step >= self.warmup:
            self.active = True
            target = self.prior_share * self.log_prior + (1 - self.prior_share) * self.kappa * self.z(self.H)
            self.log_q = (1 - self.rho) * self.log_q + self.rho * target
        self.refresh()

    def snapshot(self, step):
        return {"step": step, "active": self.active, "H": self.H.round(5).tolist(),
                "q": self.q.round(5).tolist(), "w": self.w.round(5).tolist(),
                "w_min": float(self.w.min()), "w_max": float(self.w.max()),
                "q_at_clip": int(((self.q <= self.q_min + 1e-9) | (self.q >= self.q_max - 1e-9)).sum()),
                "units": [f"{a}->{b}" if self.arm in {"TDRO", "FULL"} else a for a, b in self.units],
                "labels": self.labels}


def f0_prior_counts(cached, labels, states, granularity="edge", definition="joint"):
    """Harm counts of the frozen diagnostic training cache, in the FULL prior's group layout.

    granularity="edge" is the deployed method: one group per (immediate edge, class).
    granularity="state" is the scale-matched control requested in review: the SAME counts are
    pooled over the incoming edges of each child state before the identical z-score and
    log-linear map are applied, so the two arms differ only in the grouping, never in the
    weight range.  Edges keep their layout so the downstream code is unchanged; every edge of
    a child state simply receives that state's pooled numerator and denominator.
    """
    K = len(labels)
    G = len(DRO_EDGES) * K
    out = np.zeros(2 * G)
    for record in cached.values():
        y = labels.index(record["gt"])
        pred = {st: int(np.argmax(record["states"][st])) for st in states}
        for u, (a, b) in enumerate(DRO_EDGES):
            g = u * K + y
            if definition == "conditional":
                # P(child wrong | parent correct, y = c).  The denominator counts only the
                # opportunities that could actually be harmed, which removes the confound
                # between harm risk and how often the model is already correct at the parent.
                if pred[a] == y:
                    out[G + g] += 1.0
                    out[g] += float(pred[b] != y)
            else:
                # P(parent correct AND child wrong | y = c): a joint event frequency, so a
                # class the model usually gets right at the parent accumulates more events
                # simply by having more opportunities.
                out[G + g] += 1.0
                out[g] += float(pred[a] == y and pred[b] != y)
    if granularity == "edge":
        return out
    if granularity == "state":
        groups = {}
        for u, (_, child) in enumerate(DRO_EDGES):
            groups.setdefault(child, []).append(u)
    elif granularity == "class":
        groups = {"all": list(range(len(DRO_EDGES)))}
    else:
        raise ValueError(f"unknown prior granularity {granularity!r}")
    pooled = out.copy()
    for indices in groups.values():
        for c in range(K):
            num = sum(out[u * K + c] for u in indices)
            den = sum(out[G + u * K + c] for u in indices)
            for u in indices:
                pooled[u * K + c] = num / len(indices)
                pooled[G + u * K + c] = den / len(indices)
    return pooled


def cached_parent_correct(cached, row, parent, target, frozen) -> bool:
    """Did the frozen diagnostic predictor already answer this sample correctly at the parent?"""
    record = cached[row["sample_id"]]
    return int(np.argmax(record["states"][parent])) == target


def relational_edge_setup(cached, labels, states, risk_definition="conditional"):
    """Weights and the normaliser for the one-sided edge objective (SSOT 10.18.120).

    The finding behind ETR is a RELATION: the model is already right at a parent state and
    adding a modality breaks it.  Weighting the child's cross-entropy optimises each state
    independently and never tells the model that the child must not be worse than the parent;
    on MELD that intervention measures out at essentially zero even though the risk structure
    itself is reproducible (split-half r = 0.808).  This objective states the relation directly:

        L_edge = sum_{i,e} g_{i,e} w_{e,y_i} [ l_{i,S} - sg(l_{i,P}) ]_+  /  sum_{i,e} g w

    g_{i,e} = 1 iff the frozen diagnostic predictor is already correct at the parent, so only
    genuine "evidence that could be destroyed" is protected.  The hinge is one sided: a child
    that is BETTER than its parent contributes nothing, so rescue is never suppressed and no
    separate rescue term is needed.  There is no margin hyper-parameter (m = 0).

    w is normalised to mean 1 over the edges WITHIN each class, so the risk decides which edge
    of a class matters, never how much that class is trained -- the class-mass property, moved
    to the place where it belongs.  The base cross-entropy stays exactly uniform, so no class
    re-weighting can enter through it at all.

    Returns (w [9 x K], s_bar), where s_bar = mean over (sample, edge) of g*w is the constant
    that turns the per-sample sum into the weighted mean above.
    """
    K = len(labels)
    counts = f0_prior_counts(cached, labels, states, "edge", risk_definition)
    G = len(DRO_EDGES) * K
    b, d = counts[:G], counts[G:]
    pooled = b.sum() / max(d.sum(), 1e-9)
    H = ((b + 20.0 * pooled) / (d + 20.0)).reshape(len(DRO_EDGES), K)
    z = np.clip((H - H.mean()) / (H.std() + 1e-8), -3.0, 3.0)
    w = np.exp(0.34657359 * z)
    w = w / w.mean(0, keepdims=True)          # mean 1 over edges within every class
    if np.abs(w.mean(0) - 1.0).max() > 1e-9:
        raise RuntimeError("edge weights are not class neutral")
    total = 0.0
    for record in cached.values():
        y = labels.index(record["gt"])
        pred = {st: int(np.argmax(record["states"][st])) for st in states}
        for u, (parent, _) in enumerate(DRO_EDGES):
            if pred[parent] == y:
                total += float(w[u, y])
    s_bar = total / max(len(cached) * len(DRO_EDGES), 1)
    if s_bar <= 0:
        raise RuntimeError("no protected parent evidence in the diagnostic cache")
    return w, s_bar


def margin_edge_setup(cached, labels, states, risk_definition="conditional", harm_only=False):
    """Edge weights, frozen parent margins and the gate share, for EMP (SSOT 10.18.122).

    The parent margin comes from the frozen diagnostic predictor's cached candidate scores, so
    no parent forward is needed: on this trainer the child's own decision-mode forward already
    runs for every state, and EMP only asks for its gradient.

    m = p(y) - max_{c != y} p(c) over the label-set softmax of the SUMMED answer-token scores,
    which is the same quantity the official decision rule ranks, so m > 0 holds exactly when
    that state's prediction is correct -- the gate condition itself.
    """
    K = len(labels)
    counts = f0_prior_counts(cached, labels, states, "edge", risk_definition)
    G = len(DRO_EDGES) * K
    b, d = counts[:G], counts[G:]
    pooled = b.sum() / max(d.sum(), 1e-9)
    H = ((b + 20.0 * pooled) / (d + 20.0)).reshape(len(DRO_EDGES), K)
    z = np.clip((H - H.mean()) / (H.std() + 1e-8), -3.0, 3.0)
    w = np.exp(0.34657359 * z)
    w = w / w.mean(0, keepdims=True)          # mean 1 over the edges within every class
    if np.abs(w.mean(0) - 1.0).max() > 1e-9:
        raise RuntimeError("edge weights are not class neutral")
    parents_of = {}
    for u, (parent, child) in enumerate(DRO_EDGES):
        parents_of.setdefault(child, []).append((u, parent))
    margins = {}
    for sample_id, record in cached.items():
        y = labels.index(record["gt"])
        per_state = {}
        for st in states:
            v = np.asarray(record["states"][st], dtype=np.float64)
            v = v - v.max()
            q = np.exp(v); q /= q.sum()
            t = q[y]; q[y] = -1.0
            per_state[st] = float(t - q.max())
        margins[sample_id] = per_state
    active = total = 0
    for record_id, per_state in margins.items():
        for st in states:
            total += 1
            if any(per_state[parent] > 0 and
                   (not harm_only or per_state[st] <= 0)
                   for _, parent in parents_of.get(st, [])):
                active += 1
    rho = active / max(total, 1)
    if rho <= 0:
        raise RuntimeError("no gated edges: the diagnostic predictor is never right at a parent")
    return {"w": w, "parents": parents_of, "margins": margins, "rho": rho}


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
                       frozen, torch, dist, selection_metric, route_states=False):
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
                if route_states:
                    # STATE_LORA: score each state through ITS OWN adapter.  Without this the
                    # validation pass inherits whatever the training loop set last -- AVT, the
                    # last entry of STATES -- so all seven states would be scored through
                    # [default, AVT], the selection score would be wrong, and the wrong
                    # checkpoint would be chosen, silently.  Caught 2026-09-19 before launch.
                    model.base_model.set_adapter(["default", state])
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
        # The uniform control uses no harm prior.  When the risk table is diagnosed by the
        # control itself, the control necessarily trains before any cache exists, so its
        # absence is expected for that arm only and is recorded rather than assumed.
        cache_path = screen / "cache_complete.json"
        # The VIS arms use no risk table either (uniform CE plus a vision-side objective), so on a
        # fold whose control diagnosed nothing they may also train without the cache.  The step-0
        # scorer gate is then recorded as SKIPPED_NO_CACHE; the adapter digest check above still runs,
        # and step-0 equivalence of the zero-initialised vision adapter was proven on fold 5
        # (2026-09-19 pre-flight, max|delta| = 0.00e+00).
        cache_missing = (args.arm in {"control", "VIS_CE", "VIS_CDSC"}
                         and not cache_path.exists())
        cache_complete = {} if cache_missing else json.loads(cache_path.read_text())
        # Declared separation of diagnosis from initialization (see the step-0 gate below).
        divergence_note = {k: cache_complete.get(k) for k in
                           ("diagnostic_adapter_sha256", "training_initialization_sha256",
                            "diagnostic_predictor")}
        declared = json.loads((screen / "preregistration.json").read_text()).get("diagnostic_predictor")
        declared_divergence = bool(
            divergence_note["diagnostic_adapter_sha256"]
            and divergence_note["diagnostic_adapter_sha256"] != divergence_note["training_initialization_sha256"]
            and declared not in (None, "F0"))
        # The risk definition is part of the frozen protocol of a run directory, so a run that
        # declares one does not depend on the launcher remembering to pass the flag.
        prereg_all = json.loads((screen / "preregistration.json").read_text())
        risk_definition = args.risk_definition or prereg_all.get("risk_definition", "joint")
        if risk_definition not in ("joint", "conditional"):
            raise RuntimeError(f"unknown risk definition {risk_definition!r}")
        if not cache_missing and frozen.sha(screen / "f0_sequences.jsonl") != cache_complete["sha256"]:
            raise RuntimeError("Frozen F0 cache digest mismatch")
        if args.arm not in {"control", "A", "B", "N2_control", "N2_true", "N2_cm", "A_N2", "N1",
                            "TDRO", "SDRO", "FULL", "ETR2", "EMP", "EMP2", "EMP3", "EMP4",
                            "CAP_SHARED", "STATE_LORA", "VIS_CE", "VIS_CDSC"}:
            raise RuntimeError("Unknown training arm")

        train_manifest = (args.train_manifest.resolve() if args.train_manifest
                          else frozen.training_manifest(screen))
        validation_manifest = (args.validation_manifest.resolve()
                               if args.validation_manifest else None)
        if bool(validation_manifest) != bool(args.validate_every):
            raise RuntimeError("validation-manifest and validate-every must be set together")
        if args.arm in {"VIS_CE", "VIS_CDSC"}:
            # Both vision arms share ONE dialogue-aware order, so VIS_CDSC - VIS_CE isolates the
            # contrastive term.  Exposure is exact (3739 of 3749 samples seen exactly twice, 10 once,
            # none zero) and the step count equals make_training_order's (468) -- verified
            # 2026-09-19 on fold 5 before launch.
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location(
                "cdsc_lib", str(Path(__file__).with_name("paper2_cdsc_lib_20260919.py")))
            cdsc_lib = _ilu.module_from_spec(_spec); _spec.loader.exec_module(cdsc_lib)
            order = cdsc_lib.make_dialogue_order(frozen.rows(train_manifest), args.epoch_budget,
                                                 args.seed, args.world_size)
        else:
            order = make_training_order(len(frozen.rows(train_manifest)), args.epoch_budget, args.seed)
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
                "rank_seeds": [args.seed + r for r in range(world)],
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
        # The T2 adapters are added AFTER the F0 integrity check, so that check still proves
        # the frozen F0 part is byte-exact; adapter_digest covers every trainable tensor, so
        # adding them first made it fail (2026-09-19 10:40, the gate working as designed).
        # The step-0 scorer gate below then re-verifies, on the REAL model, that the
        # zero-initialised lora_B leaves every score unchanged.
        # --- T2: state-conditional capacity -------------------------------------------------
        # CAP_SHARED and STATE_LORA carry the SAME extra parameters (seven rank-1 adapters on
        # the same target modules).  They differ only in routing: CAP_SHARED keeps all seven
        # active for every state, STATE_LORA activates only the adapter of the state being
        # forwarded.  PEFT initialises lora_B to zero, so at step 0 both arms are bit-identical
        # to Control -- verified 2026-09-19 00:40, max|delta| = 0.00e+00 -- which keeps the F0
        # integrity gate meaningful and makes any later difference attributable to how the
        # capacity is used rather than to how much of it there is.
        if args.arm in {"CAP_SHARED", "STATE_LORA"}:
            from peft import LoraConfig
            base_spec = runtime["training"]["lora"]
            for state_name in frozen.STATES:
                model.add_adapter(state_name, LoraConfig(
                    r=int(args.state_lora_rank),
                    lora_alpha=2 * int(args.state_lora_rank),
                    lora_dropout=float(base_spec["dropout"]),
                    bias="none",
                    target_modules=str(base_spec["target_regex"])))
            all_names = ["default"] + list(frozen.STATES)
            model.base_model.set_adapter(all_names)   # PeftModel.set_adapter rejects a list
            extra = sum(q.numel() for n, q in model.named_parameters()
                        if "lora" in n and ".default." not in n)
            shared = sum(q.numel() for n, q in model.named_parameters()
                         if "lora" in n and ".default." in n)
            if rank == 0:
                config = json.loads((destination / "config.json").read_text())
                config["state_capacity"] = {
                    "arm": args.arm,
                    "routing": ("all seven rank-1 adapters active for every state"
                                if args.arm == "CAP_SHARED" else
                                "only the adapter of the forwarded state is active"),
                    "state_lora_rank": int(args.state_lora_rank),
                    "shared_lora_parameters": shared,
                    "extra_parameters": extra,
                    "parameter_match": "CAP_SHARED and STATE_LORA carry identical counts",
                    "gradient_convention": (
                        "task_loss = CE / len(STATES) is kept, so the shared adapter sees the "
                        "mean over states while each state adapter sees one seventh of its own "
                        "state's gradient; no rescaling is applied because any rescaling factor "
                        "would be a new free parameter. Per-adapter gradient norms are logged."),
                }
                atomic_json(destination / "config.json", config)
                print(json.dumps({"state_capacity": config["state_capacity"]}), flush=True)

        vision_tower = None
        if args.arm in {"VIS_CE", "VIS_CDSC"}:
            # SSOT 10.18.137/138: the frozen visual encoder carries a strong dialogue-specific
            # shortcut (random-CV 66.9 vs dialogue-grouped 40.7, a 26.2-point gap; audio 2.7).
            # A rank-r LoRA on the vision tower's attention lets the encoder be adapted; lora_B
            # starts at zero so the model is bit-identical to Control at step 0.
            from peft import LoraConfig
            if args.arm == "VIS_CDSC" and not float(args.cd_beta) > 0:
                raise RuntimeError("VIS_CDSC needs --cd-beta > 0; with beta = 0 it is VIS_CE")
            if args.arm == "VIS_CE" and float(args.cd_beta) != 0:
                raise RuntimeError("VIS_CE must run with --cd-beta 0")
            base_spec = runtime["training"]["lora"]
            model.add_adapter("vision", LoraConfig(
                r=int(base_spec["rank"]), lora_alpha=int(base_spec["alpha"]),
                lora_dropout=float(base_spec["dropout"]), bias="none",
                target_modules=r"visual\.blocks\.\d+\.attn\.(q|k|v|proj)"))
            model.base_model.set_adapter(["default", "vision"])
            vis_params = sum(q.numel() for n, q in model.named_parameters()
                             if ".vision." in n and q.requires_grad)
            if vis_params == 0:
                raise RuntimeError("vision adapter created no trainable parameters")
            hits = [(n, m) for n, m in model.named_modules() if n.split(".")[-1] == "visual"]
            vision_tower = min(hits, key=lambda x: len(x[0]))[1]
            install_visual_hook(vision_tower)
            if rank == 0:
                config = json.loads((destination / "config.json").read_text())
                config["vision_adaptation"] = {
                    "arm": args.arm,
                    "vision_lora_target": "visual.blocks.N.attn.(q|k|v|proj)",
                    "vision_lora_rank": int(base_spec["rank"]),
                    "vision_trainable_parameters": vis_params,
                    "gradient_clipping": "language LoRA and vision LoRA clipped separately, each at "
                                         "1.0 (the language LoRA keeps Control's exact rule)",
                    "batching": "dialogue-aware: each rank's local batch is built from same-dialogue "
                                "different-emotion pairs (paper2_cdsc_lib_20260919.make_dialogue_order)",
                    "cd_supcon": (None if args.arm == "VIS_CE" else {
                        "positives": "same emotion, different dialogue",
                        "hard_negatives": "different emotion, same dialogue",
                        "excluded": "same emotion AND same dialogue (ambiguous)",
                        "representation": "mean-pooled vision-tower output, L2-normalised",
                        "anchor_with_gradient_bank_stop_gradient": True,
                        "tau": float(args.cd_tau), "beta": float(args.cd_beta)}),
                }
                atomic_json(destination / "config.json", config)
                print(json.dumps({"vision_adaptation": config["vision_adaptation"]}), flush=True)

        # Run the original step-0 scorer gate on rank 0.  Other replicas were
        # independently digest-checked above and wait at the barrier.
        gate = []
        if rank == 0 and cache_missing:
            # Nothing to compare against yet; the adapter digest check above is the guarantee.
            model.save_pretrained(destination / "initial_adapter", safe_serialization=True)
            atomic_json(destination / "step0_gate.json", {
                "status": "SKIPPED_NO_CACHE",
                "reason": "uniform control trains before the risk cache exists in this protocol",
                "initial_tensor_sha256": initial_digest,
                "replicas": world,
            })
        elif rank == 0:
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
                    if not declared_divergence and (
                            error > .02 or int(value.argmax()) != int(reference.argmax())):
                        raise RuntimeError((row["sample_id"], state, error))
                    del output, inputs, value
            # The step-0 gate exists to catch a cache that does not belong to the model being
            # trained.  When the risk table is deliberately diagnosed by a DIFFERENT model than
            # the one we initialize from, the two cannot agree, and the cache says so itself:
            # cache_complete.json carries both hashes and the preregistration names the
            # diagnostic predictor.  In that case the gate records the divergence instead of
            # failing; with no such declaration it stays strict.
            atomic_json(destination / "step0_gate.json", {
                "status": "RECORDED_DIVERGENCE" if declared_divergence else "PASS",
                "pairs": len(gate),
                "max_abs_score_error": max(gate),
                "initial_tensor_sha256": initial_digest,
                "diagnostic_adapter_sha256": divergence_note.get("diagnostic_adapter_sha256"),
                "diagnostic_predictor": divergence_note.get("diagnostic_predictor"),
                "replicas": world,
            })
            atomic_json(destination / "manifest.pretrain.json", {
                "initial_tensor_sha256": initial_digest,
                "frozen_files_sha256": frozen.sha(screen / "frozen_files.json"),
                "cache_sha256": None if cache_missing else frozen.sha(screen / "f0_sequences.jsonl"),
                "harm_statistics_sha256": None if cache_missing else frozen.sha(screen / "harm_statistics.json"),
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
        cached = {} if cache_missing else {
            row["sample_id"]: row
            for row in frozen.rows(screen / "f0_sequences.jsonl")
        }
        support = ({"rho": 0.0, "events": 0, "note": "no diagnostic cache for this arm"}
                   if cache_missing else n2_support(cached, frozen.LABELS, frozen.STATES))
        if (args.arm.startswith("N2_") or args.arm in {"A_N2", "N1"}) and support["rho"] <= 0:
            raise RuntimeError("N2 has no recoverable training events")
        if rank == 0:
            atomic_json(destination / "n2_support.json", support)
            config = json.loads((destination / "config.json").read_text())
            config["n2"] = {
                "mode": {"N2_true": "true", "N2_cm": "cm", "A_N2": "true", "N1": "hard"}.get(args.arm, "none"),
                "a_weights_on_task_ce": args.arm in {"A", "A_N2"},
                "risk_definition": risk_definition,
                "lambda": args.n2_lambda,
                "support": support,
                "target": "mean frozen-F0 exact-sequence posterior (T=1) over correct immediate parents",
                "confidence_matched": "preserve GT probability and uniformize non-target mass",
            }
            atomic_json(destination / "config.json", config)
        dist.barrier()
        # The uniform control applies no harm weights; in this protocol it also trains before
        # the table exists, so an all-ones table is used and recorded as such.
        harm = ({"weights": np.ones((len(frozen.STATES), len(frozen.LABELS))).tolist()}
                if cache_missing else json.loads((screen / "harm_statistics.json").read_text()))
        weights = np.asarray(harm["weights"])
        emp = None
        if args.arm in {"EMP", "EMP2", "EMP3", "EMP4"}:
            if cache_missing:
                raise RuntimeError("EMP needs the frozen diagnostic cache for the parent margins")
            if frozen.exact_needed(label_ids):
                # The child margin comes from frozen.scores, which reads every candidate at the
                # anchor's positions.  That is exact only while at most one label is multi-token
                # (IEMOCAP-4).  On a label set like IEMOCAP-6 the margin this arm optimises would
                # not be the margin the official rule ranks, which is the defect corrected on the
                # MELD side; refuse rather than train against the wrong quantity.
                raise RuntimeError(
                    "EMP on this label set needs the exact candidate scorer: token counts "
                    f"{[len(x) for x in label_ids]} make the anchor-forced read inexact")
            emp = margin_edge_setup(cached, frozen.LABELS, frozen.STATES, risk_definition,
                                    harm_only=(args.arm in {"EMP2", "EMP4"}))
            if args.uniform_edge_weights:
                # Ablation: the class-neutral edge weights already have mean 1 over the edges
                # within each class, so replacing them by ones changes the relative emphasis
                # across the nine transitions while leaving the term's overall scale alone.
                emp["w"] = np.ones_like(emp["w"])
            if rank == 0:
                config = json.loads((destination / "config.json").read_text())
                config["edge_margin"] = {
                    "objective": ({
                        "EMP": "L_uniform + mean over gated (i,S) of the weighted mean over "
                               "incoming edges of [m_parent^{F0} - m_child^{theta}]_+",
                        "EMP2": "L_uniform + mean over frozen harmful (i,S) of the weighted "
                                "mean over incoming edges of [-m_child^{theta}]_+",
                        "EMP3": "L_uniform + mean over gated (i,S) of the weighted mean over "
                                "incoming edges of [sg(m_parent^{theta}) - m_child^{theta}]_+; "
                                "the parent margin is read from the current model's own "
                                "decision-mode forward for that parent state, which this "
                                "trainer already runs in the same microstep",
                        "EMP4": "L_uniform + mean over frozen harmful (i,S) of the weighted "
                                "mean over incoming edges of [-m_child^{theta}]_+; the auxiliary "
                                "term is averaged over all seven states and is not divided by "
                                "the sparse harmful-gate share",
                    }[args.arm]),
                    "margin": "p(y) - max_{c != y} p(c) over the label-set softmax",
                    "gate": ("the frozen diagnostic predictor has m_P > 0 and m_S <= 0"
                             if args.arm in {"EMP2", "EMP4"} else
                             "the frozen diagnostic predictor is correct at the parent, i.e. m_P > 0"),
                    "parent_margin_source": ("current model (stop-gradient)"
                                             if args.arm == "EMP3" else "frozen F0 cache"),
                    "delta": 0.0, "lambda_emp": 1.0, "gated_state_share": emp["rho"],
                    "state_normalisation": (
                        "divided by len(STATES) so the auxiliary term is a mean over all (i,S) "
                        "pairs, matching the task term's mean over states; runs before "
                        "2026-09-18 11:20 Asia/Shanghai omitted this and therefore trained at an "
                        "effective lambda of ~7 on this trainer (MELD's trainer was already at 1)"),
                    "lambda_effective_vs_mean_state_ce": 1.0,
                    "uniform_edge_weights_ablation": bool(args.uniform_edge_weights),
                    "edge_weight_class_means": emp["w"].mean(0).round(9).tolist(),
                    "edge_weight_range": [float(emp["w"].min()), float(emp["w"].max())],
                    "base_cross_entropy": "uniform over the seven states; no state or class weights"}
                atomic_json(destination / "config.json", config)
        relational = None
        if args.arm == "ETR2":
            if cache_missing:
                raise RuntimeError("ETR2 needs the frozen diagnostic cache to gate its edges")
            # The base objective stays uniform for this arm: the relation is supervised
            # directly, so nothing needs to move the per-state or per-class loss mass.
            edge_w, edge_sbar = relational_edge_setup(cached, frozen.LABELS, frozen.STATES,
                                                      risk_definition)
            parents_of = {}
            for u, (parent, child) in enumerate(DRO_EDGES):
                parents_of.setdefault(child, []).append((u, parent))
            relational = {"w": edge_w, "s_bar": edge_sbar, "parents": parents_of}
            if rank == 0:
                config = json.loads((destination / "config.json").read_text())
                config["relational_edge"] = {
                    "objective": "L_uniform + mean_{i,e} g*w*[l_S - sg(l_P)]_+ / mean_{i,e} g*w",
                    "gate": "g = 1 iff the frozen diagnostic predictor is correct at the parent",
                    "margin": 0.0, "lambda_edge": 1.0,
                    "risk_definition": risk_definition,
                    "edge_weight_class_means": edge_w.mean(0).round(9).tolist(),
                    "edge_weight_range": [float(edge_w.min()), float(edge_w.max())],
                    "s_bar": float(edge_sbar),
                    "base_cross_entropy": "uniform over the seven states; no state or class weights"}
                atomic_json(destination / "config.json", config)
        dro = None
        if args.arm in {"TDRO", "SDRO", "FULL"}:
            label_counts = np.array([sum(r["gt"] == c for r in training_rows) for c in frozen.LABELS], float)
            dro = DroState(args.arm, frozen.STATES, frozen.LABELS, label_counts / label_counts.sum(), args,
                           prior_counts=f0_prior_counts(cached, frozen.LABELS, frozen.STATES,
                                                       args.prior_granularity,
                                                       risk_definition))
            if rank == 0:
                config = json.loads((destination / "config.json").read_text())
                config["dro"] = {"arm": args.arm, "weight_normalization": args.weight_normalization,
                                 "class_mass_per_class": dro.w.mean(0).round(6).tolist(),
                                 "weights_at_clip": dro.clipped,
                                 "weight_range": [float(dro.w.min()), float(dro.w.max())],
                                 "warmup_steps": args.dro_warmup, "count_decay_beta": args.dro_beta,
                                 "shrinkage_alpha": args.dro_alpha, "kappa": args.dro_kappa, "rho": args.dro_rho,
                                 "q_min": args.dro_qmin, "q_max": args.dro_qmax,
                                 "w_min": args.dro_wmin, "w_max": args.dro_wmax, "seed": args.seed,
                                 "prior": "F0 edge x class harm (FULL)" if args.arm == "FULL" else "uniform",
                                 "initial_w": dro.w.round(5).tolist(),
                                 "groups": dro.G, "class_freq": (label_counts / label_counts.sum()).tolist(),
                                 "prior_share": dro.prior_share,
                                 "rule": "log q <- (1-rho) log q + rho (s log pi + (1-s) kappa z); child w = mean incoming-edge q; singletons 1"}
                atomic_json(destination / "config.json", config)
        video_pixels = int(runtime["training"]["video_total_pixels"])
        if rank == 0:
            atomic_json(destination / "unit_order.json", order)

        # Rank-specific streams are the standard distributed-training choice;
        # record them rather than pretending bit identity with single GPU.
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
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
                audit_task = audit_n2 = None
                dro_local = np.zeros(2 * dro.G) if dro is not None else None
                decoded_cache = {}
                cd_bank = None
                if args.arm == "VIS_CDSC" and args.cd_beta > 0:
                    bank_z, bank_y, bank_d = [], [], []
                    for index in local_indices:
                        brow = training_rows[index]
                        decoded_cache[index] = frozen.decoded_row(brow, video_pixels)
                        bank_z.append(capture_visual(model, vision_tower, frozen, processor, brow,
                                                     video_pixels, decoded_cache[index],
                                                     torch, grad=False))
                        bank_y.append(frozen.LABELS.index(brow["gt"]))
                        bank_d.append(cdsc_lib.dialogue_of(brow))
                    dia_ids = {d: k for k, d in enumerate(sorted(set(bank_d)))}
                    cd_bank = (torch.stack(bank_z), torch.tensor(bank_y, device="cuda"),
                               torch.tensor([dia_ids[d] for d in bank_d], device="cuda"), dia_ids)
                local_cd = 0.0; local_cd_n = 0
                for position, index in enumerate(local_indices):
                    row = training_rows[index]
                    target = frozen.LABELS.index(row["gt"])
                    decoded = (decoded_cache[index] if index in decoded_cache
                               else frozen.decoded_row(row, video_pixels))
                    # Frozen diagnostic posteriors.  They are read only by the B hinge and by
                    # the N2 family; the uniform control reads neither, and in this protocol it
                    # trains before any diagnostic cache exists.
                    if cached:
                        references = {
                            state: torch.tensor(
                                cached[row["sample_id"]]["states"][state], device="cuda"
                            )
                            for state in frozen.STATES
                        }
                        references = {
                            state: value - value.mean() for state, value in references.items()
                        }
                    else:
                        references = {}
                    current = {}
                    state_ce = {}
                    active = ([
                        state for state in frozen.EDGES
                        if int(references[state].argmax()) == target
                        and int(references["AVT"].argmax()) != target
                    ] if args.arm == "B" else [])
                    for state_index, state in enumerate(frozen.STATES):
                        VISUAL_GRAB["z"] = None
                        n2_mode = ("true" if args.arm in {"N2_true", "A_N2"} else
                                   ("cm" if args.arm == "N2_cm" else
                                    ("hard" if args.arm == "N1" else None)))
                        n2_target = (n2_recoverable_target(
                            references, target, state, n2_mode, torch
                        ) if n2_mode else None)
                        requires_harm_grad = state == "AVT" and bool(active)
                        emp_parents = []
                        if emp is not None:
                            per_state = emp["margins"].get(row["sample_id"], {})
                            emp_parents = [(u, p) for u, p in emp["parents"].get(state, [])
                                           if per_state.get(p, -1.0) > 0.0 and
                                           (args.arm not in {"EMP2", "EMP4"}
                                            or per_state.get(state, 0.0) <= 0.0)]
                        if args.arm == "STATE_LORA":
                            model.base_model.set_adapter(["default", state])
                        requires_decision_grad = (requires_harm_grad or n2_target is not None
                                                  or bool(emp_parents))
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
                            if emp_parents:
                                def dynamic_parent_margin(state_name, _c=current, _t=target):
                                    q = torch.softmax(_c[state_name], dim=-1)
                                    rival = torch.cat([q[:_t], q[_t + 1:]]).max()
                                    return float(q[_t] - rival)

                                # Softmax is shift invariant, so the centring above is harmless.
                                probabilities = torch.softmax(score, dim=-1)
                                competitor = torch.cat([probabilities[:target],
                                                        probabilities[target + 1:]]).max()
                                child_margin = probabilities[target] - competitor
                                numerator = child_margin * 0.0
                                denominator = 0.0
                                for unit, parent in emp_parents:
                                    coefficient = float(emp["w"][unit, target])
                                    if args.arm == "EMP":
                                        term = torch.relu(float(per_state[parent]) - child_margin)
                                    elif args.arm in {"EMP2", "EMP4"}:
                                        # EMP2 repairs only frozen harmful transitions.  The
                                        # child target is the decision boundary (margin zero),
                                        # so correct-but-less-confident children are untouched.
                                        term = torch.relu(-child_margin)
                                    else:
                                        # EMP3 reads the parent margin off the CURRENT model.
                                        # STATES is ordered so that every parent of S is
                                        # forwarded earlier in this same microstep, so the
                                        # detached score is already in `current` and no extra
                                        # forward is needed.  Stop-gradient on the parent keeps
                                        # the barrier one-sided: the child is pushed up, the
                                        # parent is never pulled down.
                                        if parent not in current:
                                            raise RuntimeError(
                                                f"EMP3: parent {parent} of {state} was not "
                                                "forwarded before the child; STATES ordering "
                                                "violates the topological assumption")
                                        term = torch.relu(
                                            dynamic_parent_margin(parent) - child_margin)
                                    numerator = numerator + coefficient * term
                                    denominator += coefficient
                                # /len(STATES) makes this a mean over ALL (i,S) pairs, so it
                                # sits on the same scale as the task term, which is likewise a
                                # mean over states (task_loss = output.loss / 7, backwarded once
                                # per state).  Without it the auxiliary term backwards once per
                                # GATED state with no state averaging at all, giving an effective
                                # lambda of ~7 rather than the declared 1 -- and MELD's trainer,
                                # where loss = output.loss + term per (i,S) microstep, was already
                                # running the declared lambda = 1.  That mismatch, not the corpus,
                                # is why the same nominal objective helped on MELD and hurt here.
                                # EMP4 deliberately keeps the sparse harmful gate's natural
                                # frequency.  Dividing by rho would turn a 4% event subset into
                                # a 25x loss multiplier and recreate the over-regularisation that
                                # motivated this arm.
                                if args.arm == "EMP4":
                                    # NOT numerator/denominator.  EMP4's term [-m_child]_+ does
                                    # not depend on the parent, so a weighted MEAN over incoming
                                    # edges divides omega out exactly -- measured at 0.000e+00,
                                    # the same defect that sank EMP2 (SSOT 10.18.127).  The
                                    # proposal's own formula is a SUM over the harmful edge set
                                    # H, which keeps omega alive and lets a child gated by three
                                    # harmful parents weigh more than one gated by a single
                                    # parent.  denominator stays computed for the audit record.
                                    emp_loss = numerator / len(frozen.STATES)
                                else:
                                    emp_loss = (numerator / denominator / emp["rho"]
                                                / len(frozen.STATES))
                                if not torch.isfinite(emp_loss):
                                    raise RuntimeError("Non-finite EMP loss")
                                (emp_loss / GLOBAL_BATCH).backward()
                                local_harm += float(emp_loss.detach())
                                del emp_loss, numerator, probabilities, child_margin
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
                                before = lora_grad_vector(model, torch) if args.grad_audit else None
                                (scaled_n2 / GLOBAL_BATCH).backward()
                                if args.grad_audit:
                                    delta = lora_grad_vector(model, torch) - before
                                    audit_n2 = delta if audit_n2 is None else audit_n2 + delta
                                    del before, delta
                                local_n2 += float(n2_loss.detach())
                                local_n2_events += 1
                                del n2_loss, scaled_n2
                        del output, inputs, score

                        output, inputs = frozen.forward(
                            model, processor, row, state, video_pixels, decoded, False
                        )
                        task_loss = output.loss / 7
                        if cd_bank is not None and state == "V":
                            z_i = VISUAL_GRAB.get("z")
                            if z_i is None:
                                raise RuntimeError("vision tower output was not captured on state V")
                            bz, by, bd, dmap = cd_bank
                            keep = torch.arange(len(by), device="cuda") != position
                            cd, n_pos, n_hard = cdsc_lib.cd_supcon_anchor(
                                z_i, target, dmap[cdsc_lib.dialogue_of(row)],
                                bz[keep], by[keep], bd[keep], float(args.cd_tau), torch)
                            if cd is not None:
                                if not torch.isfinite(cd):
                                    raise RuntimeError("Non-finite CD-SupCon loss")
                                task_loss = task_loss + float(args.cd_beta) * cd
                                local_cd += float(cd.detach()); local_cd_n += 1
                                # keep the logged "task" CE-only, comparable to Control's
                                local_task -= float(args.cd_beta) * float(cd.detach())
                        if relational is not None:
                            # Parents precede children in STATES, so every parent loss this edge
                            # needs is already cached as a detached scalar: the hinge costs no
                            # extra forward pass.
                            terms = []
                            for unit, parent in relational["parents"].get(state, []):
                                if cached_parent_correct(cached, row, parent, target, frozen):
                                    coefficient = (float(relational["w"][unit, target])
                                                   / (len(DRO_EDGES) * relational["s_bar"]))
                                    terms.append(coefficient * torch.relu(
                                        output.loss - state_ce[parent]))
                            if terms:
                                edge_loss = torch.stack(terms).sum()
                                if not torch.isfinite(edge_loss):
                                    raise RuntimeError("Non-finite relational edge loss")
                                task_loss = task_loss + edge_loss
                                local_harm += float(edge_loss.detach())
                        state_ce[state] = output.loss.detach()
                        if args.arm in {"A", "A_N2"}:
                            task_loss = task_loss * float(weights[state_index, target])
                        if dro is not None:
                            task_loss = task_loss * float(dro.w[state_index, target])
                        if not torch.isfinite(task_loss):
                            raise RuntimeError("Non-finite task loss")
                        before = lora_grad_vector(model, torch) if args.grad_audit else None
                        (task_loss / GLOBAL_BATCH).backward()
                        if args.grad_audit:
                            delta = lora_grad_vector(model, torch) - before
                            audit_task = delta if audit_task is None else audit_task + delta
                            del before, delta
                        local_task += float(task_loss.detach())
                        del output, inputs, task_loss
                    if dro is not None:
                        dro_local += dro.local_counts(
                            {name: int(value.argmax()) for name, value in current.items()}, target)

                if args.grad_audit:
                    task_norm = float(audit_task.norm()) if audit_task is not None else 0.0
                    n2_norm = float(audit_n2.norm()) if audit_n2 is not None else 0.0
                    with (destination / f"grad_audit_rank{rank}.jsonl").open("a") as stream:
                        stream.write(json.dumps({
                            "step": step_zero + 1, "rank": rank,
                            "task_grad_norm": task_norm, "n2_grad_norm": n2_norm,
                            "n2_over_task": n2_norm / task_norm if task_norm else None,
                            "local_n2_events": local_n2_events,
                        }) + "\n")
                    del audit_task, audit_n2
                reduce_gradients(model, torch, dist)
                if vision_tower is not None:
                    # The vision LoRA's gradient is several times the language LoRA's (pre-flight
                    # 2026-09-19: 2.4 from one state-V CE vs a Control step norm of ~0.6), so one
                    # joint clip at 1.0 would shrink the language LoRA's updates below Control's
                    # and confound the comparison.  Each group is clipped at the SAME 1.0
                    # separately, so the language LoRA sees exactly Control's clipping rule.
                    language_params = [q for n, q in model.named_parameters()
                                       if q.requires_grad and ".vision." not in n]
                    vision_params = [q for n, q in model.named_parameters()
                                     if q.requires_grad and ".vision." in n]
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        language_params, 1.0, error_if_nonfinite=True)
                    vision_grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        vision_params, 1.0, error_if_nonfinite=True))
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    vision_grad_norm = None
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                totals = torch.tensor(
                    [local_task, local_harm, local_n2, local_n2_events, local_cd, local_cd_n],
                    device="cuda", dtype=torch.float64,
                )
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                step = step_zero + 1
                completed_steps = step
                if dro is not None:
                    dro_counts = torch.tensor(dro_local, device="cuda", dtype=torch.float64)
                    dist.all_reduce(dro_counts, op=dist.ReduceOp.SUM)
                    dro.update(dro_counts.cpu().numpy(), step)
                    with (destination / f"dro_trajectory_rank{rank}.jsonl").open("a") as stream:
                        stream.write(json.dumps(dro.snapshot(step)) + "\n")
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
                        "cd": float(totals[4]) / max(1.0, float(totals[5])),
                        "cd_anchors": int(totals[5]),
                        "grad_norm": float(grad_norm),
                        "vision_grad_norm": vision_grad_norm,
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
                            **({"dro/w_min": float(dro.w.min()), "dro/w_max": float(dro.w.max()),
                                "dro/active": int(dro.active), "dro/risk_mean": float(dro.H.mean())}
                               if dro is not None else {}),
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
                        route_states=(args.arm == "STATE_LORA"),
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
    parser.add_argument("--arm", default="B")   # ETR2 = relational edge objective
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
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dro-warmup", type=int, default=50)
    parser.add_argument("--dro-rho", type=float, default=0.05)
    parser.add_argument("--dro-qmin", type=float, default=0.5)
    parser.add_argument("--dro-qmax", type=float, default=2.0)
    parser.add_argument("--dro-beta", type=float, default=0.95)
    parser.add_argument("--dro-alpha", type=float, default=20.0)
    parser.add_argument("--dro-kappa", type=float, default=0.34657359)
    parser.add_argument("--risk-definition", choices=["joint", "conditional"], default=None,
                        help="joint = P(parent correct and child wrong | c); conditional = "
                             "P(child wrong | parent correct, c), which removes the class-accuracy "
                             "confound. Unset: read 'risk_definition' from the run directory's "
                             "preregistration.json, falling back to 'joint'.")
    parser.add_argument("--prior-granularity", choices=["edge", "state", "class"], default="edge",
                        help="group layout of the harm prior; 'state' and 'class' are the "
                             "scale-matched controls that keep the identical mapping")
    parser.add_argument("--weight-normalization", choices=["global", "class"], default="global",
                        help="global: one class-frequency-weighted scalar over the whole weight "
                             "table (the original map; the per-class mean weight is free to "
                             "drift, which re-weights classes as a side effect).  class: "
                             "class-mass preserving, so the mean weight over the seven states is "
                             "1 for every class and only the within-class allocation moves.")
    parser.add_argument("--dro-wmin", type=float, default=0.4)
    parser.add_argument("--dro-wmax", type=float, default=2.5)
    parser.add_argument("--state-lora-rank", type=int, default=1,
                        help="T2: rank of each of the seven per-state adapters. The parameter "
                             "budget is identical for CAP_SHARED and STATE_LORA, so this only "
                             "sets how much capacity is being routed, never which arm wins.")
    parser.add_argument("--cd-beta", type=float, default=0.0,
                        help="VIS_CDSC: weight of the cross-dialogue supervised contrastive term. "
                             "Fixed from a step-0 measurement and pre-registered, never searched.")
    parser.add_argument("--cd-tau", type=float, default=0.1,
                        help="VIS_CDSC: SupCon temperature (the standard 0.1 of Khosla et al.).")
    parser.add_argument("--uniform-edge-weights", action="store_true",
                        help="EMP ablation: set every edge weight to 1, so the auxiliary term "
                             "keeps its scale but loses the risk-derived emphasis across the "
                             "nine transitions. Answers whether edge-awareness contributes, "
                             "which the omega-is-live unit test alone cannot.")
    parser.add_argument("--grad-audit", action="store_true",
                        help="Record separate local task/N2 LoRA gradient norms (smoke only)")
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
