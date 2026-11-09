#!/usr/bin/env python3
"""Narrow the classifier's label set for a second corpus.

Every Qwen-side module carries its own module-level LABELS built for MELD's
seven emotions.  IEMOCAP-4 needs four.  Rather than fork the training and
evaluation paths, this rebinds the label list (and the task prompt derived from
it) in each module at runtime.  Callers that never pass --labels keep MELD's
seven-way behaviour byte-for-byte.
"""
import importlib

MODULES = (
    "run_meld_qwen_lora_sft",
    "run_planb_meld_qwen_pilot",
    "run_meld_qwen_sft_selection_eval",
    "run_meld_c1_oof",
)


def apply_label_set(labels):
    labels = list(labels)
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate labels: {labels}")
    touched = []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(mod, "set_label_set"):
            mod.set_label_set(labels)
            touched.append(name + ".set_label_set")
        if hasattr(mod, "LABELS"):
            mod.LABELS = type(mod.LABELS)(labels)
            touched.append(name + ".LABELS")
    return labels, touched


def check_first_token_uniqueness(processor, labels):
    """The differentiable surrogate indexes one logit per label, so the first
    token of each label must be distinct."""
    tok = processor.tokenizer
    first = {l: tok(l, add_special_tokens=False).input_ids[0] for l in labels}
    if len(set(first.values())) != len(labels):
        raise RuntimeError(f"label first tokens collide: {first}")
    return first
