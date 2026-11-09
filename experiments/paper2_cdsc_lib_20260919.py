"""Pure helpers for the CD-SupCon arm (SSOT 10.18.138): dialogue-aware order and the loss.

Kept free of model code so both can be unit-tested before any GPU run.
"""
from __future__ import annotations
import collections
import numpy as np

GLOBAL_BATCH = 16


def dialogue_of(row):
    return row.get("dialogue_name") or row.get("dialogue_id")


def make_dialogue_order(rows, epoch_budget, seed, world, global_batch=GLOBAL_BATCH):
    """Order in which every rank's local slice is built from SAME-DIALOGUE, DIFFERENT-EMOTION pairs.

    Exposure is exact: every utterance appears exactly once per epoch, as in make_training_order,
    and the total is truncated to a multiple of global_batch.  Within an epoch, utterances of each
    dialogue are paired greedily across different emotions; leftovers that cannot be paired
    differently are paired with each other.  Pairs are packed into local batches of
    global_batch // world, and `world` local batches are interleaved so that
    global[rank::world] reproduces local batch `rank` exactly.  A same-dialogue pair therefore
    never gets split across ranks.
    """
    assert global_batch % world == 0 and (global_batch // world) % 2 == 0
    local = global_batch // world
    rng = np.random.default_rng(seed)
    by_dia = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_dia[dialogue_of(r)].append(i)
    order, carry = [], []
    for _ in range(max(1, epoch_budget)):
        pairs, singles = [], []
        for d in rng.permutation(sorted(by_dia)):
            by_emo = collections.defaultdict(list)
            for i in by_dia[d]:
                by_emo[rows[i]["gt"]].append(i)
            pools = [list(rng.permutation(v)) for v in by_emo.values()]
            while sum(len(p) for p in pools) >= 2:
                pools.sort(key=len, reverse=True)
                if len(pools) >= 2 and pools[1]:
                    pairs.append((pools[0].pop(), pools[1].pop()))      # different emotions
                else:
                    pairs.append((pools[0].pop(), pools[0].pop()))      # only one emotion left
                pools = [p for p in pools if p]
            for p in pools:                                             # an odd leftover
                singles.extend(p)
        # Singletons from different dialogues are paired with each other, so EVERY unit has
        # size two and a local batch of size `local` (even) holds exactly local/2 units.  No
        # padding and no duplication: exposure stays exact except for the final partial block.
        # (The first version padded short batches at random: exposure came out min 0 / max 4.)
        singles = list(rng.permutation(singles))
        while len(singles) >= 2:
            pairs.append((singles.pop(), singles.pop()))
        carry.extend(singles); singles = []
        rng.shuffle(pairs)
        units = [u for pr in pairs for u in [pr]]
        per_local = local // 2
        locals_ = [sum((list(u) for u in units[k:k + per_local]), [])
                   for k in range(0, len(units) - per_local + 1, per_local)]
        for g in range(0, len(locals_) - world + 1, world):
            block = [None] * global_batch
            for r in range(world):
                for k, idx in enumerate(locals_[g + r]):
                    block[k * world + r] = idx
            order.extend(block)
    n = len(rows) * max(1, epoch_budget)
    n = n // global_batch * global_batch
    return [int(x) for x in order[:n]]   # plain ints: the order is written to JSON


def cd_supcon_anchor(z_i, y_i, d_i, bank_z, bank_y, bank_d, tau, torch):
    """Anchor-with-gradient supervised contrastive loss, others stop-gradient.

    positives : same emotion, DIFFERENT dialogue
    negatives : different emotion (the same-dialogue ones are the hard negatives -- they sit
                closest under the measured visual shortcut, so they dominate the denominator)
    excluded  : same emotion AND same dialogue -- ambiguous between shortcut and genuine emotion
    Returns (loss, n_pos, n_hard_neg); loss is None when the anchor has no positive.
    """
    zi = torch.nn.functional.normalize(z_i.float(), dim=-1)
    zb = torch.nn.functional.normalize(bank_z.float(), dim=-1)
    sim = (zb @ zi) / tau
    same_y = bank_y == y_i; same_d = bank_d == d_i
    pos = same_y & ~same_d
    keep = ~(same_y & same_d)                         # drop the ambiguous same/same pairs
    n_pos = int(pos.sum()); n_hard = int((~same_y & same_d).sum())
    if n_pos == 0:
        return None, 0, n_hard
    log_den = torch.logsumexp(sim[keep], dim=0)
    loss = -(sim[pos] - log_den).mean()
    return loss, n_pos, n_hard
