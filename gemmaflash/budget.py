"""Per-layer expert budgets: how many of each MoE layer's 128 experts stay in memory.

Layers differ in how many distinct experts decode touches, so an even split
wastes memory. fit(mean) spreads mean * 30 resident experts over the 30 MoE
layers greedily: every layer starts at 32, and two slots at a time go to the
layer whose LRU misses drop the most. The miss counts per layer and capacity
(data/miss_table.json) were simulated on a routing trace of 12 held-out
prompts, none of them in the bench set.
"""

import json
from pathlib import Path

NUM_EXPERTS = 128
MIN_PER_LAYER = 32  # one verify forward (4 tokens x top-8) can route 32 distinct experts
RECORD_BYTES = 3358720  # one expert's gate/up/down weights, scales and biases (padded)
_TABLE = json.loads((Path(__file__).parent / "data/miss_table.json").read_text())
NUM_LAYERS = len(_TABLE["misses"])  # MoE layers


def fit(mean):
    """Per-layer resident experts with the given mean (32..128)."""
    caps, misses = _TABLE["capacities"], _TABLE["misses"]
    n = len(misses)
    if not caps[0] <= mean <= caps[-1]:
        raise ValueError(f"experts mean must be between {caps[0]} and {caps[-1]}, got {mean}")
    total = round(mean * n)
    step = caps[1] - caps[0]
    at = [0] * n
    while sum(caps[i] for i in at) + step <= total:
        gain = [misses[l][at[l]] - misses[l][at[l] + 1] if at[l] + 1 < len(caps) else -1 for l in range(n)]
        at[max(range(n), key=gain.__getitem__)] += 1
    return [caps[i] for i in at]


def expand(budget):
    """One int for every layer, or a per-layer list, as a per-layer list."""
    b = [int(budget)] * NUM_LAYERS if isinstance(budget, (int, float)) else [int(x) for x in budget]
    if len(b) != NUM_LAYERS or not all(MIN_PER_LAYER <= x <= NUM_EXPERTS for x in b):
        raise ValueError(f"a budget is {NUM_LAYERS} counts between {MIN_PER_LAYER} and {NUM_EXPERTS}")
    return b


def describe(budget):
    """Resident expert memory and the held-out trace's decode hit rate for a budget."""
    budget = expand(budget)
    caps, misses = _TABLE["capacities"], _TABLE["misses"]
    if any(c not in caps for c in budget):
        hit = None  # not on the simulated grid
    else:
        m = sum(misses[l][caps.index(c)] for l, c in enumerate(budget))
        hit = 1 - m / sum(_TABLE["lookups"])
    return {
        "mean": sum(budget) / len(budget),
        "resident_gb": sum(budget) * RECORD_BYTES / 2**30,
        "heldout_hit_rate": hit,
    }
