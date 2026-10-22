"""Routed experts streamed from SSD into a small resident cache, gated in-stream.

Each MoE layer keeps C of its 128 experts resident in slot tensors shaped like
the stock expert weights ([C, ...] instead of [128, ...]). The rest live in a
per-layer store on disk (store.py: the checkpoint's bytes, one record per
expert).

Decode: the router's indices go to the native gate (native/xgate.cpp). The GPU
spins on a coherent flag while a C++ worker does the LRU lookup, reads misses
straight into the slot tensors (F_NOCACHE) and writes slot indices; Python
never syncs. The worker then prefetches the next layer's predicted experts.
The expert kernels read the same weight rows, so outputs are bit-identical to
the fully resident model.

Prefill (more than DECODE_BLOCK_SIZE tokens) touches most experts: routed
(token, expert) rows of resident experts run straight from the slot tensors,
rows of missing experts from a small buffer (shared by all layers) the misses
are read into. Each output row of the sorted gather kernels depends only on
its own input row and expert weights, so splitting the rows into two calls and
scattering them back is bit-identical to the stock call. The most recently
used experts are then copied into the slots.
"""

import fcntl
import json
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

DTYPES = {"U32": mx.uint32, "BF16": mx.bfloat16}
TOKEN_CHUNK = 256  # prefill tokens per expert computation (bounds its temporaries)
PREFILL_MIN = 9  # mlx-vlm DECODE_BLOCK_SIZE + 1: longer inputs take the sorted path

XG = [None]  # the native extension
STATS = dict(prefill_reads=0, read_s=0.0, sync_s=0.0)
_local = threading.local()
_RECORD = [0]
_MISSBUF = []
_MISS_ROWS = [0]  # rows for the layer with the most non-resident experts


def _miss_buffer(n, like):
    """Rows for the experts a prefill reads in, shared by all layers."""
    if not _MISSBUF:
        _MISSBUF.extend(mx.zeros((n, *a.shape[1:]), a.dtype) for a in like)
        mx.eval(_MISSBUF)
    return _MISSBUF


def _stage():
    """Per-thread buffer for one expert record."""
    s = getattr(_local, "stage", None)
    if s is None:
        s = _local.stage = np.empty(_RECORD[0], np.uint8)
    return s


def _bytes(a):
    """Writable byte view of an evaluated MLX array (same memory the GPU reads)."""
    mx.eval(a)
    return np.frombuffer(memoryview(a), dtype=np.uint8).reshape(a.shape[0], -1)


class LayerCache:
    def __init__(self, layer, idx, store, layout, capacity, pool):
        self.layer, self.capacity, self.pool = layer, capacity, pool
        self.record = layout["record_bytes"]
        self.fields = layout["fields"]
        self.fields_key = [(f["proj"], f["part"]) for f in self.fields]
        self.fd = os.open(store / f"L{idx:02d}.bin", os.O_RDONLY)
        fcntl.fcntl(self.fd, fcntl.F_NOCACHE, 1)
        self.num_experts = layout["num_experts"]
        self.sg = layer.experts.switch_glu
        for f in self.fields:
            lin = getattr(self.sg, f["proj"])
            setattr(lin, f["part"], mx.zeros((capacity, *f["shape"]), DTYPES[f["dtype"]]))
        # The extension reads the first experts and owns slot_of / LRU.
        self.xid = XG[0].add_layer(
            str(store / f"L{idx:02d}.bin"), self.record, self.num_experts, capacity,
            [f["offset"] for f in self.fields], [f["nbytes"] for f in self.fields], self._arrays(),
        )

    def _arrays(self):
        return [getattr(getattr(self.sg, f["proj"]), f["part"]) for f in self.fields]

    def _read(self, jobs, arrays):
        """jobs: (expert, row) pairs; reads each expert's record into row of arrays."""
        if not jobs:
            return
        t = time.perf_counter()
        views = [_bytes(a) for a in arrays]

        def one(job):
            self._read_one(views, *job)

        if len(jobs) == 1:
            one(jobs[0])
        else:
            list(self.pool.map(one, jobs))
        STATS["read_s"] += time.perf_counter() - t

    def _read_one(self, views, e, row):
        # One aligned read of the whole record, then copy the fields out:
        # uncached reads into the fields' unaligned rows are ~3x slower.
        stage = _stage()
        n = os.preadv(self.fd, [stage], e * self.record)
        assert n == self.record, (self.layer, e, n)
        for f, v in zip(self.fields, views):
            v[row] = stage[f["offset"]:f["offset"] + f["nbytes"]]

    def _compute(self, idx_np, x, w, fast, slots_now, buf, bufpos, slot_of):
        """Routed experts for idx_np's tokens: resident rows from the slots,
        missing ones from buf (rows bufpos), in the given kernel regime."""
        # Same as SwitchGLU's sorted path, with the routed rows split by
        # where their expert lives; each part sorted by its own row index.
        # The sorted gather kernel changes (and so does its rounding) at
        # rows >= max(4 * experts in the weight tensor, 16); the stock call
        # has len(flat) rows over num_experts, and every part is kept on
        # the same side of that line.
        sg = self.sg
        flat = idx_np.reshape(-1)
        k = idx_np.shape[-1]
        xf = mx.expand_dims(x, (-2, -3)).flatten(0, -3)  # [T, 1, H]
        res = slot_of[flat] >= 0
        parts, outs = [], []
        for mask, rhs, arrays in ((res, slot_of[flat], slots_now), (~res, bufpos[flat], buf)):
            pos = np.nonzero(mask)[0]
            if not len(pos):
                continue
            pos = pos[np.argsort(rhs[pos], kind="stable")]
            r = rhs[pos].astype(np.uint32)
            xi = xf[mx.array(pos // k)]
            if fast:
                # Weights sliced to the experts used, zero rows on the last
                # of them padded up to the fast kernel's minimum.
                m = int(r[-1]) + 1
                pad = max(4 * m, 16) - len(r)
                arrays = [a[:m] for a in arrays]
                if pad > 0:
                    xi = mx.concatenate([xi, mx.zeros((pad, *xi.shape[1:]), xi.dtype)])
                    r = np.concatenate([r, np.full(pad, m - 1, np.uint32)])
                chunks = [(0, len(r))]
            else:
                step = 4 * arrays[0].shape[0] - 1
                chunks = [(i, min(i + step, len(r))) for i in range(0, len(r), step)]
            lin = dict(zip(self.fields_key, arrays))

            def qmm(v, proj, ri):
                lp = getattr(sg, proj)
                return mx.gather_qmm(
                    v, lin[(proj, "weight")], lin[(proj, "scales")], lin[(proj, "biases")],
                    rhs_indices=ri, transpose=True, group_size=lp.group_size,
                    bits=lp.bits, mode=lp.mode, sorted_indices=True,
                )

            ys = []
            for a, b in chunks:
                ri, xc = mx.array(r[a:b]), xi[a:b]
                h = sg.activation(qmm(xc, "up_proj", ri), qmm(xc, "gate_proj", ri))
                ys.append(qmm(h, "down_proj", ri))
            y = ys[0] if len(ys) == 1 else mx.concatenate(ys)
            outs.append(y[: len(pos)])
            parts.append(pos)
        perm = np.concatenate(parts)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        y = mx.concatenate(outs, axis=0)[mx.array(inv)]
        y = mx.unflatten(y, 0, idx_np.shape).squeeze(-2)
        return (mx.expand_dims(w, -1) * y).sum(-2)

    def prefill(self, idx_np, x, w):
        """Run the experts over a whole routed set, then cache the most recent ones."""
        XG[0].drain()
        XG[0].wait_all(self.xid)
        slot_of = np.array(XG[0].slot_of(self.xid), np.int64)
        flat = idx_np.reshape(-1)
        u = np.unique(flat)
        slots_now = self._arrays()
        miss = u[slot_of[u] < 0]
        buf = _miss_buffer(_MISS_ROWS[0], slots_now)
        STATS["prefill_reads"] += len(miss)
        self._read([(int(e), r) for r, e in enumerate(miss)], buf)
        bufpos = np.full(self.num_experts, -1, np.int64)
        bufpos[miss] = np.arange(len(miss))

        # The kernel regime is the stock call's (all T*k rows over
        # num_experts); the rows are then computed TOKEN_CHUNK tokens at a
        # time so only one chunk's temporaries are alive.
        fast = len(flat) >= max(4 * self.num_experts, 16)
        outs = []
        for t0 in range(0, idx_np.shape[1], TOKEN_CHUNK):
            sl = slice(t0, t0 + TOKEN_CHUNK)
            o = self._compute(idx_np[:, sl], x[:, sl], w[:, sl], fast, slots_now, buf, bufpos, slot_of)
            mx.eval(o)
            outs.append(o)
        out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)
        mx.eval(out)
        # The parts' temporaries differ in size from layer to layer, so the
        # allocator cache can't reuse them; let them go.
        mx.clear_cache()

        # LRU order = order of last use within the prompt; keep the newest C.
        last = OrderedDict()
        for e in flat.tolist():
            last.pop(e, None)
            last[e] = None
        newest = list(last)[-self.capacity:]
        need = [e for e in newest if slot_of[e] < 0]
        if need:
            # The GPU is done with this layer (out was evaluated): copy on the
            # CPU so the slot tensors keep their buffers.
            bv = [_bytes(t) for t in buf]
            sv = [_bytes(a) for a in slots_now]
            slots = XG[0].evict_for(self.xid, len(need), newest)
            for e, s in zip(need, slots):
                for t, a in zip(bv, sv):
                    a[s] = t[bufpos[e]]
            XG[0].assign(self.xid, need, slots)
        XG[0].touch(self.xid, newest)
        return out


def install(model, store, budget, prefetch=1, threads=4):
    """Replace every MoE layer's experts with a streamed cache.

    budget: resident experts per MoE layer (one int for all, or one per layer).
    prefetch: experts of the next layer to prefetch per gate (0 = off).
    """
    import mlx_vlm.models.gemma4.language as gl

    from . import native

    store = Path(store).expanduser()
    layout = json.loads((store / "layout.json").read_text())
    if XG[0] is None:
        XG[0] = native.load()
    XG[0].set_prefetch(prefetch)
    _RECORD[0] = layout["record_bytes"]
    pool = ThreadPoolExecutor(threads)
    layers = model.language_model.model.layers
    caches = []
    for i, layer in enumerate(layers):
        if not getattr(layer, "enable_moe", False):
            continue
        assert i in layout["layers"]
        assert layer.per_layer_input_gate is None and layer.layer_scalar is not None
        cap = budget[len(caches)] if isinstance(budget, (list, tuple)) else budget
        c = LayerCache(layer, i, store, layout, cap, pool)
        _MISS_ROWS[0] = max(_MISS_ROWS[0], layout["num_experts"] - cap)
        layer.__dict__["_xcache"] = c
        layer.__dict__["_xroute"] = mx.compile(lambda a, b, L=layer: _route(L, a, b))
        if caches:
            # The previous MoE layer also predicts this layer's routing.
            prev = caches[-1].layer
            prev.__dict__["_xroute"] = mx.compile(lambda a, b, L=prev, N=layer: _route(L, a, b, N))
            XG[0].set_next(caches[-1].xid, c.xid)
        layer.__dict__["_xdense"] = mx.compile(lambda a, L=layer: _dense(L, a))
        layer.__dict__["_xpost"] = mx.compile(lambda r, h1, y, L=layer: _post(L, r, h1, y))
        caches.append(c)
    if isinstance(budget, (list, tuple)) and len(budget) != len(caches):
        raise ValueError(f"budget has {len(budget)} entries, the model has {len(caches)} MoE layers")

    orig = gl.DecoderLayer.__call__

    def call(self, x, mask=None, cache=None, per_layer_input=None, shared_kv=None, offset=None):
        xc = self.__dict__.get("_xcache")
        if xc is None:
            return orig(self, x, mask, cache, per_layer_input, shared_kv, offset)
        h, shared_kv, offset = self.self_attn(
            self.input_layernorm(x), mask, cache, shared_kv=shared_kv, offset=offset
        )
        residual, idx, w, h2, *pidx = self.__dict__["_xroute"](h, x)
        h1 = self.__dict__["_xdense"](residual)
        if h.shape[-2] < PREFILL_MIN:
            # Slot indices come out of the command stream: the GPU waits at
            # the gate while the extension's worker looks the experts up.
            sidx = XG[0].gate(xc.xid, idx, pidx[0] if prefetch and pidx else None)
            out = self.__dict__["_xpost"](residual, h1, self.experts(h2, sidx, w))
            mx.async_eval(out)
            return out, shared_kv, offset
        t = time.perf_counter()
        idx_np = np.array(idx)
        STATS["sync_s"] += time.perf_counter() - t
        y = xc.prefill(idx_np, h2, w)
        return self.__dict__["_xpost"](residual, h1, y), shared_kv, offset

    gl.DecoderLayer.__call__ = call
    return caches


def _route(layer, h, x, nxt=None):
    h = x + layer.post_attention_layernorm(h)
    idx, w = layer.router(h)
    # Contiguous: the native gate reads the indices from memory as laid out.
    out = (h, mx.contiguous(idx), w, layer.pre_feedforward_layernorm_2(h))
    if nxt is not None:
        out += (mx.contiguous(nxt.router(h)[0]),)  # next MoE layer's routing, guessed from this residual
    return out


def _dense(layer, h):
    return layer.post_feedforward_layernorm_1(layer.mlp(layer.pre_feedforward_layernorm(h)))


def _post(layer, residual, h1, y):
    h = residual + layer.post_feedforward_layernorm(h1 + layer.post_feedforward_layernorm_2(y))
    return h * layer.layer_scalar


def reset_stats():
    for k in STATS:
        STATS[k] = 0
    if XG[0] is not None:
        XG[0].reset_stats()


def stats():
    """Streaming counters: Python prefill side plus the native gate's."""
    XG[0].drain()
    nat = XG[0].stats()
    return {
        "decode_lookups": nat["decode_lookups"],
        "decode_misses": nat["decode_misses"],
        "decode_hit_rate": 1 - nat["decode_misses"] / max(nat["decode_lookups"], 1),
        "prefill_reads": STATS["prefill_reads"],
        "read_s": STATS["read_s"] + nat["read_s"],
        "gate_hold_s": nat["hold_s"],
        "gates": nat["gates"],
        "prefetch_reads": nat["prefetch_reads"],
        "prefetch_used": nat["prefetch_used"],
    }
