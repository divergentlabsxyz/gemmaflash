"""Gemma 4 26B-A4B on Apple Silicon: MTP speculative decoding + SSD-streamed experts."""

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

from . import budget as _budget
from . import patches

MODEL = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"
DRAFT_MODEL = "mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit"
STORE_ROOT = Path("~/.cache/gemmaflash/experts").expanduser()
LAZY = {"vision_tower", "embed_vision", "audio_tower", "embed_audio"}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class _RusageV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
        "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime",
        "child_user_time", "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
        "child_pageins", "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten",
        "qos_default", "qos_maintenance", "qos_background", "qos_utility", "qos_legacy",
        "qos_user_initiated", "qos_user_interactive", "billed_system_time", "serviced_system_time",
        "logical_writes", "lifetime_max_phys_footprint", "instructions", "cycles", "billed_energy",
        "serviced_energy", "interval_max_phys_footprint", "runnable_time")]


def footprint_gb():
    """This process's physical footprint (Activity Monitor's Memory), GB: now and lifetime max."""
    r = _RusageV4()
    ctypes.CDLL("/usr/lib/libSystem.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.byref(r))
    return r.phys_footprint / 2**30, r.lifetime_max_phys_footprint / 2**30


class Runtime:
    """Load once, then stream() or generate().

    experts_mean: mean routed experts kept in memory per MoE layer (32..128);
        the rest are read from SSD on demand. 128 keeps every expert resident.
    expert_budget: explicit per-layer counts (30 ints); overrides experts_mean.
    prefetch: next-layer experts the gate prefetches per layer (0 = off).
    cache_limit_mb: MLX allocator cache cap (None = MLX default).
    draft_vocab: drafter tokens considered (FR-Spec); None = full vocabulary.
    """

    def __init__(self, experts_mean=80, expert_budget=None, *, model=MODEL, draft_model=DRAFT_MODEL,
                 draft_block_size=3, draft_vocab=65536, prefetch=1, cache_limit_mb=64, store=None,
                 verbose=True):
        from mlx_vlm import load

        say = log if verbose else (lambda *a: None)
        t = time.perf_counter()
        if expert_budget is None and experts_mean is not None and experts_mean < _budget.NUM_EXPERTS:
            expert_budget = _budget.fit(experts_mean)
        self.budget = _budget.expand(expert_budget) if expert_budget is not None else None
        self.draft_block_size = draft_block_size
        if cache_limit_mb is not None:
            mx.set_cache_limit(int(cache_limit_mb * 2**20))

        patches.install_light_detokenizer_setup()
        # Lazy: streamed experts and vision weights are never read in.
        self.model, self.processor = load(model, lazy=True)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        patches.install_batched_greedy_verify()
        patches.install_compiled_ffn()
        from . import window_cache

        window_cache.install()
        if self.budget:
            from . import expert_stream

            store = Path(store).expanduser() if store else STORE_ROOT / model.split("/")[-1]
            if not (store / "layout.json").exists():
                self._build_store(model, store, say)
            expert_stream.install(self.model, store, self.budget, prefetch=prefetch)
            d = _budget.describe(self.budget)
            say(f"experts: mean {d['mean']:.1f} of 128 per layer resident ({d['resident_gb']:.2f} GB), rest streamed from {store}")
        mx.eval({k: v for k, v in self.model.parameters().items() if k not in LAZY})

        self.draft = self.draft_kind = None
        if draft_model:
            from mlx_vlm.speculative.drafters import load_drafter

            self.draft, self.draft_kind = load_drafter(draft_model, kind="mtp")
            if draft_vocab:
                assert self.draft.masked_embedding is None
                self.draft.masked_embedding = patches.SubsetDraftHead(self.draft.model.embed_tokens, draft_vocab)
        self.last = {}
        say(f"loaded in {time.perf_counter() - t:.1f}s, footprint {footprint_gb()[0]:.2f} GB")

    @staticmethod
    def _build_store(model, store, say):
        from huggingface_hub import snapshot_download

        snap = snapshot_download(model)
        say(f"writing the expert store to {store} (once, ~12 GB)...")
        subprocess.run([sys.executable, str(Path(__file__).parent / "store.py"), snap, str(store)], check=True,
                       stdout=subprocess.DEVNULL)

    def format(self, prompt, thinking=False):
        """Chat-template a user string or a list of {"role", "content"} messages."""
        msgs = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        return self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, enable_thinking=thinking
        )

    def stream(self, prompt, max_tokens=512, temperature=0.0, thinking=False, raw=False):
        """Yield text pieces. prompt: a user string, a message list, or (raw=True) a formatted prompt."""
        from mlx_vlm import stream_generate

        text = prompt if raw else self.format(prompt, thinking)
        kw = dict(temperature=temperature, max_tokens=max_tokens)
        if self.draft is not None:
            kw.update(draft_model=self.draft, draft_kind=self.draft_kind, draft_block_size=self.draft_block_size)
        t0 = time.perf_counter()
        t_first = None
        n = n_first = n_prompt = 0
        for r in stream_generate(self.model, self.processor, text, **kw):
            if r.is_draft:
                continue
            n = r.generation_tokens or n + 1
            n_prompt = r.prompt_tokens
            if t_first is None:
                t_first, n_first = time.perf_counter(), n
            yield r.text
        t_end = time.perf_counter()
        dt = t_end - (t_first or t_end)
        self.last = dict(prompt_tokens=n_prompt, gen_tokens=n, ttft_s=(t_first or t_end) - t0,
                         decode_tokens=n - n_first, decode_s=dt,
                         decode_tps=(n - n_first) / dt if dt > 0 else 0.0)

    def generate(self, prompt, **kw):
        return "".join(self.stream(prompt, **kw))

    def stats(self):
        """Expert-streaming counters since load (or reset_stats())."""
        if not self.budget:
            return {}
        from . import expert_stream

        return expert_stream.stats()

    def reset_stats(self):
        if self.budget:
            from . import expert_stream

            expert_stream.reset_stats()
