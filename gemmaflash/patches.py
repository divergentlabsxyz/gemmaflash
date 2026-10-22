"""Bit-exact patches to mlx-vlm's Gemma 4 decode path (mlx-vlm 0.7.9)."""

import json
import re
from pathlib import Path

import mlx.core as mx

DATA = Path(__file__).parent / "data"


def install_batched_greedy_verify():
    """Greedy MTP verify: argmax every position in one projection.

    Stock mlx-vlm leaves target_tokens unset for Gemma 4, so acceptance falls
    back to projecting the lm_head one position at a time with a sync after
    each. Attaching the batched argmax to the verify graph gives the same
    greedy tokens with a single lm_head read and a single sync.
    """
    import mlx_vlm.speculative.mtp as mtp

    orig = mtp._mtp_verify_target

    def verify(lm, verify_input, prompt_cache, sampler, *, sample_target_tokens=True):
        res = orig(lm, verify_input, prompt_cache, sampler, sample_target_tokens=sample_target_tokens)
        if sample_target_tokens and res.target_tokens is None:
            logits = mtp._mtp_logits_from_hidden(lm, res.hidden)
            res.target_tokens = mx.argmax(logits, axis=-1)
        return res

    mtp._mtp_verify_target = verify


def install_compiled_ffn():
    """mx.compile the pure post-attention half of each Gemma 4 MoE layer.

    Attention mutates the KV cache, so it stays eager; everything after it
    (norms, residuals, dense MLP, router, experts) is a pure function of the
    attention output and the layer input. Logits are bit-identical.
    """
    import mlx_vlm.models.gemma4.language as gl

    def ffn_half(layer, h, residual):
        h = residual + layer.post_attention_layernorm(h)
        residual = h
        h1 = layer.post_feedforward_layernorm_1(layer.mlp(layer.pre_feedforward_layernorm(h)))
        idx, w = layer.router(h)
        h2 = layer.experts(layer.pre_feedforward_layernorm_2(h), idx, w)
        h2 = layer.post_feedforward_layernorm_2(h2)
        h = residual + layer.post_feedforward_layernorm(h1 + h2)
        return h * layer.layer_scalar

    orig = gl.DecoderLayer.__call__

    def call(self, x, mask=None, cache=None, per_layer_input=None, shared_kv=None, offset=None):
        if not self.enable_moe or self.per_layer_input_gate is not None or self.layer_scalar is None:
            return orig(self, x, mask, cache, per_layer_input, shared_kv, offset)
        h, shared_kv, offset = self.self_attn(
            self.input_layernorm(x), mask, cache, shared_kv=shared_kv, offset=offset
        )
        f = self.__dict__.get("_ffn_compiled")
        if f is None:
            f = self.__dict__["_ffn_compiled"] = mx.compile(lambda a, b: ffn_half(self, a, b))
        return f(h, x), shared_kv, offset

    gl.DecoderLayer.__call__ = call


class SubsetDraftHead:
    """Drafter head over a frequency-ranked token subset (FR-Spec).

    The assistant drafter ties its head to a 262k-token 4-bit embedding;
    projecting onto all of it is ~40% of a draft step. Drafts are only
    proposals: the target's verify still picks every emitted token, so a
    smaller draft vocabulary can lower acceptance but never change outputs.
    Ranking comes from tools/build_draft_vocab.py.
    """

    def __init__(self, embed, n):
        import numpy as np

        rank = np.load(DATA / "draft_vocab_rank.npy")[:n]
        self.ids = mx.array(np.sort(rank))
        self.vocab = embed.weight.shape[0]
        self.w = embed.weight[self.ids]
        self.s = embed.scales[self.ids]
        self.b = embed.biases[self.ids]
        self.group_size, self.bits = embed.group_size, embed.bits
        mx.eval(self.ids, self.w, self.s, self.b)

    def _logits(self, hidden):
        return mx.quantized_matmul(
            hidden, self.w, self.s, self.b, transpose=True,
            group_size=self.group_size, bits=self.bits,
        )

    def argmax(self, hidden, _weight=None):
        """Greedy drafting."""
        return self.ids[mx.argmax(self._logits(hidden), axis=-1)]

    def __call__(self, hidden, _weight=None):
        """Sampled drafting: full-vocab logits, -inf outside the subset."""
        sub = self._logits(hidden)
        full = mx.full((*sub.shape[:-1], self.vocab), -mx.inf, sub.dtype)
        full[..., self.ids] = sub
        return full


def install_light_detokenizer_setup():
    """Pick mlx-vlm's streaming detokenizer class without parsing tokenizer.json.

    Stock load_tokenizer json.loads the whole 31 MB file (a Python object tree
    the heap keeps after it is freed, ~0.3 GB of footprint) only to read its
    "decoder" entry. Read that entry from an mmap of the file instead; the
    same class is chosen.
    """
    import mmap
    from functools import partial

    import mlx_vlm.tokenizer_utils as TU
    import mlx_vlm.utils as U

    def decoder_spec(path):
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for m in re.finditer(rb'"decoder"\s*:\s*', mm):
                head = mm[m.end():m.end() + 65536].decode("utf-8", "ignore")
                try:
                    v, _ = json.JSONDecoder().raw_decode(head)
                except json.JSONDecodeError:
                    continue
                if v is None or (isinstance(v, dict) and "type" in v):
                    return v

    orig = U.load_tokenizer

    def load_tokenizer(model_path, return_tokenizer=True, tokenizer_config_extra={}):
        if return_tokenizer or not (model_path / "tokenizer.json").exists():
            return orig(model_path, return_tokenizer, tokenizer_config_extra)
        d = decoder_spec(model_path / "tokenizer.json")
        if d is not None and TU._is_spm_decoder(d):
            return TU.SPMStreamingDetokenizer
        if d is not None and TU._is_spm_decoder_no_space(d):
            return partial(TU.SPMStreamingDetokenizer, trim_space=False)
        if d is not None and TU._is_bpe_decoder(d):
            return TU.BPEStreamingDetokenizer
        return TU.NaiveStreamingDetokenizer

    U.load_tokenizer = load_tokenizer
