"""Sliding-window KV cache that writes new tokens in place.

mlx-vlm's RotatingKVCache serves any multi-token update (every speculative
verify) by concatenating: it rebuilds the window in temporal order, drops the
oldest tokens and appends the new ones, copying the whole window of K and V
in every sliding layer on every verify.

WindowKVCache keeps the tokens in temporal order in a linear buffer with
headroom. An update slice-writes the new tokens after the last valid one and
returns a view of the last min(n, W - 1) valid tokens plus the new ones,
the same keys in the same order the concatenating path produces, so
attention sees identical inputs. The buffer is compacted (last W - 1 tokens
moved to the front) only when the headroom runs out.
"""

import mlx.core as mx
from mlx_vlm.models.cache import RotatingKVCache

HEADROOM = 256


class WindowKVCache(RotatingKVCache):
    def __init__(self, max_size, keep=0):
        assert keep == 0
        super().__init__(max_size, keep)
        self._buf_k = None
        self._buf_v = None
        self._end = 0  # valid tokens stored in the buffer, oldest first

    def _alloc(self, like_k, like_v, size):
        B, H, _, dk = like_k.shape
        dv = like_v.shape[3]
        return mx.zeros((B, H, size, dk), like_k.dtype), mx.zeros((B, H, size, dv), like_v.dtype)

    def update_and_fetch(self, keys, values):
        S = keys.shape[2]
        n_prev = min(self._end, self.max_size - 1)
        # Drop the previous window views so the buffer can be updated in place.
        self.keys = self.values = None
        if self._buf_k is None:
            self._buf_k, self._buf_v = self._alloc(keys, values, S + HEADROOM)
        elif self._end + S > self._buf_k.shape[2]:
            size = n_prev + S + HEADROOM
            new_k, new_v = self._alloc(keys, values, size)
            if n_prev:
                new_k[..., :n_prev, :] = self._buf_k[..., self._end - n_prev : self._end, :]
                new_v[..., :n_prev, :] = self._buf_v[..., self._end - n_prev : self._end, :]
            self._buf_k, self._buf_v = new_k, new_v
            self._end = n_prev
        self._buf_k[..., self._end : self._end + S, :] = keys
        self._buf_v[..., self._end : self._end + S, :] = values
        start = self._end - n_prev
        self._end += S
        self.offset += S
        self.keys = self._buf_k[..., start : self._end, :]
        self.values = self._buf_v[..., start : self._end, :]
        self._idx = self.keys.shape[2]
        return self.keys, self.values

    def trim(self, n):
        n = min(self.offset, n, self._end)
        self.offset -= n
        self._end -= n
        if n and self.keys is not None:
            self.keys = self.keys[..., : self.keys.shape[2] - n, :]
            self.values = self.values[..., : self.values.shape[2] - n, :]
        self._idx = 0 if self.keys is None else self.keys.shape[2]
        return n

    def _temporal_order(self, v):
        return v  # always stored oldest first

    @property
    def state(self):
        return self.keys, self.values

    @state.setter
    def state(self, v):
        raise NotImplementedError("WindowKVCache does not restore state")

    def make_mask(self, N, window_size=None, return_array=False):
        if N == 1:
            # The window view never holds more than max_size tokens and is in
            # temporal order, so a single query needs no mask.
            return None
        return super().make_mask(N, window_size=window_size, return_array=return_array)

    @property
    def nbytes(self):
        if self._buf_k is None:
            return 0
        return self._buf_k.nbytes + self._buf_v.nbytes


def install():
    import mlx_vlm.models.gemma4.language as gl

    orig = gl.LanguageModel.make_cache

    def make_cache(self):
        return [
            WindowKVCache(c.max_size, c.keep) if type(c) is RotatingKVCache else c
            for c in orig(self)
        ]

    gl.LanguageModel.make_cache = make_cache
