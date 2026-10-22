# gemmaflash

Fast, low-memory decoding of **Gemma 4 26B-A4B** (QAT 4-bit) on Apple Silicon.
You pick how many experts per layer stay in memory; the rest stream from the SSD.
Greedy output is byte-identical to the stock model at every setting.

| experts per layer (mean) | memory (peak footprint) | decode |
|---|---|---|
| 128 (all resident) | 15.9 GB† | 109 tok/s |
| 96 | 12.4 GB† | 99 tok/s |
| 80 (default) | 10.8 GB | 89 tok/s |
| 64 | 9.3 GB | 66 tok/s |

M5 Pro, 24 GB; 8 prompts, 256 greedy tokens each. Stock mlx-lm gets 79 tok/s on the same set.
Expect a few % run-to-run variance. † Measured without the 64 MB MLX cache cap that is on by default.

## Install

Needs an Apple Silicon Mac (macOS 26.2+), Python 3.10+, `cmake` and the Xcode command line tools.

```sh
pip install .
```

On first use the model (~15 GB) is downloaded, the expert store (~12 GB) is written to
`~/.cache/gemmaflash/experts`, and the native gate is compiled (a few seconds).

## Use

```sh
gemmaflash run "Explain the CAP theorem." --experts-mean 80
gemmaflash chat --experts-mean 72
gemmaflash bench --experts-mean 64        # speed + memory on the bundled prompts
gemmaflash budget --experts-mean 70       # show the per-layer split, no model load
```

```python
from gemmaflash import Runtime

rt = Runtime(experts_mean=80)
for piece in rt.stream("Write a haiku about SSDs.", max_tokens=256):
    print(piece, end="")
print(rt.last["decode_tps"], rt.stats()["decode_hit_rate"])
```

## Dialing memory

`experts_mean` (32 to 128) is the average number of each MoE layer's 128 experts kept in memory;
each unit costs ~100 MB (30 layers × 3.36 MB). Layers are not equal, so the mean is split
unevenly: every layer starts at 32 and slots go, two at a time, to the layer whose cache misses
drop most, using LRU misses simulated on a routing trace of 12 held-out prompts.
`gemmaflash budget` prints the split and the trace's hit rate. To set layers by hand, pass 30
counts: `--budget 102,112,80,...` or `Runtime(expert_budget=[...])`.

## How it works

- **MTP speculative decoding**: Gemma 4's assistant drafter proposes 2 tokens per round;
  the target verifies them with a single batched argmax. The drafter's head scores only the 64k most
  frequent tokens.
- **Compiled MoE half and an in-place sliding-window KV cache**, both bit-exact.
- **SSD expert streaming**: each layer's resident experts live in slot tensors; misses are read
  with `pread` (`F_NOCACHE`) straight into them. Prefill splits the expert matmuls between resident
  and freshly read experts without changing a bit.
- **Native spin gate** (`gemmaflash/native/xgate.cpp`): an MLX primitive in the command stream.
  The GPU writes a coherent flag and spins; a C++ worker reads the router output, does the LRU
  lookup and SSD reads, writes slot indices and releases the GPU. Python never syncs during decode.
  A Metal shared event costs ~130 µs per round trip; the spin costs about nothing.
- **Next-layer prefetch**: the worker reads the next layer's predicted top expert while the GPU computes.

Limits: text only, one generation at a time, and the speed work targets greedy decoding (sampling
works but is slower). mlx 0.32.4 and mlx-vlm 0.7.9 are pinned because the runtime patches their internals.

## Papers

Used in the experiments behind this runtime:

- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (2022): draft-then-verify.
- Zhao et al., [FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling](https://arxiv.org/abs/2502.14856) (2025): the drafter's reduced vocabulary.
- McDanel et al., [MoE-Spec: Expert Budgeting for Efficient Speculative Decoding](https://arxiv.org/abs/2602.16052) (2026): verify cost tracks unique experts.
- Kim et al., [FlashMoE: Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for Mixture-of-Experts Inference on Edge Devices](https://arxiv.org/abs/2601.17063) (2026): SSD expert store with per-layer LRU cache.
- Zhong et al., [AdapMoE: Adaptive Sensitivity-based Expert Gating and Management for Efficient MoE Inference](https://arxiv.org/abs/2408.10284) (2024): per-layer cache budgets.
- Madan et al., [Speculating Experts Accelerates Inference for Mixture-of-Experts](https://arxiv.org/abs/2603.19289) (2026): next-layer expert prefetch.
- Li et al., [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://doi.org/10.52202/085713-4562) (2025): drafter tested, MTP won.
- Chen et al., [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) (2026): drafter tested, MTP won.

Reviewed, not implemented:
[LLM in a flash](https://doi.org/10.18653/v1/2024.acl-long.678) ·
[Hiding Offloading Latency with Speculative Decoding](https://arxiv.org/abs/2508.21706) ·
[MoE Inference with Apple Silicon NPUs](https://arxiv.org/abs/2604.18788) ·
[SpecFit](https://arxiv.org/abs/2610.15417) ·
[BigMoMo](https://arxiv.org/abs/2609.14643) ·
[FLE: Fast Local Experts](https://doi.org/10.5281/zenodo.23272400) ·
[Lever](https://arxiv.org/abs/2605.16786) ·
[Dynamic Expert Quantization](https://arxiv.org/abs/2511.15015) ·
[RAZOR](https://arxiv.org/abs/2609.30465) ·
[REAM](https://arxiv.org/abs/2604.04356) ·
[BaseRT](https://arxiv.org/abs/2607.19438) ·
[LLM-42](https://arxiv.org/abs/2601.17768)
