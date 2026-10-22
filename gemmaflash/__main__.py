"""Command line: gemmaflash {run,chat,bench,budget} [--experts-mean N] ..."""

import argparse
import json
import sys
from pathlib import Path

from . import budget


def parse_budget(s):
    b = [int(x) for x in s.split(",")]
    return b if len(b) > 1 else b[0]


def add_model_args(p):
    p.add_argument("--experts-mean", type=float, default=80,
                   help="mean routed experts kept in memory per layer, 32..128 (default 80; 128 = all resident)")
    p.add_argument("--budget", type=parse_budget, default=None,
                   help="explicit per-layer resident experts: 30 comma-separated ints, or one int for every layer")
    p.add_argument("--prefetch", type=int, default=1, help="next-layer experts prefetched per gate (0 = off)")
    p.add_argument("--cache-limit-mb", type=float, default=64, help="MLX allocator cache cap")
    p.add_argument("--no-draft", action="store_true", help="plain autoregressive decoding (no MTP drafter)")


def add_gen_args(p):
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--thinking", action="store_true", help="let the model think before answering")


def load(a):
    from .runtime import DRAFT_MODEL, Runtime

    return Runtime(experts_mean=a.experts_mean, expert_budget=a.budget, prefetch=a.prefetch,
                   cache_limit_mb=a.cache_limit_mb, draft_model=None if a.no_draft else DRAFT_MODEL)


def cmd_budget(a):
    b = budget.expand(a.budget) if a.budget is not None else budget.fit(a.experts_mean)
    d = budget.describe(b)
    hit = f"{d['heldout_hit_rate'] * 100:.2f}%" if d["heldout_hit_rate"] is not None else "n/a"
    print(f"mean {d['mean']:.2f}  resident experts {d['resident_gb']:.2f} GB  held-out decode hit rate {hit}")
    print(",".join(map(str, b)))


def cmd_run(a):
    rt = load(a)
    for piece in rt.stream(a.prompt, max_tokens=a.max_tokens, temperature=a.temperature, thinking=a.thinking):
        print(piece, end="", flush=True)
    print()
    s = rt.last
    print(f"\n[{s['gen_tokens']} tokens, ttft {s['ttft_s']:.2f}s, decode {s['decode_tps']:.1f} tok/s]", file=sys.stderr)


def cmd_chat(a):
    rt = load(a)
    msgs = []
    while True:
        try:
            q = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q.strip():
            continue
        msgs.append({"role": "user", "content": q})
        out = ""
        for piece in rt.stream(msgs, max_tokens=a.max_tokens, temperature=a.temperature, thinking=a.thinking):
            out += piece
            print(piece, end="", flush=True)
        print(f"\n[{rt.last['decode_tps']:.1f} tok/s]", file=sys.stderr)
        msgs.append({"role": "assistant", "content": out})


def cmd_bench(a):
    from .runtime import footprint_gb

    prompts = json.loads((Path(__file__).parent / "data/bench_prompts.json").read_text())
    rt = load(a)
    rt.generate("Say hello in five words.", max_tokens=32)  # warmup / compile
    rt.reset_stats()
    rows = []
    for p in prompts:
        text = rt.generate(p["text"], max_tokens=a.max_tokens)
        r = dict(rt.last, id=p["id"], text=text)
        print(f"[{p['id']}] prompt {r['prompt_tokens']} gen {r['gen_tokens']} ttft {r['ttft_s']:.2f}s "
              f"decode {r['decode_tps']:.1f} tok/s", file=sys.stderr)
        rows.append(r)

    def tps(rs):
        return sum(r["decode_tokens"] for r in rs) / sum(r["decode_s"] for r in rs)

    res = {
        "decode_tps": round(tps(rows), 2),
        "decode_tps_short": round(tps([r for r in rows if r["id"] != "longctx"]), 2),
        "decode_tps_longctx": round(tps([r for r in rows if r["id"] == "longctx"]), 2),
        "mean_ttft_s": round(sum(r["ttft_s"] for r in rows) / len(rows), 3),
        "footprint_peak_gb": round(footprint_gb()[1], 2),
        "experts_mean": round(sum(rt.budget) / len(rt.budget), 2) if rt.budget else 128,
    }
    st = rt.stats()
    if st:
        res.update(decode_hit_rate=round(st["decode_hit_rate"], 4), gate_hold_s=round(st["gate_hold_s"], 2))
    print(json.dumps(res))
    if a.out:
        Path(a.out).write_text(json.dumps({"result": res, "generations": [
            {"id": r["id"], "text": r["text"]} for r in rows]}, indent=1))


def main():
    ap = argparse.ArgumentParser(prog="gemmaflash", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="generate from one prompt")
    p.add_argument("prompt")
    add_model_args(p)
    add_gen_args(p)
    p = sub.add_parser("chat", help="interactive multi-turn chat")
    add_model_args(p)
    add_gen_args(p)
    p = sub.add_parser("bench", help="decode speed and memory on the bundled prompts (greedy, 256 tokens)")
    add_model_args(p)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--out", help="write the result and generations as JSON here")
    p = sub.add_parser("budget", help="print the fitted per-layer budget for a mean (no model load)")
    add_model_args(p)
    a = ap.parse_args()
    {"run": cmd_run, "chat": cmd_chat, "bench": cmd_bench, "budget": cmd_budget}[a.cmd](a)


if __name__ == "__main__":
    main()
