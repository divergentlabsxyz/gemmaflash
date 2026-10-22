"""Rank Gemma 4 tokens by frequency in general assistant text (FR-Spec).

Corpus: assistant turns of HuggingFaceTB/smol-smoltalk (train shard 0) and
nvidia/HelpSteer2 responses, both from the local HF cache. GSM8K is left out
on purpose: the bench's math prompts are GSM8K problems.

Writes gemmaflash/data/draft_vocab_rank.npy: all token ids that occur, most frequent
first (ids < 1024, the special/control range, are put at the front).
"""

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

OUT = Path(__file__).resolve().parent.parent / "gemmaflash/data/draft_vocab_rank.npy"
MAX_DOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 60000

tok = AutoTokenizer.from_pretrained("mlx-community/gemma-4-26B-A4B-it-qat-4bit")
texts = []
smol = Path(snapshot_download("HuggingFaceTB/smol-smoltalk", repo_type="dataset", local_files_only=True))
tbl = pq.read_table(smol / "data/train-00000-of-00004.parquet", columns=["messages"])
for msgs in tbl.column("messages").to_pylist()[:MAX_DOCS]:
    texts += [m["content"] for m in msgs if m["role"] == "assistant"]
hs = Path(snapshot_download("nvidia/HelpSteer2", repo_type="dataset", local_files_only=True))
with gzip.open(hs / "train.jsonl.gz", "rt") as f:
    texts += [json.loads(line)["response"] for line in f]

counts = Counter()
for i in range(0, len(texts), 2000):
    for ids in tok(texts[i : i + 2000], add_special_tokens=False)["input_ids"]:
        counts.update(ids)
total = sum(counts.values())
special = list(range(1024))
ranked = special + [t for t, _ in counts.most_common() if t >= 1024]
np.save(OUT, np.array(ranked, dtype=np.int32))
print(f"{len(texts)} texts, {total/1e6:.1f}M tokens, {len(counts)} distinct ids -> {len(ranked)} ranked")
