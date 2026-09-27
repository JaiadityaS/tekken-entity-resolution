"""Stage 2 driver: blocking for a whole split -> work/{split}_cands_raw.parquet.

Usage: python make_candidates.py train|test
"""
import sys
import time

from blocking import generate_candidates
from common import load_sources
from config import WORK_DIR
from params import BLOCK

if __name__ == "__main__":
    split = sys.argv[1]
    t = time.time()
    s1, r = load_sources(split)
    print(f"{split}: s1={len(s1):,} r={len(r):,}", flush=True)
    cands = generate_candidates(s1, r, **BLOCK, log=lambda m: print(m, flush=True))
    cands.to_parquet(WORK_DIR / f"{split}_cands_raw.parquet", index=False)
    print(f"{split}: {len(cands):,} raw candidate pairs in {time.time() - t:.0f}s")
