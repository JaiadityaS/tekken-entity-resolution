"""Development helper: run blocking for a random sample of train S1 against the FULL
Source-2/3 pool and report recall vs. candidate-set size.

Usage: python dev_blocking.py [sample_frac] [max_df]
"""
import sys
import time

import numpy as np

from blocking import generate_candidates
from common import load_sources
from config import DATA_DIR, WORK_DIR
from evaluate import load_truth
from params import BLOCK

if __name__ == "__main__":
    frac = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    max_df = int(sys.argv[2]) if len(sys.argv) > 2 else BLOCK["max_df"]
    t = time.time()
    s1, r = load_sources("train")
    s1 = s1.sample(frac=frac, random_state=0)
    print(f"loaded s1={len(s1):,} r={len(r):,} in {time.time() - t:.0f}s", flush=True)
    cands = generate_candidates(s1, r, BLOCK["channels"], max_df=max_df,
                                min_score=BLOCK["min_score"], log=lambda m: print(m, flush=True))
    cands.to_parquet(WORK_DIR / f"dev_cands_{frac}_{max_df}.parquet", index=False)
    true, _ = load_truth(DATA_DIR / "train" / "train_ground_truth.tsv")
    true = true[true.s1_id.isin(set(s1.entity_id))]
    h = cands.merge(true.assign(y=1), on=["s1_id", "r_id"], how="left")
    h["y"] = h.y.fillna(0)
    n1 = len(s1)
    print(f"union: recall {h.y.sum() / len(true):.4f} cands/s1 {len(h) / n1:.1f}")
    for ch in [c[3:] for c in h.columns if c.startswith("sc_")]:
        for k in (5, 10, 20, 30):
            m = h[f"rk_{ch}"] < k
            print(f"  {ch:5s} top{k:<3d} recall {h.y[m].sum() / len(true):.4f} cands/s1 {m.sum() / n1:.1f}")
    for k in (3, 5, 8, 10, 15, 20):
        m = h.brank < k
        print(f"  union best-rank<{k:<3d} recall {h.y[m].sum() / len(true):.4f} cands/s1 {m.sum() / n1:.1f}")
    print(f"total {time.time() - t:.0f}s")
