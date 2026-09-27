"""Stage 6: inference on the test split.

raw blocking candidates -> stage A filter (p0 >= tau) = candidate_pairs.tsv
                        -> stage B (p1) -> 1-1 assignment + threshold = matching_results.tsv
Usage: python predict.py
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import load_sources, stream_stage_a
from config import OUT_DIR, WORK_DIR
from decide import decide, write_lists
from features import add_group_features, index_frame

MODEL_DIR = WORK_DIR / "models"


def main(log=lambda m: print(m, flush=True)):
    t = time.time()
    meta = json.load(open(MODEL_DIR / "meta.json"))
    s1, r = load_sources("test")
    s1_ids = s1.entity_id.values
    cands = pd.read_parquet(WORK_DIR / "test_cands_raw.parquet")
    log(f"raw test candidates: {len(cands):,} ({len(cands) / len(s1):.2f}/S1)")

    m_a = lgb.Booster(model_file=str(MODEL_DIR / "stageA.txt"))
    s1i, ri = index_frame(s1), index_frame(r)
    del r
    K, _ = stream_stage_a(cands, s1i, ri, m_a, meta["feats_a"], meta["tau"], WORK_DIR / "test_stageA",
                          n_iter=meta["stage_a_iters"], log=log)
    del cands, s1i, ri
    kept = K[["s1_id", "r_id", "p0", "bscore"]].copy()
    F = K.drop(columns=["s1_id", "r_id", "p0"])
    del K
    log(f"final candidates: {len(kept):,} ({len(kept) / len(s1):.2f}/S1)")
    # candidate_pairs.tsv = exactly the set the final model scores
    write_lists(kept, s1_ids, OUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids")

    G = add_group_features(kept, F)
    ms = [lgb.Booster(model_file=str(MODEL_DIR / f"stageB_{k}.txt")) for k in (0, 1)]
    kept["p1"] = np.mean([m.predict(G[meta["feats_b"]], num_threads=0) for m in ms], axis=0)
    kept[["s1_id", "r_id", "p0", "p1"]].to_parquet(WORK_DIR / "test_scores.parquet", index=False)

    matches = decide(kept, meta["thr"], "p1")
    out = write_lists(matches, s1_ids, OUT_DIR / "matching_results.tsv", "matched_entity_ids")
    n = (out.matched_entity_ids != "").sum()
    log(f"wrote {len(out):,} rows, {n:,} with matches, {len(matches):,} pairs "
        f"({len(matches) / len(out):.2f}/S1), thr={meta['thr']:.2f} [{time.time() - t:.0f}s]")
    for c in sorted(s1.country.unique()):
        ids = set(s1_ids[s1.country.values == c])
        mm = matches[matches.s1_id.isin(ids)]
        log(f"  {c}: {len(mm) / len(ids):.2f} matches/S1")


if __name__ == "__main__":
    main()
