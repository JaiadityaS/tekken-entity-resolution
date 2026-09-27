"""Rewrite matching_results.tsv from saved test scores with per-country thresholds (no retraining).

Needs work/test_scores.parquet and work/test_s1_norm.parquet (written by predict.py / prepare.py).
Usage: python rethreshold.py --thr 0.70 --country France=0.85 [--out ../../../output/matching_France085.tsv]
"""
import argparse
from pathlib import Path

import pandas as pd

from config import OUT_DIR, WORK_DIR, norm_path
from decide import decide, write_lists

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.70, help="default threshold for every country")
    ap.add_argument("--country", action="append", default=[], help="per-country override, e.g. France=0.85")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "matching_results.tsv")
    a = ap.parse_args()
    over = {k: float(v) for k, v in (c.split("=") for c in a.country)}

    s1 = pd.read_parquet(norm_path("test", 1), columns=["entity_id", "country"])
    scores = pd.read_parquet(WORK_DIR / "test_scores.parquet", columns=["s1_id", "r_id", "p1"])
    thr = s1.set_index("entity_id").country.map(lambda c: over.get(c, a.thr))
    # S2/S3 records never cross countries, so a per-row threshold before 1-1 assignment is exact
    scores = scores[scores.p1 >= scores.s1_id.map(thr).values]
    matches = decide(scores, 0.0, "p1")
    out = write_lists(matches, s1.entity_id.values, a.out, "matched_entity_ids")
    print(f"wrote {a.out}: {len(matches):,} pairs, {(out.matched_entity_ids != '').sum():,} S1 with matches")
    for c, t in sorted(s1.groupby("country").size().items()):
        ids = set(s1.entity_id[s1.country == c])
        print(f"  {c}: thr={over.get(c, a.thr):.2f}  {matches.s1_id.isin(ids).sum() / len(ids):.2f} matches/S1")
