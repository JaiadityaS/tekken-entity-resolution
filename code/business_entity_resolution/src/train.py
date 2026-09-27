"""Stage 4: train the matcher on the training split.

Data split over train S1 entities (deterministic crc32 buckets):
  * A-set (15%): trains stage A (pairwise-similarity LightGBM) = learned candidate filter
  * B-set (85%): stage A is streamed over it (out-of-sample), pairs with p0 >= tau are kept
    (this is the final candidate set), then stage B (pairwise + group/competition
    features) is trained 2-fold out-of-fold on the B-set, and the decision threshold is
    tuned on the OOF predictions for macro F0.5.

Outputs: work/models/{stageA.txt, stageB_0.txt, stageB_1.txt, meta.json}
Usage: python train.py
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import compute_features, load_sources, s1_bucket, stream_stage_a, train_lgb
from features import add_group_features, index_frame
from config import DATA_DIR, WORK_DIR
from decide import decide, decide_expected, tune_threshold
from evaluate import load_truth, macro_f05
from params import A_BUCKETS, LGB_PARAMS, LGB_ROUNDS, STAGE_A_ITERS, TAU_FLOOR, TAU_RECALL

MODEL_DIR = WORK_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def label(c, true):
    return c.merge(true.assign(y=1), on=["s1_id", "r_id"], how="left").y.fillna(0).values.astype(np.int8)


def main(log=lambda m: print(m, flush=True)):
    t = time.time()
    s1, r = load_sources("train")
    true, s1_ids = load_truth(DATA_DIR / "train" / "train_ground_truth.tsv")
    cands = pd.read_parquet(WORK_DIR / "train_cands_raw.parquet")
    log(f"raw candidates {len(cands):,} ({len(cands) / len(s1_ids):.2f}/S1)")

    b_all = s1_bucket(s1_ids)
    a_ids = set(s1_ids[b_all < A_BUCKETS])
    b_ids = s1_ids[b_all >= A_BUCKETS]
    true_b = true[true.s1_id.isin(set(b_ids))]
    nb = len(b_ids)

    # ---------------- stage A: learned candidate filter ----------------
    path_a = MODEL_DIR / "stageA.txt"
    if path_a.exists():  # resume: stage A is deterministic given the data split
        m_a = lgb.Booster(model_file=str(path_a))
        log("stage A loaded from disk")
    else:
        ca = cands[cands.s1_id.isin(a_ids).values].reset_index(drop=True)
        Fa = compute_features(ca, s1, r, log=log)
        m_a = train_lgb(Fa, label(ca, true), LGB_PARAMS, LGB_ROUNDS, log=log, name="stageA")
        m_a.save_model(str(path_a))
        del Fa, ca
        log(f"stage A trained [{time.time() - t:.0f}s]")
    feats_a = m_a.feature_name()

    # ---------------- stream stage A over all pairs (spilled to disk) ----------------
    # (A-set pairs are kept too so that the competition features see every S1, as at test)
    s1i, ri = index_frame(s1), index_frame(r)
    del s1, r
    kept, n_pos = stream_stage_a(cands, s1i, ri, m_a, feats_a, TAU_FLOOR, WORK_DIR / "train_stageA",
                                 true=true, n_iter=STAGE_A_ITERS, log=log)
    del s1i, ri, cands
    kept["is_b"] = ~kept.s1_id.isin(a_ids).values
    kb = kept[kept.is_b]
    log(f"blocking recall (all S1) {n_pos / len(true):.4f}; after floor {TAU_FLOOR} (B-set): "
        f"{kb.y.sum() / len(true_b):.4f} at {len(kb) / nb:.2f} cands/S1")
    # tau: largest filter threshold that keeps TAU_RECALL of the true B-set pairs left
    pos = np.sort(kb.p0.values[kb.y.values == 1])
    tau = float(pos[int((1 - TAU_RECALL) * len(pos))])
    keep_b = kb.p0.values >= tau
    log(f"tau={tau:.4f}: final candidate recall {kb.y.values[keep_b].sum() / len(true_b):.4f} "
        f"at {keep_b.sum() / nb:.2f} cands/S1")
    del kb
    kept = kept[kept.p0.values >= tau].reset_index(drop=True)
    ids = kept[["s1_id", "r_id", "p0", "y", "is_b", "bscore"]]
    Fb = kept.drop(columns=["s1_id", "r_id", "p0", "y", "is_b"])
    del kept
    kept = ids

    # ---------------- stage B: 2-fold OOF over B-set S1s, group features ----------------
    G = add_group_features(kept, Fb)
    feats_b = list(G.columns)
    y = kept.y.values
    is_b = kept.is_b.values
    fold = s1_bucket(kept.s1_id.values) % 2
    p1 = np.full(len(y), np.nan, dtype=np.float32)
    for k in (0, 1):
        tr = is_b & (fold == k)
        te = is_b & (fold != k)
        m = train_lgb(G[tr], y[tr], LGB_PARAMS, LGB_ROUNDS, log=log, name=f"stageB_{k}")
        p1[te] = m.predict(G[te], num_threads=0)
        m.save_model(str(MODEL_DIR / f"stageB_{k}.txt"))
    kept["p1"] = p1
    kept = kept[is_b].reset_index(drop=True)
    kept[["s1_id", "r_id", "p0", "p1", "y"]].to_parquet(WORK_DIR / "train_oof.parquet", index=False)
    log(f"stage B trained [{time.time() - t:.0f}s]")

    # ---------------- threshold ----------------
    for col in ("p0", "p1"):
        res, thr = tune_threshold(kept, true_b, b_ids, col=col)
        log(f"{col}: best thr {thr:.2f} macro F0.5 {res.f05.max():.5f}")
    res, thr = tune_threshold(kept, true_b, b_ids, col="p1", grid=np.arange(0.3, 0.9, 0.02))
    f_nouniq, _ = macro_f05(decide(kept, thr, "p1", unique=False), true_b, b_ids)
    log(res.to_string(index=False))
    log(f"FINAL OOF macro F0.5 = {res.f05.max():.5f} at thr={thr:.2f} "
        f"(without 1-1 assignment: {f_nouniq:.5f})")
    for a in (1.0, 1.05, 1.1):
        f_exp, _ = macro_f05(decide_expected(kept, "p1", alpha=a), true_b, b_ids)
        log(f"expected-F0.5 decision alpha={a}: {f_exp:.5f}")
    imp = pd.Series(lgb.Booster(model_file=str(MODEL_DIR / "stageB_0.txt")).feature_importance("gain"),
                    index=feats_b).sort_values(ascending=False)
    log("top stage-B features:\n" + (imp / imp.sum()).head(25).round(4).to_string())
    json.dump({"feats_a": feats_a, "feats_b": feats_b, "tau": tau, "thr": thr,
               "stage_a_iters": STAGE_A_ITERS,
               "oof_f05": float(res.f05.max())}, open(MODEL_DIR / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
