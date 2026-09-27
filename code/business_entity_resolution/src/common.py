"""Shared loaders and the chunked feature/stage-A streaming used by train and predict."""
import time
import zlib
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import N_JOBS, norm_path
from features import NORM_COLS, index_frame, pair_features


def load_sources(split):
    """Normalised Source 1 and the concatenation of Source 2 + Source 3 (needed columns)."""
    cols = NORM_COLS + ["country"]
    s1 = pd.read_parquet(norm_path(split, 1), columns=cols)
    r = pd.concat([pd.read_parquet(norm_path(split, 2), columns=cols),
                   pd.read_parquet(norm_path(split, 3), columns=cols)], ignore_index=True)
    return s1, r


def s1_bucket(ids, n=20):
    """Deterministic bucket in [0, n) for each S1 id (used for data splits)."""
    return np.fromiter((zlib.crc32(x.encode()) % n for x in ids), dtype=np.int16, count=len(ids))


def compute_features(cands, s1, r, chunk=1_000_000, log=print):
    """Pairwise features for all pairs of `cands` (float32 DataFrame, same row order)."""
    s1i, ri = index_frame(s1), index_frame(r)
    parts, t = [], time.time()
    with Pool(N_JOBS) as pool:
        for i in range(0, len(cands), chunk):
            parts.append(pair_features(cands.iloc[i:i + chunk], s1i, ri, pool=pool).astype(np.float32))
            log(f"  features {min(i + chunk, len(cands)):,}/{len(cands):,} {time.time() - t:.0f}s")
    return pd.concat(parts, ignore_index=True)


def stream_stage_a(cands, s1i, ri, booster, feats, tau, out_dir, true=None, n_iter=None,
                   chunk=1_000_000, log=print):
    """Stage A as a learned candidate filter, streamed in chunks to bound memory.

    For each chunk: pairwise features -> p0 (stage-A model, first n_iter trees) -> keep
    pairs with p0 >= tau.  Kept rows (ids, p0, optional label y, all features) are
    written to out_dir/part_*.parquet and read back at the end.
    s1i / ri are the source frames indexed by entity_id (features.index_frame).
    Returns (DataFrame of kept rows, number of true pairs among ALL streamed pairs).
    """
    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    t, n_kept, n_pos = time.time(), 0, 0
    with Pool(N_JOBS) as pool:
        for k, i in enumerate(range(0, len(cands), chunk)):
            c = cands.iloc[i:i + chunk]
            F = pair_features(c, s1i, ri, pool=pool).astype(np.float32)
            p0 = booster.predict(F[feats], num_iteration=n_iter, num_threads=N_JOBS)
            keep = p0 >= tau
            K = F[keep].reset_index(drop=True)
            K.insert(0, "s1_id", c.s1_id.values[keep])
            K.insert(1, "r_id", c.r_id.values[keep])
            K.insert(2, "p0", p0[keep].astype(np.float32))
            if true is not None:
                y = c[["s1_id", "r_id"]].merge(true.assign(y=1), on=["s1_id", "r_id"],
                                               how="left").y.fillna(0).values.astype(np.int8)
                n_pos += int(y.sum())
                K.insert(3, "y", y[keep])
            K.to_parquet(out_dir / f"part_{k:04d}.parquet", index=False)
            n_kept += len(K)
            log(f"  stage A {min(i + chunk, len(cands)):,}/{len(cands):,} kept {n_kept:,} "
                f"{time.time() - t:.0f}s")
            del c, F, K
    return pd.read_parquet(out_dir), n_pos


def train_lgb(X, y, params, rounds, valid_frac=0.1, seed=42, log=print, name=""):
    """Train LightGBM with a random validation slice for early stopping."""
    rng = np.random.default_rng(seed)
    va = rng.random(len(y)) < valid_frac
    dtr = lgb.Dataset(X[~va], y[~va], free_raw_data=True)
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    m = lgb.train(params, dtr, rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    log(f"  {name}: {m.best_iteration} rounds, valid logloss "
        f"{m.best_score['valid_0']['binary_logloss']:.5f}")
    return m
