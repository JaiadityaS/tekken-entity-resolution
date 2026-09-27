"""Stage 2: candidate generation (blocking).

Each record is turned into a bag of hashed "blocking keys" built from the normalised
name and address:

  name     n: core tokens          p: unordered token pairs     k: phonetic skeletons
           K: sorted skeleton set  c/C: concat-name prefix/suffix  Q: concat skeleton
  address  a: address words        A: adjacent word bigrams     m/M: numbers / number pairs
           h: house-number x word
  cross    x: house-number x name token

Keys are IDF weighted, keys with df > max_df are dropped (no identity signal, and they
dominate the cost), rows are L2 normalised, and a multi-threaded sparse top-k matrix
product (sparse_dot_topn) retrieves the most similar Source-2/3 records for every
Source-1 record.  This is done in several independent *channels* (name-only,
address-only, all keys), each with its own top-k, and the results are unioned - so a
strong name match is not crowded out by records sharing a busy address and vice versa.
Blocking runs independently per country label (open set: whatever S1 contains).

Cost is O(sum over kept keys of df) per S1 record, i.e. linear in the data size.
"""
import gc
import time
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

from config import N_JOBS, batched_imap
from normalize import skeleton

# Reduced from 28 to 24 to cap peak RAM at ~256 MB (2^28 caused a 2 GiB int64 allocation).
# 2^24 = 16 M hash slots; collision rate is negligible for the ~5-15 M distinct keys seen here.
HASH_BITS = 24
HASH_MASK = (1 << HASH_BITS) - 1

BLOCK_COLS = ["n_core", "n_skel", "n_concat", "a_alpha", "a_nums", "a_first_num"]


def record_keys(core, skel, concat, alpha, nums, first_num):
    """Blocking keys for one record; the first character of each key is its type."""
    keys = []
    nt = [t for t in core.split() if len(t) >= 2]
    keys += ["n" + t for t in nt]
    head = nt[:6]
    for i in range(len(head)):
        for j in range(i + 1, len(head)):
            a, b = sorted((head[i], head[j]))
            keys.append("p" + a + "|" + b)
    sk = [s for s in skel.split() if len(s) >= 3]
    keys += ["k" + s for s in sk]
    if skel:
        keys.append("K" + " ".join(sorted(skel.split())))
    if len(concat) >= 6:
        keys.append("c" + concat[:7])
        keys.append("C" + concat[-7:])
        qs = skeleton(concat)
        if len(qs) >= 4:
            keys.append("Q" + qs[:6])
    at = [t for t in alpha.split() if len(t) >= 3]
    keys += ["a" + t for t in at]
    keys += ["A" + at[i] + "|" + at[i + 1] for i in range(len(at) - 1)]
    if first_num:
        keys += ["h" + first_num + "|" + t for t in at[:8]]
        keys += ["x" + first_num + "|" + t for t in nt[:3]]
    ns = [x for x in nums.split() if len(x) >= 2]
    keys += ["m" + x for x in ns]
    for i in range(min(len(ns), 4)):
        for j in range(i + 1, min(len(ns), 4)):
            keys.append("M" + ns[i] + "|" + ns[j])
    return keys


def _hash_chunk(cols):
    """Worker: (lengths int32, hashed keys int32, key types uint8) for a chunk."""
    lens, out, typ = [], [], []
    for rec in zip(*cols):
        ks = record_keys(*rec)
        lens.append(len(ks))
        out.extend(zlib.crc32(k.encode()) & HASH_MASK for k in ks)
        typ.extend(ord(k[0]) for k in ks)
    return (np.asarray(lens, dtype=np.int32), np.asarray(out, dtype=np.int32),
            np.asarray(typ, dtype=np.uint8))


def hash_keys(df, pool, chunk=20000):
    """Hashed blocking keys for every row of df -> (indptr int64, keys int32, types uint8)."""
    cols = [df[c].values for c in BLOCK_COLS]
    jobs = ([c[i:i + chunk].tolist() for c in cols] for i in range(0, len(df), chunk))
    lens, keys, typs = [], [], []
    for l, k, ty in batched_imap(pool, _hash_chunk, jobs):
        lens.append(l)
        keys.append(k)
        typs.append(ty)
    lens = np.concatenate(lens)
    indptr = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=indptr[1:])
    return indptr, np.concatenate(keys), np.concatenate(typs)


def _tfidf(indptr, keys, typs, colmap, idf, types):
    """L2-normalised binary-tf IDF CSR matrix using only keys of the given types."""
    cols = colmap[keys]
    keep = cols >= 0
    if types is not None:
        keep &= np.isin(typs, np.frombuffer(types.encode(), dtype=np.uint8))
    ck = np.zeros(len(keep) + 1, dtype=np.int64)
    np.cumsum(keep, out=ck[1:])
    cols = cols[keep]
    m = sp.csr_matrix((idf[cols], cols, ck[indptr]), shape=(len(indptr) - 1, len(idf)),
                      dtype=np.float32)
    m.sum_duplicates()
    m.data = idf[m.indices]  # duplicates inside a record count once
    norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norm[norm == 0] = 1
    return sp.diags((1 / norm).astype(np.float32)) @ m


def block_country(k1, k2, channels, max_df, min_score, n_threads, keep_rank=None, log=None):
    """Multi-channel top-k retrieval S1 -> R for one country.

    channels: {name: (key types or None for all, top_k)}.
    Returns DataFrame[i (s1 row), j (r row), sc_<name>..., rk_<name>...]; a channel that
    did not retrieve the pair gets score 0 and rank top_k.  With keep_rank, only pairs
    ranked < keep_rank in at least one channel are returned.
    """
    n1, n2 = len(k1[0]) - 1, len(k2[0]) - 1
    df = np.bincount(np.concatenate([k1[1], k2[1]]), minlength=1 << HASH_BITS).astype(np.int32)
    usable = (df >= 2) & (df <= max_df)
    colmap = np.full(1 << HASH_BITS, -1, dtype=np.int32)
    colmap[usable] = np.arange(int(usable.sum()), dtype=np.int32)
    idf = np.log((n1 + n2) / df[usable]).astype(np.float32)
    del df, usable
    keys, chan, score, rank = [], [], [], []
    for ci, (name, (types, top_k)) in enumerate(channels.items()):
        t = time.time()
        A = _tfidf(*k1, colmap, idf, types)
        B = _tfidf(*k2, colmap, idf, types).T.tocsr()
        C = sp_matmul_topn(A, B, top_n=top_k, threshold=min_score, sort=True,
                           n_threads=n_threads)
        del A, B
        cnt = np.diff(C.indptr)
        rows = np.repeat(np.arange(n1, dtype=np.int64), cnt)
        keys.append(rows * n2 + C.indices)
        rank.append((np.arange(C.nnz) - np.repeat(C.indptr[:-1], cnt)).astype(np.int16))
        score.append(C.data.astype(np.float32))
        chan.append(np.full(C.nnz, ci, dtype=np.int8))
        del C, rows
        if log:
            log(f"    channel {name}: {len(keys[-1]):,} pairs {time.time() - t:.0f}s")
    del colmap
    uniq, inv = np.unique(np.concatenate(keys), return_inverse=True)
    del keys
    chan, score, rank = np.concatenate(chan), np.concatenate(score), np.concatenate(rank)
    names = list(channels)
    sc = np.zeros((len(uniq), len(names)), dtype=np.float32)
    rk = np.empty((len(uniq), len(names)), dtype=np.int16)
    rk[:] = np.array([channels[n][1] for n in names], dtype=np.int16)
    sc[inv, chan] = score
    rk[inv, chan] = rank
    del inv, chan, score, rank
    if keep_rank is not None:
        m = rk.min(axis=1) < keep_rank
        uniq, sc, rk = uniq[m], sc[m], rk[m]
    out = pd.DataFrame({"i": (uniq // n2).astype(np.int32), "j": (uniq % n2).astype(np.int32)})
    for ci, n in enumerate(names):
        out[f"sc_{n}"] = sc[:, ci]
        out[f"rk_{n}"] = rk[:, ci]
    return out


def generate_candidates(s1, r, channels, max_df=5000, min_score=0.02, keep_rank=None, log=print):
    """Candidate pairs for all countries.

    s1, r: normalised frames (r = Source 2 and Source 3 concatenated).
    Returns DataFrame[s1_id, r_id, sc_*, rk_*, bscore, brank] where bscore is the best
    channel score and brank the best channel rank.
    """
    parts = []
    with Pool(N_JOBS) as pool:
        for country in sorted(s1.country.unique()):
            t = time.time()
            s1c = s1[s1.country == country]
            rc = r[r.country == country]
            if len(rc) == 0:
                continue
            k1 = hash_keys(s1c, pool)
            k2 = hash_keys(rc, pool)
            th = time.time() - t
            n_keys = len(k1[1]) + len(k2[1])
            p = block_country(k1, k2, channels, max_df, min_score, N_JOBS, keep_rank, log)
            del k1, k2
            p.insert(0, "s1_id", s1c.entity_id.values[p.i.values])
            p.insert(1, "r_id", rc.entity_id.values[p.j.values])
            p = p.drop(columns=["i", "j"])
            parts.append(p)
            log(f"  [{country}] s1={len(s1c):,} r={len(rc):,} pairs={len(p):,} "
                f"({len(p) / max(len(s1c), 1):.1f}/s1) keys={n_keys:,} "
                f"hash {th:.0f}s total {time.time() - t:.0f}s")
            del s1c, rc, p
            gc.collect()
    c = pd.concat(parts, ignore_index=True)
    sc = [x for x in c.columns if x.startswith("sc_")]
    rk = [x for x in c.columns if x.startswith("rk_")]
    c["bscore"] = c[sc].max(axis=1).astype(np.float32)
    c["brank"] = c[rk].min(axis=1).astype(np.int16)
    return c
