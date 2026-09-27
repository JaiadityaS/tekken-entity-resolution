"""Stage 3: pairwise features for (Source-1 record, candidate record) pairs.

All string similarities use rapidfuzz.process.cpdist, which is vectorised over the
aligned pair lists and multi-threaded, so tens of millions of pairs are feasible.
No feature depends on the country label, so unseen countries (France) are handled
by the same model.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

NORM_COLS = ["entity_id", "n_name", "n_core", "n_skel", "n_legal", "n_concat", "n_is_domain",
             "n_is_indic", "a_addr", "a_alpha", "a_nums", "a_first_num"]


def index_frame(df):
    """Keep only the columns used for features, indexed by entity_id."""
    return df[NORM_COLS].set_index("entity_id")


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=1, dtype=np.float32, **kw)


def _set_stats(a, b):
    """Jaccard / containment for space-separated token strings (row-wise)."""
    jac = np.zeros(len(a), np.float32)
    cont = np.zeros(len(a), np.float32)
    inter_n = np.zeros(len(a), np.int16)
    for i, (x, y) in enumerate(zip(a, b)):
        sx, sy = set(x.split()), set(y.split())
        if sx and sy:
            it = len(sx & sy)
            jac[i] = it / len(sx | sy)
            cont[i] = it / min(len(sx), len(sy))
            inter_n[i] = it
        else:
            jac[i] = cont[i] = -1
    return jac, cont, inter_n


def pair_features(pairs, s1, r, pool=None, chunk=100_000):
    """pairs: DataFrame[s1_id, r_id, bscore, brank]; s1/r: normalised frames indexed by
    entity_id (see index_frame).

    Returns a DataFrame of numeric features aligned with `pairs` (row order kept).
    With a multiprocessing pool the work is split into chunks across processes.
    """
    A = s1.loc[pairs.s1_id.values].reset_index(drop=True)
    B = r.loc[pairs.r_id.values].reset_index(drop=True)
    P = pairs[["r_id"] + [c for c in pairs.columns if c[:3] in ("sc_", "rk_")]
              + ["bscore", "brank"]].reset_index(drop=True)
    jobs = [(P.iloc[i:i + chunk], A.iloc[i:i + chunk], B.iloc[i:i + chunk])
            for i in range(0, len(P), chunk)]
    parts = pool.map(_pair_features_chunk, jobs) if pool else [_pair_features_chunk(j) for j in jobs]
    return pd.concat(parts, ignore_index=True)


def _pair_features_chunk(job):
    """Features for one aligned chunk (pairs P, left records A, right records B)."""
    pairs, A, B = job
    A = A.reset_index(drop=True)
    B = B.reset_index(drop=True)
    F = pd.DataFrame(index=pd.RangeIndex(len(pairs)))
    for c in pairs.columns:
        if c[:3] in ("sc_", "rk_") or c in ("bscore", "brank"):
            F[c] = pairs[c].values
    F["is_s3"] = pairs.r_id.str.startswith("S3").values.astype(np.int8)

    # ---- name ----
    ac, bc = A.n_core.tolist(), B.n_core.tolist()
    F["nm_tset"] = _cp(ac, bc, fuzz.token_set_ratio)
    F["nm_tsort"] = _cp(ac, bc, fuzz.token_sort_ratio)
    F["nm_ratio"] = _cp(ac, bc, fuzz.ratio)
    F["nm_partial"] = _cp(ac, bc, fuzz.partial_ratio)
    acat, bcat = A.n_concat.tolist(), B.n_concat.tolist()
    F["nm_cat_jw"] = _cp(acat, bcat, JaroWinkler.normalized_similarity)
    F["nm_cat_partial"] = _cp(acat, bcat, fuzz.partial_ratio)
    F["nm_full_tset"] = _cp(A.n_name.tolist(), B.n_name.tolist(), fuzz.token_set_ratio)
    F["nm_skel_tset"] = _cp(A.n_skel.tolist(), B.n_skel.tolist(), fuzz.token_set_ratio)
    F["nm_jac"], F["nm_cont"], F["nm_inter"] = _set_stats(ac, bc)  # lists: fast iteration
    na = A.n_core.str.count(" ").values + 1
    nb = B.n_core.str.count(" ").values + 1
    F["nm_ntok_a"], F["nm_ntok_b"] = na, nb
    F["nm_ntok_diff"] = nb - na
    F["nm_len_ratio"] = (B.n_concat.str.len().values + 1) / (A.n_concat.str.len().values + 1)
    F["nm_first_tok_eq"] = (A.n_core.str.split(" ").str[0].values
                            == B.n_core.str.split(" ").str[0].values).astype(np.int8)
    la, lb = A.n_legal.to_numpy(dtype=object), B.n_legal.to_numpy(dtype=object)
    F["lg_a"] = (la != "").astype(np.int8)
    F["lg_b"] = (lb != "").astype(np.int8)
    F["lg_eq"] = (la == lb).astype(np.int8)
    lj, _, _ = _set_stats(la.tolist(), lb.tolist())
    F["lg_jac"] = lj
    F["b_domain"] = B.n_is_domain.values
    F["b_indic"] = B.n_is_indic.values

    # ---- address ----
    aa, ba = A.a_addr.tolist(), B.a_addr.tolist()
    F["ad_b_empty"] = (B.a_addr.values == "").astype(np.int8)
    F["ad_tset"] = _cp(aa, ba, fuzz.token_set_ratio)
    F["ad_tsort"] = _cp(aa, ba, fuzz.token_sort_ratio)
    F["ad_partial"] = _cp(aa, ba, fuzz.partial_token_set_ratio)
    F["ad_alpha_tset"] = _cp(A.a_alpha.tolist(), B.a_alpha.tolist(), fuzz.token_set_ratio)
    F["ad_alpha_jac"], F["ad_alpha_cont"], _ = _set_stats(A.a_alpha.tolist(), B.a_alpha.tolist())
    anums, bnums = A.a_nums.tolist(), B.a_nums.tolist()
    F["ad_num_jac"], F["ad_num_cont"], F["ad_num_inter"] = _set_stats(anums, bnums)
    fa, fb = A.a_first_num.to_numpy(dtype=object), B.a_first_num.to_numpy(dtype=object)
    F["ad_fnum_eq"] = np.where((fa == "") | (fb == ""), -1, (fa == fb).astype(np.int8))
    fb_in_a = [(y in x.split()) if y else False for x, y in zip(anums, fb.tolist())]
    F["ad_fnum_in"] = np.where(fb == "", -1, np.asarray(fb_in_a, np.int8))
    F["ad_nnum_b"] = B.a_nums.str.count(" ").values + (B.a_nums.values != "")
    F["ad_ntok_b"] = B.a_addr.str.count(" ").values + (B.a_addr.values != "")
    return F


def add_group_features(pairs, F):
    """Context features across the candidate list of each S1 and each candidate record.

    Captures the one-to-one structure of the data: every S2/S3 record belongs to at most
    one S1 entity, so a candidate that scores better against another S1 is suspicious.
    `pairs` must carry a first-pass score column `p0`.
    """
    g1 = pairs.groupby("s1_id", sort=False)
    F["s1_ncand"] = g1.r_id.transform("size").values
    F["s1_max_b"] = g1.bscore.transform("max").values
    F["b_gap"] = F.s1_max_b - F.bscore
    F["p0"] = pairs.p0.values
    F["p0_rank_s1"] = g1.p0.rank(ascending=False, method="first").values
    F["p0_max_s1"] = g1.p0.transform("max").values
    F["p0_sum_s1"] = g1.p0.transform("sum").values
    gr = pairs.groupby("r_id", sort=False)
    F["r_ncand"] = gr.s1_id.transform("size").values
    F["p0_rank_r"] = gr.p0.rank(ascending=False, method="first").values
    F["p0_max_r"] = gr.p0.transform("max").values
    F["p0_gap_r"] = F.p0_max_r - F.p0
    # second best competitor for this r (0 if none)
    order = pairs[["r_id", "p0"]].sort_values(["r_id", "p0"], ascending=[True, False])
    second = order[order.groupby("r_id", sort=False).cumcount() == 1].set_index("r_id").p0
    F["p0_second_r"] = pairs.r_id.map(second).fillna(0).values
    return F
