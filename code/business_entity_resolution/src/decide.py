"""Stage 5: turn pair probabilities into final match lists.

1. One-to-one constraint: each Source-2/3 record is kept only for the S1 entity that
   gives it the highest probability (the ground truth never links a record to two S1s).
2. Threshold: keep pairs with probability >= thr (tuned for macro F0.5 on validation).
"""
import numpy as np
import pandas as pd


def assign_unique(pairs, col="p1"):
    """Keep, for every r_id, only its best-scoring S1."""
    best = pairs.groupby("r_id", sort=False)[col].transform("max")
    return pairs[pairs[col] >= best].drop_duplicates("r_id")


def decide(pairs, thr, col="p1", unique=True):
    sel = pairs[pairs[col] >= thr]
    if unique:
        sel = assign_unique(sel, col)
    return sel[["s1_id", "r_id"]]


def decide_expected(pairs, col="p1", alpha=1.0, floor=0.05, unique=True):
    """Per-S1 plug-in maximisation of expected F0.5.

    With F0.5 = 1.25*TP / (0.25*n_true + n_pred), for the top-m candidates (by p) of an S1
        E[F](m) ~= 1.25 * sum_{i<=m} p_i / (0.25 * alpha * sum_all p + m)
    (alpha > 1 accounts for true matches the candidate set missed), and the empty
    prediction is worth P(no true match) ~= prod(1 - p_i).  The best option is chosen
    independently for every S1.  Pairs with p < floor are never selected.
    """
    sel = assign_unique(pairs, col) if unique else pairs
    sel = sel[sel[col] >= floor][["s1_id", "r_id", col]].sort_values(
        ["s1_id", col], ascending=[True, False], kind="stable")
    g = sel.groupby("s1_id", sort=False)[col]
    p = sel[col].values.astype(np.float64)
    cum = g.cumsum().values
    tot = g.transform("sum").values
    m = g.cumcount().values + 1
    ef = 1.25 * cum / (0.25 * alpha * tot + m)
    sel = sel.assign(ef=ef, logq=np.log1p(-np.minimum(p, 0.999999)))
    best_m = sel.groupby("s1_id", sort=False).ef.transform("max").values
    p_empty = np.exp(sel.groupby("s1_id", sort=False).logq.transform("sum").values)
    # keep the prefix up to the argmax (ef is evaluated per prefix length m)
    m_best = pd.Series(np.where(ef >= best_m, m, np.iinfo(np.int32).max), index=sel.index)
    first_best = m_best.groupby(sel.s1_id.values, sort=False).transform("min").values
    keep = (m <= first_best) & (best_m > p_empty)
    return sel[keep][["s1_id", "r_id"]]


def tune_threshold(pairs, true_pairs, s1_ids, col="p1", grid=None, unique=True, beta=0.5):
    """Grid-search the probability threshold that maximises macro F0.5.

    Thresholding commutes with the one-to-one assignment (an r keeps its best S1 iff that
    score passes), so the assignment is done once and each threshold is a vectorised pass.
    """
    grid = np.arange(0.2, 0.96, 0.05) if grid is None else grid
    s1_index = pd.Index(pd.unique(np.asarray(s1_ids)))
    true = true_pairs[true_pairs.s1_id.isin(s1_index)]
    sel = assign_unique(pairs, col) if unique else pairs
    sel = sel[sel.s1_id.isin(s1_index)]
    y = sel[["s1_id", "r_id"]].merge(true.assign(y=1), on=["s1_id", "r_id"], how="left").y.fillna(0).values
    code = s1_index.get_indexer(sel.s1_id)
    n_true = np.bincount(s1_index.get_indexer(true.s1_id), minlength=len(s1_index))
    p = sel[col].values
    b2 = beta * beta
    res = []
    for thr in grid:
        m = p >= thr
        n_pred = np.bincount(code[m], minlength=len(s1_index))
        tp = np.bincount(code[m], weights=y[m], minlength=len(s1_index))
        prec = np.divide(tp, n_pred, out=np.zeros(len(tp)), where=n_pred > 0)
        rec = np.divide(tp, n_true, out=np.zeros(len(tp)), where=n_true > 0)
        f = np.divide((1 + b2) * prec * rec, b2 * prec + rec, out=np.zeros(len(tp)),
                      where=(prec + rec) > 0)
        f[(n_true == 0) & (n_pred == 0)] = 1.0
        res.append((float(thr), float(f.mean())))
    res = pd.DataFrame(res, columns=["thr", "f05"])
    return res, float(res.loc[res.f05.idxmax(), "thr"])


def write_lists(pairs, s1_ids, path, col_name):
    """Write the challenge TSV: one row per S1 id, comma-joined ids (empty if none)."""
    lists = pairs.groupby("s1_id").r_id.agg(",".join)
    out = pd.DataFrame({"source1_entity_id": s1_ids})
    out[col_name] = out.source1_entity_id.map(lists).fillna("")
    out.to_csv(path, sep="\t", index=False, lineterminator="\n")
    return out
