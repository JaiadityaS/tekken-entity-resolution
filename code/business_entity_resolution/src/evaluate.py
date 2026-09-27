"""Scoring utilities: the official macro F0.5 and blocking diagnostics."""
import numpy as np
import pandas as pd


def load_truth(path):
    """Ground-truth TSV -> DataFrame[s1_id, r_id] (one row per true pair) and list of S1 ids."""
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
    ids = gt.source1_entity_id.values
    pairs = gt.assign(r_id=gt.matched_entity_ids.str.split(",")).explode("r_id")
    pairs = pairs[pairs.r_id.fillna("") != ""]
    return pairs.rename(columns={"source1_entity_id": "s1_id"})[["s1_id", "r_id"]], ids


def macro_f05(pred_pairs, true_pairs, s1_ids, beta=0.5):
    """Macro-averaged F_beta exactly as described by the organisers.

    pred_pairs / true_pairs: DataFrame[s1_id, r_id]; s1_ids: every S1 entity evaluated.
    Empty prediction on an empty truth scores 1; any prediction on an empty truth scores 0.
    """
    s1_ids = pd.Index(pd.unique(np.asarray(s1_ids)))
    pred = pred_pairs[pred_pairs.s1_id.isin(s1_ids)].drop_duplicates()
    true = true_pairs[true_pairs.s1_id.isin(s1_ids)]
    tp = pred.merge(true, on=["s1_id", "r_id"]).groupby("s1_id").size()
    n_pred = pred.groupby("s1_id").size()
    n_true = true.groupby("s1_id").size()
    df = pd.DataFrame(index=s1_ids)
    df["tp"] = tp.reindex(s1_ids).fillna(0).values
    df["np"] = n_pred.reindex(s1_ids).fillna(0).values
    df["nt"] = n_true.reindex(s1_ids).fillna(0).values
    b2 = beta * beta
    p = np.where(df.np > 0, df.tp / df.np.clip(lower=1), 0.0)
    r = np.where(df.nt > 0, df.tp / df.nt.clip(lower=1), 0.0)
    f = np.where(p + r > 0, (1 + b2) * p * r / (b2 * p + r + 1e-12), 0.0)
    f = np.where((df.nt == 0) & (df.np == 0), 1.0, f)
    return float(f.mean()), df.assign(f=f)


def blocking_report(cands, true_pairs, s1_ids, ks=(1, 3, 5, 10, 15, 20, 30, 50)):
    """Pair recall and per-S1 'complete' rate as a function of the rank cut-off."""
    s1_ids = set(s1_ids)
    true = true_pairs[true_pairs.s1_id.isin(s1_ids)]
    c = cands[cands.s1_id.isin(s1_ids)]
    hit = c.merge(true.assign(y=1), on=["s1_id", "r_id"], how="left")
    hit["y"] = hit.y.fillna(0)
    rows = []
    for k in ks:
        h = hit[hit.brank < k]
        rows.append({"k": k, "pair_recall": h.y.sum() / len(true),
                     "cands_per_s1": len(h) / len(s1_ids),
                     "precision": h.y.mean()})
    return pd.DataFrame(rows)
