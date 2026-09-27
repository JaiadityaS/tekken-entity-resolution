"""Stage 1: read raw TSVs, normalise every name/address, cache as parquet.

Usage: python prepare.py [train|test|all]
"""
import sys
import time
from multiprocessing import Pool

import pandas as pd

from config import N_JOBS, batched_imap, norm_path, raw_path
from normalize import core_skeleton, normalize_address, normalize_name


def read_tsv(path, **kw):
    """Read a challenge TSV with every column as string and no NA coercion."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3, **kw)


def _norm_chunk(args):
    names, addrs = args
    n = [normalize_name(x) for x in names]
    a = [normalize_address(x) for x in addrs]
    return n, a


def normalize_frame(df, pool, chunk=20000):
    """Return df with normalised columns appended (n_* from name, a_* from address).

    Flushes accumulated results to DataFrames every FLUSH rows to avoid a
    pyarrow contiguous-memory realloc failure on large source files (e.g. test_source3).
    """
    jobs = ((df.business_name.values[i:i + chunk].tolist(),
             df.business_address.values[i:i + chunk].tolist())
            for i in range(0, len(df), chunk))
    names, addrs = [], []
    nd_parts, ad_parts = [], []
    FLUSH = 400_000  # flush to DataFrame every 400 k rows to keep peak alloc small
    for n, a in batched_imap(pool, _norm_chunk, jobs):
        names.extend(n)
        addrs.extend(a)
        if len(names) >= FLUSH:
            nd_parts.append(pd.DataFrame(names))
            ad_parts.append(pd.DataFrame(addrs))
            names, addrs = [], []
    if names:
        nd_parts.append(pd.DataFrame(names))
        ad_parts.append(pd.DataFrame(addrs))
    nd = pd.concat(nd_parts, ignore_index=True).add_prefix("n_")
    ad = pd.concat(ad_parts, ignore_index=True).add_prefix("a_")
    out = pd.concat([df.reset_index(drop=True), nd, ad], axis=1)
    out["n_is_domain"] = out.n_is_domain.astype("int8")
    out["n_is_indic"] = out.n_is_indic.astype("int8")
    return out


def _skel_chunk(cores):
    return [core_skeleton(c) for c in cores]


def refresh_skeleton(pool):
    """Recompute n_skel from n_core in the cached files (after a skeleton() change)."""
    for split in ("train", "test"):
        for src in (1, 2, 3):
            t = time.time()
            df = pd.read_parquet(norm_path(split, src))
            cores = df.n_core.values
            jobs = (cores[i:i + 50000].tolist() for i in range(0, len(df), 50000))
            df["n_skel"] = [x for part in batched_imap(pool, _skel_chunk, jobs) for x in part]
            df.to_parquet(norm_path(split, src), index=False)
            print(f"{split} s{src}: skeleton refreshed {time.time() - t:.0f}s", flush=True)


def main(splits):
    with Pool(N_JOBS) as pool:
        for split in splits:
            for src in (1, 2, 3):
                if norm_path(split, src).exists():
                    print(f"{split} s{src}: cached", flush=True)
                    continue
                t = time.time()
                df = read_tsv(raw_path(split, src))
                out = normalize_frame(df, pool)
                out.to_parquet(norm_path(split, src), index=False)
                print(f"{split} s{src}: {len(out):,} rows in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "skel":
        with Pool(N_JOBS) as p:
            refresh_skeleton(p)
    else:
        main(["train", "test"] if arg == "all" else [arg])
