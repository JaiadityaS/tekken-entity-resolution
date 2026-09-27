"""Paths and global settings. Override the roots with env vars BER_DATA / BER_WORK / BER_OUT."""
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
PROJECT = _HERE.parents[2]  # .../amazonMLProj

DATA_DIR = Path(os.environ.get("BER_DATA", PROJECT / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK", PROJECT / "work"))
OUT_DIR = Path(os.environ.get("BER_OUT", PROJECT / "output"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_JOBS = int(os.environ.get("BER_JOBS", os.cpu_count() or 4))
SEED = 42


def raw_path(split, src):
    """split in {'train','test'}, src in {1,2,3}."""
    return DATA_DIR / split / f"{split}_source{src}.tsv"


def norm_path(split, src):
    return WORK_DIR / f"{split}_s{src}_norm.parquet"


def batched_imap(pool, func, jobs, batch=None):
    """pool.imap over a lazy job generator, submitting at most `batch` jobs at a time so
    that the pickled inputs of a huge file are never all materialised at once."""
    import itertools
    batch = batch or 2 * N_JOBS
    jobs = iter(jobs)
    while True:
        chunk = list(itertools.islice(jobs, batch))
        if not chunk:
            return
        yield from pool.imap(func, chunk)
