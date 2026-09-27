# Business Entity Resolution – Amazon ML Challenge 2026

Pipeline: **normalise → multi-channel IDF blocking → learned candidate filter (stage A) →
match model with competition features (stage B) → one-to-one assignment + F0.5 threshold**.

## Environment
* Python 3.13, 16 GB RAM, 12 CPU cores (no GPU needed). The full pipeline runs in about 2.5 hours.
* `pip install -r requirements.txt`. Every dependency is MIT, Apache-2.0 or BSD licensed.
* No external data, APIs or pretrained models are used. The only model is LightGBM (MIT), trained from scratch.

## Data layout
By default the code expects the challenge data at `<project>/student_resource/dataset/{train,test}/`,
where `<project>` is three levels above `src/`. Override with environment variables:
`BER_DATA` (dataset dir), `BER_WORK` (cache/model dir), `BER_OUT` (output dir).

## Reproduce end-to-end
Run from `src/`:

```bash
python prepare.py all            # normalise names/addresses of all 6 files -> work/*_norm.parquet
python make_candidates.py train  # blocking on the training split -> work/train_cands_raw.parquet
python make_candidates.py test   # blocking on the test split     -> work/test_cands_raw.parquet
python train.py                  # stage A + stage B models, tau, threshold -> work/models/
python predict.py                # -> output/candidate_pairs.tsv, output/matching_results.tsv
python ../../../student_resource/utils/validate_submission.py \
    --matching ../../../output/matching_results.tsv \
    --candidate ../../../output/candidate_pairs.tsv \
    --test-dir ../../../student_resource/dataset/test
```

`run_all.py` runs the same steps in order.

## Source files (`src/`)
| file | role |
|---|---|
| `config.py` | paths, parallelism, a batched `imap` helper |
| `normalize.py` | name/address normalisation: Indic-script transliteration, accent folding, legal-form canonicalisation, OCR-digit fixes, domain-name handling, phonetic skeletons, address abbreviations, state codes, number extraction |
| `prepare.py` | stage 1: normalises the raw TSVs in parallel and caches them as parquet |
| `blocking.py` | stage 2: hashed blocking keys, IDF weighting with a document-frequency cap, and three-channel sparse top-k retrieval (`sparse_dot_topn`) per country |
| `make_candidates.py` | driver for stage 2 |
| `features.py` | stage 3: about 45 pairwise similarity features (rapidfuzz) plus group/competition features |
| `common.py` | loaders and the chunked stage-A streaming filter |
| `train.py` | stage 4: trains stage A (candidate filter) and stage B (matcher), then tunes tau and the threshold |
| `decide.py` | one-to-one assignment, threshold search and TSV writer |
| `predict.py` | stage 6: test inference that writes both output files |
| `evaluate.py` | official macro F0.5 and blocking diagnostics |
| `dev_blocking.py` | recall-versus-candidate-size study on a train sample |
| `params.py` | all tunable parameters |
