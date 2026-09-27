# Team Tekken: plan for 27 Sep (deadline 23:59 IST)

Current leaderboard (LB) score is **0.956**; out-of-fold (validation) macro F0.5 is 0.9687. Likely cause of the gap: France is 15% of test but absent from training, and it is over-matched (3.55 matches/S1 vs 3.2–3.3 elsewhere). Realistic target tonight: **0.960–0.965**.

## Setup (everyone)
```bash
git clone https://github.com/JaiadityaS/tekken-entity-resolution.git
cd tekken-entity-resolution
pip install -r code/business_entity_resolution/requirements.txt
```
Unzip the challenge data so that `student_resource/dataset/{train,test}/` sits in the repo root (it is git-ignored).
The trained models are already in `work/models/`. **Do not retrain.**

## Assignments
| who | branch | task | deadline |
|---|---|---|---|
| Jaiaditya | `main` | France-threshold variants via `rethreshold.py` (needs `work/test_scores.parquet`, only on Jaiaditya's machine) | now |
| Puranjay | `fr-names` | French clean-up **B**: address + name rules, France rows only | code 18:30, run done 21:00 |
| Dhruv | `fr-address` | French clean-up **A**: address rules only, France rows only | run done 20:30 |
| Rajdeep | – | owns portal uploads + submission log, trims doc to 1–2 pages, final pick at 22:30, final zip by 23:30 | 23:30 |

### French rules (apply only where `country == "France"`; India/US output must stay byte-identical)
- Address:
  - Map departments to regions (Gironde→Nouvelle-Aquitaine, Nord→Hauts-de-France, Loire-Atlantique→Pays de la Loire, and so on).
  - Street types: `r.`, `av`, `bd`, `imp`, `pl`, `all`.
  - Attach `bis`/`ter` to the house number.
  - Split elisions (`lOrne`, `l'Orne` → `orne`).
  - Drop `de/du/des/la/le/les`.
- Name (Puranjay only):
  - The same stop words and elisions.
  - Strip `Sté`/`Société`.
  - Fold `Ç` and similar characters.

### Test-only rerun (about 1.5–2 h; no retraining needed because France was never in training)
From `code/business_entity_resolution/src/`:
```bash
python prepare.py test
python make_candidates.py test
python predict.py
python ../../../student_resource/utils/validate_submission.py --matching ../../../output/matching_results.tsv --candidate ../../../output/candidate_pairs.tsv --test-dir ../../../student_resource/dataset/test
```
Send `output/matching_results.tsv` to Rajdeep **only if the validator prints PASS**. Also send `work/test_scores.parquet` to Jaiaditya so the best France threshold can be applied on top.

### Re-threshold (no rerun)
```bash
python rethreshold.py --country France=0.90 --out ../../../output/sub_fr090.tsv
```

## Rules for coding agents
- No geocoding, external APIs, internet data or pretrained models (banned by the challenge).
- Do not touch `work/models/`, and do not retrain.
- Only France-specific changes.
- Commit to your own branch. Never push large data or output files; `.gitignore` already blocks them.
