# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Tekken
**Team Members:** Rajdeep Singh (Team Leader), Puranjay Rajput, Dhruv Mahajan, Jaiaditya Singh Rawat
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
The pipeline works in stages: normalise, retrieve candidates, filter with a learned model, then match with a second learned model. Every name and address is normalised in a language-independent way. This includes transliterating seven Indic scripts and building a phonetic consonant skeleton, so that for example `praibhet` and `private` both become `prbt`.

A three-channel sparse IDF retrieval returns about 26 candidates per Source-1 entity using linear-time sparse top-k products. A LightGBM pairwise model then acts as a *learned blocking filter*, cutting this to **3.96 candidates per S1** (the `candidate_pairs.tsv` set). A second LightGBM model adds competition features, which exploit the fact that every S2/S3 record belongs to at most one S1 entity. It then makes the final decision with a threshold tuned for macro F0.5.

Out-of-fold macro F0.5 on the training data is **0.9679**.

---

## 2. Methodology

### 2.1 Problem Analysis
Our exploratory data analysis on the training data found:
* **Scale:** 2.2M S1, 5.0M S2 and 5.3M S3 records in train; 1.7M, 4.9M and 5.1M in test.
* **Matches never cross countries.** In a 20k-pair sample, 0% of matches linked records from different countries, so all blocking is done per country label. We treat the label set as open: whatever countries S1 contains, including France, which appears only in the test set.
* **Every S2/S3 id is matched to at most one S1 entity.** All 7.64M matched ids are unique, so the problem is a one-to-many assignment, not general clustering.
* **Group sizes:** 5.6% of S1 entities are singletons and the median group has 3–4 matches (maximum 11). About 26% of S2/S3 records match nothing and act as distractors.
* **Postcodes are almost absent.** ZIP codes appear in about 10% of addresses and Indian PINs in about 0%, so the useful address identifiers are **house/plot numbers plus street and locality words**.
* **Noise we reverse:**
  * Case changes, injected accents (`Cómmittee`) and OCR digit swaps (`Denta1`, `lncorporated`).
  * Junk prefixes (`***`, `>>`, `--`, `(INDIA)`).
  * Legal-form variants (`Pvt`/`Private`, `SARL`, `LLC`).
  * Domain names (`halcyonproductions.com`) and duplicated tokens.
  * `NULL`/`<NULL>`/`N/A` placeholders, leading zeros (`004514`), ordinals (`Fifth`→`5`) and US/India state names versus codes.
  * Component reordering in addresses.
  * About 4% of S2/S3 names and some addresses written in **Devanagari, Tamil, Telugu, Kannada, Malayalam, Bengali and Gujarati** scripts.
  * About 3.4% of S2/S3 records with no address at all.

### 2.2 Solution Strategy
**Approach Type:** Multi-channel IDF blocking, then a learned candidate filter (LightGBM), then a LightGBM matcher with group features, then one-to-one assignment and an F0.5-tuned threshold.

**Core Innovation:**
1. **Phonetic consonant skeletons** that align transliterated Indic text with English. We transliterate with Harvard-Kyoto, map anusvara to *n*, drop the final schwa, merge sound-alike consonants (ph/f, v/w/b, t/d, c/k/ch, sh/s, x/ks) and remove vowels. Examples: `devalapars`=`developers`→`tblprs` and `phaundeshan`=`foundation`→`fntsn`.
2. **Multi-channel retrieval**, so strong name evidence is not crowded out by records sharing a busy address, and vice versa.
3. **Competition features** that encode the at-most-one-S1 constraint: how an S2/S3 record's score for this S1 compares with its best score for any other S1.

---

## 3. Candidate Generation (Blocking)

**Blocking keys.** Each record produces hashed keys, with the key type encoded in the prefix:

| type | key |
|---|---|
| `n` | core name token (legal forms and stop words removed) |
| `p` | unordered pair of name tokens (robust to transposition) |
| `k` / `K` | phonetic skeleton of each token / sorted skeleton set |
| `c` / `C` / `Q` | 7-character prefix / suffix of the concatenated name, and its skeleton (catches domains like `renaudmayesbuckley.com`) |
| `a` / `A` | address word / adjacent address-word bigram |
| `h` | house number × address word |
| `m` / `M` | address number / pair of numbers (Indian `383/17`-style numbering) |
| `x` | house number × name token |

**Scoring.**
* Keys are IDF weighted with binary term frequency, and rows are L2-normalised.
* Keys with document frequency above 5,000 are dropped. They carry almost no identity signal, and removing them bounds the cost at O(Σ df), which is linear in the data.
* Retrieval uses a multi-threaded sparse top-k product (`sparse_dot_topn`) in three **channels**: name keys (top 20), address keys (top 20) and all keys (top 30).
* We keep the union of pairs that rank in the top 12 of at least one channel.

**Learned filter (final candidate set).**
* A LightGBM model on about 50 pairwise features (Section 4) is trained on a disjoint 15% of the training S1 entities.
* It scores every blocked pair, and pairs with p0 ≥ τ are kept.
* τ (= 0.0305) is set on held-out training entities to retain 99.5% of the true pairs that blocking found.
* This filtered set is exactly the input of the final matcher and is written to `candidate_pairs.tsv`.

| stage | train (held-out S1) recall of true pairs | candidates / S1 |
|---|---|---|
| single-channel IDF top-50 (baseline) | 0.951 | 50 |
| 3-channel union, best rank < 12 | 0.957 | 25.7 |
| + learned filter (τ) = `candidate_pairs.tsv` | 0.952 | 3.96 |

Test: 43.7M blocked pairs (25.2/S1), reduced to 7.51M (4.34/S1) in `candidate_pairs.tsv`.
Compared with all-pairs comparison inside a country, the reduction ratio is above 99.999%.

**How we avoided losing true matches:**
* Channels are independent, so a record with an empty address is still found by name keys, and a trade name like `veogild` is found by the address.
* Skeleton keys cover transliteration and typos. Concatenated-name keys cover domains and spacing.
* Number-pair keys cover Indian plot numbering.
* The filter threshold is chosen to keep 99.5% of reachable positives.

---

## 4. Matching Model

**Features used (stage A, pairwise, ~50):**
* **Name:**
  * rapidfuzz token-set, token-sort, ratio and partial ratio on the core name.
  * Jaro-Winkler and partial ratio on the concatenated name (for domains).
  * Token-set ratio on the full name with legal forms, and on the phonetic skeletons.
  * Token Jaccard, containment and intersection count.
  * Token counts and their difference, length ratio and first-token equality.
  * Legal-form presence, equality and Jaccard; domain and Indic-script flags.
* **Address:**
  * Token-set, token-sort and partial-token-set ratios.
  * Word-only token-set ratio, Jaccard and containment.
  * Number Jaccard, containment and intersection.
  * House-number equality, and whether the house number is contained in the other record's numbers.
  * Empty-address flag, number and token counts.
* **Blocking signals:** score and rank in each channel, best score and best rank, and source (S2 or S3).

No feature uses the country label, so France (unseen in training) is handled by the same model.

**Stage B (final model) adds group / competition features** computed over the filtered candidate graph:
* Per S1: number of candidates, p0 rank, max and sum, and gap to the best blocking score.
* Per S2/S3 record: number of competing S1s, rank of this S1 by p0, best and second-best p0 over all S1s, and the gap to the best.

**Model type:** LightGBM binary classifiers (MIT licence, trained from scratch, about 1–2k trees of 127 leaves). No pretrained or external models or data.
**Training protocol:** stage A is trained on 15% of train S1 entities. Stage B is trained 2-fold out-of-fold over the other 85% (split by S1 entity), and both fold models are averaged at test time.
**Decision:** each S2/S3 record is assigned only to its best-scoring S1 (the one-to-one constraint). Then pairs with p1 ≥ θ are kept, where **θ = 0.70** maximises out-of-fold macro F0.5. Singletons fall out naturally when no candidate passes.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, out-of-fold on 1.88M training S1 entities):** 0.9679
  - stage A alone (p0 thresholded): 0.9625 (at p0 ≥ 0.60)
- **Common false positives (wrong merges):** Of the 7.43M out-of-fold candidate pairs, 38.6k are false positives (vs 6.02M true positives, so pair precision is about 99.4%). They fall into three groups:
  * **Sibling companies in the same building:** these share a distinctive first token and the exact address but have a different business word (`Wing Chemicals` vs `Wing Logistics`, `Good Constructions` vs `Good Consulting`, `First Exports` vs `First Producer`).
  * **Near-clone distractors:** the name is almost identical and only the house number differs slightly (`Regional Committee Inc., 23 Lewis Street` vs `28 Lewis St`; `1346 I L Pugh Road` vs `134`). Our number features treat these as OCR noise.
  * **Unrelated trade names at an identical address:** for example `ZEPHWEX` at `005832 Martel Ave`. Here the address carries the whole score.
- **Common false negatives (missed matches):** There are two sources of missed matches:
  * **Pairs lost at blocking and filtering:** about 4.8% of true pairs never reach the matcher (final candidate recall is 0.952).
  * **Pairs scored below θ:** 161.6k true pairs, 2.6% of those in the candidate set. The main patterns are:
    * *Name-only evidence:* the S2/S3 record has an empty address (`Federal Energia` vs `Federal Energia Inc`). The model stays cautious because distractors with near-clone names exist.
    * *Perturbed house numbers:* combined with a changed legal form or suffix (`Hamm Dynamics, 284` vs `Hamm Dynamics LLC, 439`; `Interstate Bancorp, 115` vs `11`).
    * *Names fully in Tamil or Telugu script:* (`திருப்பதி பெஸ்ட் அக்ரோ` for `Tirupati Best Agro`), where skeleton similarity is only partial.
    * *Generic suffix replacement:* the distinctive word is replaced by a generic one (`DL Aldel` → `DL Center`, `Garrity's Landscaping` → `6arrity's Center`).

---

## 6. Conclusion
Careful, language-independent normalisation plus multi-channel retrieval gives a high recall ceiling at about 26 candidates per entity. A learned filter then cuts the set to about 3.96 per entity with little recall loss. The biggest single gain came from modelling the data's at-most-one-S1 structure through competition features. The main remaining errors are transliterations too divergent to align, and DBA names that share only a partial address.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/` (entry point `run_all.py`):
`prepare.py` (normalise) → `make_candidates.py` (blocking) → `train.py` (stage A, τ, stage B, θ) → `predict.py` (both output files).
Supporting modules:
* `normalize.py`: normalisation and transliteration.
* `blocking.py`: keys, IDF and sparse top-k retrieval.
* `features.py`: pairwise and group features.
* `decide.py`: assignment, threshold and writers.
* `evaluate.py`: official metric.
* `params.py`: all settings.

See the README for exact commands. Runtime is about 2.5 hours on a 12-core, 16 GB laptop.

### B. Additional Results
**Threshold sensitivity (out-of-fold macro F0.5 by θ).** The curve is flat between 0.62 and 0.76, so the chosen θ is robust:

| θ | 0.50 | 0.60 | 0.66 | **0.70** | 0.74 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|---|
| F0.5 | 0.9665 | 0.9676 | 0.9679 | **0.9679** | 0.9678 | 0.9674 | 0.9651 |

**Ablations (out-of-fold):**
* Stage A alone: 0.9625.
* Stage B with competition features: 0.9679 (+0.54 points).
* Without the one-to-one assignment: 0.96786.
* An expected-F0.5 per-entity decision rule gave 0.9682. We did not adopt it, because the gain (+0.03 points) is within noise.

**Stage-B feature importance (gain share):**
* p0: 0.79
* best p0 of the record over all S1s: 0.042
* second-best p0: 0.031
* gap to best: 0.023
* rank of this S1: 0.017

**Test run:**
* 1,732,544 S1 entities.
* 43.7M blocked pairs, reduced to 7.51M candidates.
* 5.70M predicted matches (3.29 per S1).
* 1,626,101 S1 entities with at least one match; 106,443 predicted singletons.
* Matches per S1 by country: France 3.55, India 3.17, US 3.34.
* The official `validate_submission.py` reports PASS for both output files.
