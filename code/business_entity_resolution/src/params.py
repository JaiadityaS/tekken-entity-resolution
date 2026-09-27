"""Tunable hyper-parameters, chosen on the training-set validation folds."""

# blocking: top-k per S1 from the sparse IDF product, keys with df > max_df dropped
BLOCK = dict(
    channels={
        "name": ("npkKcCQ", 20),   # name-only keys
        "addr": ("aAhmM", 20),     # address-only keys
        "all": (None, 30),         # every key type incl. name x number crosses
    },
    max_df=5000, min_score=0.02,
    keep_rank=12,                  # union keeps pairs ranked < 12 in some channel
)

# data split: S1 crc32 buckets (of 20) < A_BUCKETS train stage A (the candidate filter)
A_BUCKETS = 3
# final candidate set (= candidate_pairs.tsv) keeps pairs with stage-A p0 >= tau, where
# tau is the largest value retaining TAU_RECALL of the true pairs found by blocking
TAU_RECALL = 0.995
TAU_FLOOR = 0.003  # pairs below this are discarded while streaming (memory bound)
# stage-A scoring uses only its first trees: ~2.5x faster, same for train and test
STAGE_A_ITERS = 500

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, verbose=-1, seed=42,
)
LGB_ROUNDS = 1500
