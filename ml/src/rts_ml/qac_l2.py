"""QAC L2: session-aware LightGBM reranker over the L1 completions.

Every input is a Feature View feature (session from the Kafka stream, user from Delta, candidate fields from the
request), so training and serving read the same values. The crosses that match a candidate to the session are
computed here, inside the logged model, so they can't drift between the two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rts_ml import l2
from rts_shared.retrieval import QAC_L2_REQUEST_COLUMNS

# Session and user features as the Feature Views name them (`rts_shared.features`, 30-minute / 30-day windows).
SESSION_NUMERIC = ("sess_n_clicks_30m", "sess_n_carts_30m", "sess_n_searches_30m", "sess_avg_price_30m",
                   "sess_min_price_30m", "sess_max_price_30m", "sess_std_price_30m", "sess_last_price")
USER_NUMERIC = ("user_median_price_30d", "user_n_clicks_30d", "user_n_carts_30d")
GAPS = {"price_gap_session": "sess_avg_price_30m", "price_gap_user": "user_median_price_30d"}
CROSSES = ("sess_cat_share", "cat_is_last", "brand_is_last", "user_cat_share", *GAPS)
IDENTITIES = ("category", "brand")  # matched against the session through the crosses, never scored directly
# Left out: `sess_last_event_type`, always the current search in training but the click or cart before at serving.
MODEL_COLUMNS = (*SESSION_NUMERIC, *USER_NUMERIC,
                 *(c for c in QAC_L2_REQUEST_COLUMNS if c not in IDENTITIES), *CROSSES)


def add_crosses(df: pd.DataFrame) -> pd.DataFrame:
    """The candidate x session/user features LightGBM scores on; returns a copy with the cross columns added. The price
    gaps compare the completion's typical clicked price with the session's and the user's."""
    return l2.add_crosses(df, "median_clicked_price", GAPS)


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Raw candidate rows (features + request fields) -> the float matrix LightGBM scores, in MODEL_COLUMNS order."""
    return add_crosses(df)[list(MODEL_COLUMNS)].astype(float)


def rank_metrics(df: pd.DataFrame, score: str, group: str = "group_id", label: str = "label",
                 k: int = 5) -> pd.DataFrame:
    """Per group: reciprocal rank of the positive within the top k (`mrr`), and whether it ranks first (`success1`).
    Groups whose candidates hold no positive score 0, so L1 misses count against both L1 and L2."""
    ranks = df.groupby(group)[score].rank(ascending=False, method="first")
    hit = df[label] == 1
    rr = (1.0 / ranks).where(hit & (ranks <= k), 0.0)
    out = pd.DataFrame({group: df[group], "mrr": rr, "success1": (hit & (ranks == 1)).astype(float)})
    return out.groupby(group, as_index=False).max()


def score(ranker, df: pd.DataFrame) -> np.ndarray:
    """L2 scores for raw candidate rows, in row order; higher ranks first."""
    return ranker.predict(model_matrix(df))


def fit_valid_split(train: pd.DataFrame, valid_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The train split's candidate rows -> (fit, valid): the last `valid_days` early-stop the ranker. Groups whose
    candidates hold no positive (L1 missed the submitted query) are left out: they teach nothing."""
    cutoff = train["event_time"].max() - pd.Timedelta(days=valid_days)
    train = train[train.groupby("group_id")["label"].transform("max") == 1]
    return train[train["event_time"] < cutoff], train[train["event_time"] >= cutoff]
