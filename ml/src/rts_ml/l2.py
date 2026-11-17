"""What QAC and SERP L2 share: the candidate x session/user crosses, LambdaRank fitting and the L1-vs-L2 report.

Pandas only (no torch, no Spark), so every task and both logged models import it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LGBM_PARAMS = {"objective": "lambdarank", "num_leaves": 31, "learning_rate": 0.05, "n_estimators": 1000,
               "min_child_samples": 20, "verbose": -1}
EARLY_STOPPING_ROUNDS = 50


def _share(part: pd.Series, *totals: pd.Series) -> pd.Series:
    total = sum(t.fillna(0) for t in totals)
    return (part.fillna(0) / total.where(total > 0)).fillna(0.0)


def _log_gap(price: pd.Series, reference: pd.Series) -> pd.Series:
    """How far the candidate's price sits from a reference price tier; NaN if either is missing."""
    return np.abs(np.log(price.astype(float) / reference.astype(float)))


def add_crosses(df: pd.DataFrame, price: str, gaps: dict[str, str]) -> pd.DataFrame:
    """The candidate x session/user features; returns a copy with the cross columns added. `gaps` maps each price gap
    column to the reference price column it compares the candidate's `price` column with."""
    out = df.copy()
    out["sess_cat_share"] = _share(df["sess_cat_cnt_30m"], df["sess_n_clicks_30m"], df["sess_n_carts_30m"])
    out["user_cat_share"] = _share(df["user_cat_cnt_30d"], df["user_n_clicks_30d"], df["user_n_carts_30d"])
    # Nulls never match: a session with no engaged event, or a candidate with no brand.
    out["cat_is_last"] = (df["category"].notna() & (df["category"] == df["sess_last_category"])).astype(int)
    out["brand_is_last"] = (df["brand"].notna() & (df["brand"] == df["sess_last_brand"])).astype(int)
    for gap, reference in gaps.items():
        out[gap] = _log_gap(df[price], df[reference])
    return out


def _grouped(df: pd.DataFrame, group: str) -> tuple[pd.DataFrame, list[int]]:
    """LightGBM reads each query group as a contiguous block of rows."""
    df = df.sort_values(group, kind="stable")
    return df, df.groupby(group, sort=False).size().tolist()


def fit_ranker(train: pd.DataFrame, valid: pd.DataFrame, matrix, group: str = "group_id", label: str = "label",
               eval_at: int = 5):
    """LambdaRank on raw candidate rows; early-stops on validation NDCG@`eval_at`. `matrix` turns raw rows into
    model inputs."""
    import lightgbm as lgb

    train, train_groups = _grouped(train, group)
    valid, valid_groups = _grouped(valid, group)
    ranker = lgb.LGBMRanker(**LGBM_PARAMS)
    ranker.fit(matrix(train), train[label], group=train_groups,
               eval_set=[(matrix(valid), valid[label])], eval_group=[valid_groups], eval_at=[eval_at],
               callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    return ranker


def bootstrap_ci(values, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Pass per-group L2 - L1 differences for a paired CI on the lift."""
    values = np.asarray(values, dtype=float)
    means = np.random.default_rng(seed).choice(values, size=(n_boot, len(values))).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def slice_report(df: pd.DataFrame, slices: list[str], per_group, metrics: tuple[str, ...], group: str = "group_id",
                 label: str = "label", n_boot: int = 1000, seed: int = 0) -> list[dict]:
    """L1 vs L2 on scored candidate rows (`l1`, `l2` score columns), overall and per value of each slice column
    (constant within a group). One row per (slice, value, metric) with a paired bootstrap CI on the lift.
    `per_group(df, score, group, label)` returns one row per group with the `metrics` columns."""
    per_group = (per_group(df, "l1", group, label)
                 .merge(per_group(df, "l2", group, label), on=group, suffixes=("_l1", "_l2"))
                 .merge(df.groupby(group, as_index=False)[slices].first(), on=group))
    parts = [("all", "all", per_group)] + [(s, v, g) for s in slices for v, g in per_group.groupby(s)]
    report = []
    for name, value, g in parts:
        for metric in metrics:
            l1, l2 = g[f"{metric}_l1"], g[f"{metric}_l2"]
            lo, hi = bootstrap_ci(l2 - l1, n_boot=n_boot, seed=seed)
            report.append({"slice": name, "value": value, "metric": metric, "n": len(g), "l1": l1.mean(),
                           "l2": l2.mean(), "lift": l2.mean() - l1.mean(), "lift_lo": lo, "lift_hi": hi})
    return report
