"""Contract: the L1 retrieval outputs the app reads, and the L2 calls that rerank them. Owned by ml.

QAC: `qac_prefix_completions` in Delta, synced to Lakebase; the app reads every rank for one prefix, keyed and
filtered by the functions below so build and lookup can't drift.
SERP: the AI Search hybrid index over products; the app requests `SERP_INDEX_COLUMNS`.
L2: `qac_l2_scores` / `serp_l2_scores` send L1's candidates to the endpoint and return one score per candidate.
"""

from __future__ import annotations

import re

# Per-completion stats from the train split, carried on every prefix row for L2.
QAC_COMPLETION_FEATURES = ("pop_30d", "pop_7d", "ctr", "category", "category_share", "brand", "brand_share",
                           "median_clicked_price")
QAC_PREFIX_PK = ("prefix", "rank")
QAC_PREFIX_COLUMNS = (*QAC_PREFIX_PK, "completion", "prefix_len", "n_tokens", "is_word_boundary",
                      *QAC_COMPLETION_FEATURES)
# What the QAC L2 endpoint takes per candidate (its RequestSource features): the L1 row, `rank` sent as `l1_rank`.
QAC_L2_CANDIDATE_COLUMNS = ("l1_rank", "prefix_len", "n_tokens", "is_word_boundary", *QAC_COMPLETION_FEATURES)
# Every request field the model reads. The endpoint strips the lookup keys before the model, so being a guest is sent
# as its own field rather than read from `user_id`.
QAC_L2_REQUEST_COLUMNS = (*QAC_L2_CANDIDATE_COLUMNS, "is_guest")

SERP_INDEX_PK = "product_id"
SERP_INDEX_TEXT = "search_text"    # title + brand; embedded and keyword-matched, not returned
SERP_INDEX_COLUMNS = (SERP_INDEX_PK, "title", "brand", "category", "price")
SERP_INDEX_SCORE = "score"         # the hybrid score the index adds to every hit, after the requested columns
# What the SERP L2 endpoint takes per candidate (its RequestSource features): where L1 put it (1-based rank, hybrid
# score) and the index row's fields. `product_id` is sent too, as the item-feature lookup key the endpoint strips.
SERP_L2_CANDIDATE_COLUMNS = ("l1_rank", "l1_score", "price", "brand", "category")
SERP_L2_REQUEST_COLUMNS = (*SERP_L2_CANDIDATE_COLUMNS, "is_guest")
L1_FIELDS = ("l1_rank", "l1_score")  # where L1 put a candidate: set from its position and score, not read off the row


# Prefix keys: the table build normalizes completions with `normalize`; the app looks up `lookup_prefix(typed)`.
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def lookup_prefix(typed: str, max_len: int) -> str:
    """Key for what the user typed: normalized like the completions, but a trailing space is kept."""
    key = re.sub(r"\s+", " ", typed.lower()).lstrip()
    return key[:max_len]


def filter_completions(rows: list[dict], typed: str) -> list[dict]:
    """Input longer than the key: keep completions that start with everything typed."""
    full = re.sub(r"\s+", " ", typed.lower()).lstrip()
    return [r for r in rows if r["completion"].startswith(full)]


def qac_l2_records(rows: list[dict], session_id: str, user_id: str | None) -> list[dict]:
    """QAC L2 endpoint `dataframe_records` for L1 rows: lookup keys plus each row's candidate fields."""
    return [{"session_id": session_id, "user_id": user_id, "l1_rank": r["rank"],
             **{c: r[c] for c in QAC_L2_CANDIDATE_COLUMNS if c not in L1_FIELDS}, "is_guest": int(user_id is None)}
            for r in rows]


def serp_l2_records(hits: list[dict], session_id: str, user_id: str | None) -> list[dict]:
    """SERP L2 endpoint `dataframe_records` for index hits in L1 order (`SERP_INDEX_COLUMNS` + `SERP_INDEX_SCORE`)."""
    return [{"session_id": session_id, "user_id": user_id, SERP_INDEX_PK: h[SERP_INDEX_PK], "l1_rank": i,
             "l1_score": h[SERP_INDEX_SCORE], **{c: h[c] for c in SERP_L2_CANDIDATE_COLUMNS if c not in L1_FIELDS},
             "is_guest": int(user_id is None)} for i, h in enumerate(hits, start=1)]


def l2_scores(w, endpoint: str, records: list[dict]) -> list[float]:
    """Query an L2 endpoint (`w`: a WorkspaceClient) with `records`; one score per record, in record order. Raises
    ValueError if the endpoint returns anything else, so a short or malformed response can't be ranked."""
    if not records:
        return []
    predictions = w.serving_endpoints_data_plane.query(name=endpoint, dataframe_records=records).predictions
    if len(predictions) != len(records):
        raise ValueError(f"{endpoint} returned {len(predictions)} scores for {len(records)} candidates")
    try:
        return [float(p) for p in predictions]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{endpoint} returned non-numeric scores: {predictions[:3]}") from error


def qac_l2_scores(w, endpoint: str, rows: list[dict], session_id: str | None, user_id: str | None) -> list[float]:
    """QAC L2 scores for L1 rows, in row order. The endpoint looks up session and user features by key."""
    return l2_scores(w, endpoint, qac_l2_records(rows, session_id, user_id))


def serp_l2_scores(w, endpoint: str, hits: list[dict], session_id: str | None, user_id: str | None) -> list[float]:
    """SERP L2 scores for index hits, in hit order. The endpoint also looks up item features by `product_id`."""
    return l2_scores(w, endpoint, serp_l2_records(hits, session_id, user_id))
