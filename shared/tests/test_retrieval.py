from types import SimpleNamespace

import pytest

from rts_shared.retrieval import (QAC_COMPLETION_FEATURES, QAC_L2_CANDIDATE_COLUMNS, QAC_L2_REQUEST_COLUMNS,
                                  QAC_PREFIX_COLUMNS, QAC_PREFIX_PK, SERP_INDEX_COLUMNS,
                                  SERP_INDEX_PK, SERP_INDEX_TEXT, SERP_L2_CANDIDATE_COLUMNS, SERP_L2_REQUEST_COLUMNS,
                                  SERP_INDEX_SCORE, filter_completions, l2_scores, lookup_prefix, normalize,
                                  qac_l2_records, qac_l2_scores, serp_l2_records, serp_l2_scores)


def test_qac_prefix_table_columns():
    assert QAC_PREFIX_PK == ("prefix", "rank")
    assert QAC_PREFIX_COLUMNS[:2] == QAC_PREFIX_PK
    assert len(QAC_PREFIX_COLUMNS) == len(set(QAC_PREFIX_COLUMNS))
    # L2 reads the completion features straight off the L1 rows, so they are part of the table.
    assert set(QAC_COMPLETION_FEATURES) <= set(QAC_PREFIX_COLUMNS)
    # L2 matches the candidate's category and brand against the session's.
    assert {"category", "category_share", "brand", "brand_share"} <= set(QAC_COMPLETION_FEATURES)
    assert {"completion", "prefix_len", "n_tokens", "is_word_boundary"} <= set(QAC_PREFIX_COLUMNS)


def test_qac_l2_candidate_columns_come_from_the_prefix_row():
    # The app sends each L1 row's fields to the L2 endpoint; `rank` is renamed `l1_rank`, the rest pass as is.
    assert QAC_L2_CANDIDATE_COLUMNS[0] == "l1_rank"
    assert set(QAC_L2_CANDIDATE_COLUMNS[1:]) <= set(QAC_PREFIX_COLUMNS) - {"prefix", "rank", "completion"}
    assert {"category", "brand", *QAC_COMPLETION_FEATURES} <= set(QAC_L2_CANDIDATE_COLUMNS)


def test_serp_index_columns():
    assert SERP_INDEX_COLUMNS[0] == SERP_INDEX_PK == "product_id"
    assert len(SERP_INDEX_COLUMNS) == len(set(SERP_INDEX_COLUMNS))
    # The app shows title/brand/price and L2 keys category features on category; the embedded text isn't returned.
    assert {"title", "brand", "category", "price"} <= set(SERP_INDEX_COLUMNS)
    assert SERP_INDEX_TEXT not in SERP_INDEX_COLUMNS


def test_normalize():
    assert normalize("  Camera   BAGS ") == "camera bags"
    assert normalize("Camera\tBag") == "camera bag"
    assert normalize("") == ""


def test_lookup_prefix_normalizes_and_truncates():
    assert lookup_prefix("  CAM", max_len=12) == "cam"
    assert lookup_prefix("camera bag for sony", max_len=12) == "camera bag f"
    assert lookup_prefix("", max_len=12) == ""
    assert lookup_prefix("camera ", max_len=12) == "camera "  # a typed trailing space is kept


def test_filter_completions_for_input_longer_than_the_key():
    rows = [{"completion": "camera bag for sony"}, {"completion": "camera bag for canon"}]
    assert filter_completions(rows, "Camera bag for s") == [{"completion": "camera bag for sony"}]
    assert filter_completions(rows, "cam") == rows


def test_qac_l2_records_send_keys_and_each_l1_rows_candidate_fields():
    row = {"prefix": "ca", "rank": 2, "completion": "camera bag", "prefix_len": 2, "n_tokens": 2,
           "is_word_boundary": False, "pop_30d": 30, "pop_7d": 4, "ctr": 0.5, "category": "Camera & Photo",
           "category_share": 0.9, "brand": "canon", "brand_share": 0.4, "median_clicked_price": 40.0}
    records = qac_l2_records([row], session_id="s1", user_id=None)
    # The endpoint looks up session and user features by key and strips the keys before the model, so whether the
    # shopper is logged in travels as its own request field.
    assert records == [{"session_id": "s1", "user_id": None, "l1_rank": 2,
                        **{c: row[c] for c in QAC_L2_CANDIDATE_COLUMNS[1:]}, "is_guest": 1}]
    assert qac_l2_records([row], session_id="s1", user_id="u1")[0]["is_guest"] == 0
    assert set(records[0]) == {"session_id", "user_id", *QAC_L2_REQUEST_COLUMNS}


def test_serp_l2_candidate_columns_come_from_the_index_row():
    # The index row's fields, plus where L1 put it; product_id is a lookup key (item features), not a request field.
    assert SERP_L2_CANDIDATE_COLUMNS[:2] == ("l1_rank", "l1_score")
    assert set(SERP_L2_CANDIDATE_COLUMNS[2:]) <= set(SERP_INDEX_COLUMNS)
    assert SERP_INDEX_PK not in SERP_L2_REQUEST_COLUMNS
    assert SERP_L2_REQUEST_COLUMNS == (*SERP_L2_CANDIDATE_COLUMNS, "is_guest")


def test_serp_l2_records_send_keys_and_each_hits_candidate_fields():
    hits = [{"product_id": "B01", "title": "Sony bag", "brand": "sony", "category": "Camera & Photo", "price": 40.0,
             "score": 0.9},
            {"product_id": "B02", "title": "Bag", "brand": None, "category": "Camera & Photo", "price": 12.0,
             "score": 0.7}]
    records = serp_l2_records(hits, session_id="s1", user_id=None)
    # Hits arrive in L1 order; rank is 1-based like the QAC table. The endpoint strips the keys before the model.
    assert records[0] == {"session_id": "s1", "user_id": None, "product_id": "B01", "l1_rank": 1, "l1_score": 0.9,
                          "price": 40.0, "brand": "sony", "category": "Camera & Photo", "is_guest": 1}
    assert records[1]["l1_rank"] == 2 and records[1]["brand"] is None
    assert serp_l2_records(hits, session_id="s1", user_id="u1")[0]["is_guest"] == 0
    assert set(records[0]) == {"session_id", "user_id", SERP_INDEX_PK, *SERP_L2_REQUEST_COLUMNS}


class FakeEndpoints:
    """`w` for the L2 calls: records what each endpoint was sent and answers with `predictions`."""
    def __init__(self, predictions=None):
        self.predictions, self.calls = predictions, []
        self.serving_endpoints_data_plane = self

    def query(self, name, dataframe_records):
        self.calls.append((name, dataframe_records))
        preds = self.predictions if self.predictions is not None else [0.5] * len(dataframe_records)
        return SimpleNamespace(predictions=preds)


QAC_ROW = {"rank": 1, "prefix_len": 2, "n_tokens": 1, "is_word_boundary": True, "pop_30d": 5, "pop_7d": 1,
           "ctr": 0.4, "category": "Toys", "category_share": 0.8, "brand": "lego", "brand_share": 0.5,
           "median_clicked_price": 20.0}
SERP_HIT = {"product_id": "B01", "title": "Bag", "brand": "sony", "category": "Camera & Photo", "price": 40.0,
            SERP_INDEX_SCORE: 0.9}


def test_l2_scores_send_the_records_and_return_one_float_per_record():
    w = FakeEndpoints([1, 0.25])
    assert l2_scores(w, "rts-qac-l2", [{"a": 1}, {"a": 2}]) == [1.0, 0.25]
    assert w.calls == [("rts-qac-l2", [{"a": 1}, {"a": 2}])]


@pytest.mark.parametrize("predictions", [[0.1], [0.1, 0.2, 0.3], [{"score": 0.1}, {"score": 0.2}]])
def test_l2_scores_reject_a_response_that_is_not_one_number_per_record(predictions):
    with pytest.raises(ValueError):
        l2_scores(FakeEndpoints(predictions), "ep", [{"a": 1}, {"a": 2}])


def test_l2_scores_skip_the_call_without_records():
    w = FakeEndpoints()
    assert l2_scores(w, "ep", []) == [] and w.calls == []


def test_qac_and_serp_l2_scores_send_their_records():
    w = FakeEndpoints()
    assert qac_l2_scores(w, "qac", [QAC_ROW, QAC_ROW], "s1", None) == [0.5, 0.5]
    assert serp_l2_scores(w, "serp", [SERP_HIT], "s1", "u1") == [0.5]
    assert w.calls == [("qac", qac_l2_records([QAC_ROW, QAC_ROW], "s1", None)),
                       ("serp", serp_l2_records([SERP_HIT], "s1", "u1"))]
