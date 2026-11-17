"""DCNv2 tests. Run in their own pytest process: on macOS, torch and LightGBM in one process load two OpenMP
runtimes and segfault (OMP_NUM_THREADS=1 doesn't help), so `testpaths` leaves this folder out."""

import numpy as np
import torch

from rts_ml.serp_l2 import ndcg_per_group, pack_searches
from rts_ml.serp_l2_dcn import DCNv2, fit_dcn, listwise_loss, score
from serp_l2_data import searches


def packed(df, size=12):
    _, x, y, mask = pack_searches(df.rename(columns={"group_id": "search_id"}), n=size)
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def batches(x, y, mask, size=64):
    return lambda: [(x[i:i + size], y[i:i + size], mask[i:i + size]) for i in range(0, len(x), size)]


def test_dcn_scores_every_candidate_and_standardizes_inputs():
    model = DCNv2(n_features=4, mean=np.ones(4), std=np.full(4, 2.0))
    assert model(torch.ones(2, 7, 4)).shape == (2, 7) and model(torch.ones(5, 4)).shape == (5,)
    # Inputs equal to the mean standardize to zero, whatever the scale.
    assert torch.allclose(model.standardize(torch.ones(3, 4)), torch.zeros(3, 4))


def test_listwise_loss_ignores_padding():
    logits = torch.tensor([[2.0, 0.0, 50.0]])
    y, mask = torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[1.0, 1.0, 0.0]])
    expected = -torch.log_softmax(torch.tensor([2.0, 0.0]), 0)[0]
    assert torch.isclose(listwise_loss(logits, y, mask), expected)


def test_dcn_learns_session_intent_and_beats_the_l1_order():
    torch.manual_seed(0)
    x, y, mask = packed(searches(400, seed=0))
    valid = packed(searches(100, seed=1))
    model = fit_dcn(batches(x, y, mask), valid, mean=x.reshape(-1, x.shape[-1]).mean(0).numpy(),
                    std=x.reshape(-1, x.shape[-1]).std(0).numpy(), max_epochs=15)
    test = searches(200, seed=2)
    test["l1"], test["l2"] = -test["l1_rank"], score(model, test)
    assert ndcg_per_group(test, "l2")["ndcg"].mean() > ndcg_per_group(test, "l1")["ndcg"].mean() + 0.3


def test_saved_model_scores_raw_rows_like_training(tmp_path):
    import mlflow

    from rts_ml.serp_l2_model import SerpRanker

    x, y, mask = packed(searches(50, seed=0))
    model = fit_dcn(batches(x, y, mask), packed(searches(20, seed=1)), mean=np.zeros(x.shape[-1]),
                    std=np.ones(x.shape[-1]), max_epochs=2)
    # The endpoint strips the lookup keys before calling the model: it gets features and request fields only.
    test = searches(5, seed=2).drop(columns=["session_id", "user_id", "product_id"])
    mlflow.pyfunc.save_model(tmp_path / "model", python_model=SerpRanker(model),
                             code_paths=["ml/src/rts_ml", "shared/src/rts_shared"])
    loaded = mlflow.pyfunc.load_model(str(tmp_path / "model"))
    assert np.allclose(np.asarray(loaded.predict(test)), score(model, test), atol=1e-6)
