import numpy as np

from rehearsal.models.order import learn_olem_order_indices


def test_kozachenko_order_estimator_runs_without_torch_or_scipy():
    matrix = np.array(
        [
            [0.0, 0.1, 0.2],
            [1.0, 1.1, 0.9],
            [2.0, 1.9, 2.2],
            [3.0, 3.2, 2.8],
        ],
        dtype=float,
    )

    order, scores = learn_olem_order_indices(matrix, entropy_estimator="kozachenko")

    assert sorted(order) == [0, 1, 2]
    assert set(scores) == {0, 1, 2}
    assert all(np.isfinite(value) for value in scores.values())
