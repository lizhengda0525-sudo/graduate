import numpy as np

from tc_cmlp.analysis.network import change_curve, network_flows, predict_soz


def test_network_flows_follow_target_source_convention() -> None:
    matrices = np.asarray(
        [
            [
                [9.0, 2.0, 0.0],
                [3.0, 9.0, 4.0],
                [5.0, 0.0, 9.0],
            ]
        ]
    )
    outflow, inflow, netflow = network_flows(matrices)

    np.testing.assert_array_equal(outflow, [[8.0, 2.0, 4.0]])
    np.testing.assert_array_equal(inflow, [[2.0, 7.0, 5.0]])
    np.testing.assert_array_equal(netflow, [[6.0, -5.0, -1.0]])


def test_change_curve_ignores_diagonal() -> None:
    matrices = np.zeros((2, 2, 2))
    matrices[1, 0, 0] = 10.0
    matrices[1, 0, 1] = 2.5
    np.testing.assert_array_equal(change_curve(matrices), [2.5])


def test_predict_soz_counts_high_outflow_windows() -> None:
    outflow = np.asarray(
        [
            [0.0, 1.0, 5.0, 2.0],
            [0.0, 1.0, 6.0, 2.0],
            [0.0, 8.0, 1.0, 2.0],
        ]
    )
    prediction, counts, _ = predict_soz(
        outflow=outflow,
        ictal_mask=np.asarray([True, True, False]),
        top_fraction=0.25,
    )

    np.testing.assert_array_equal(counts, [0, 0, 2, 0])
    np.testing.assert_array_equal(prediction, [False, False, True, False])
