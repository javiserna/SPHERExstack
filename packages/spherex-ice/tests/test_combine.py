import numpy as np

from spherex_ice import combine as combine_mod


def test_combine_epochs_median_rejects_a_minority_outlier():
    # Three epochs' tau at the same pixel: two agree (~0.3), one is a
    # one-off artifact (3.0) -- the median should follow the majority.
    r1 = {"tau": np.array([[0.30]])}
    r2 = {"tau": np.array([[0.32]])}
    r3 = {"tau": np.array([[3.00]])}

    combined = combine_mod.combine_epochs_median([r1, r2, r3])

    assert combined["n_epochs"][0, 0] == 3
    assert combined["coverage"] == 1.0
    assert abs(combined["tau"][0, 0] - 0.32) < 1e-6  # median of [0.30, 0.32, 3.00] is the middle value


def test_combine_epochs_median_handles_partial_coverage():
    # Pixel (0,0): covered by both epochs. Pixel (0,1): only epoch 2.
    r1 = {"tau": np.array([[1.0, np.nan]])}
    r2 = {"tau": np.array([[3.0, 5.0]])}

    combined = combine_mod.combine_epochs_median([r1, r2])

    assert combined["n_epochs"].tolist() == [[2, 1]]
    assert combined["tau"][0, 0] == 2.0  # median of 1.0, 3.0
    assert combined["tau"][0, 1] == 5.0  # only epoch 2 had this pixel
    assert combined["coverage"] == 1.0
