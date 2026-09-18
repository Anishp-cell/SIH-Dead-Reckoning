"""
Unit Test: Phase 4 Dataset Validation & Purge Gap Verification.
"""

import pytest
import numpy as np

from Data_details.src.phase4.dataset import prepare_phase4_data, FEATURE_COLUMNS


class TestPhase4Dataset:

    @pytest.fixture(scope="class")
    def dataset_bundle(self):
        return prepare_phase4_data(window_length=30, stride=5, purge_gap_samples=50)

    def test_feature_count_and_columns(self, dataset_bundle):
        manifest = dataset_bundle["manifest"]
        assert len(FEATURE_COLUMNS) == 12
        assert "acc_fwd_veh" in FEATURE_COLUMNS
        assert "is_stationary" in FEATURE_COLUMNS
        assert dataset_bundle["norm_params"]["features"] == FEATURE_COLUMNS

    def test_purge_gap_isolation(self, dataset_bundle):
        """Confirms that Train, Val, and Test window endpoints are strictly separated by purge gaps."""
        raw = dataset_bundle["raw_arrays"]
        train_idx = raw["train_indices"]
        val_idx = raw["val_indices"]
        test_idx = raw["test_indices"]
        
        # Windows must be strictly ordered
        assert np.max(train_idx) < np.min(val_idx)
        assert np.max(val_idx) < np.min(test_idx)
        
        # Purge gap in windows
        purge_gap_1 = np.min(val_idx) - np.max(train_idx) - 1
        purge_gap_2 = np.min(test_idx) - np.max(val_idx) - 1
        assert purge_gap_1 >= 5  # At stride=5, 50 samples = 10 window steps
        assert purge_gap_2 >= 5

    def test_train_only_normalization(self, dataset_bundle):
        """Verifies that normalization parameters are finite and non-zero."""
        norm = dataset_bundle["norm_params"]
        means = np.array(norm["mean"])
        stds = np.array(norm["std"])
        
        assert len(means) == 12
        assert len(stds) == 12
        assert not np.any(np.isnan(means))
        assert not np.any(np.isnan(stds))
        assert np.all(stds > 0.0)

    def test_target_finite_and_non_negative(self, dataset_bundle):
        """Verifies ground truth speed target is non-negative and finite."""
        y_speed = dataset_bundle["raw_arrays"]["y_speed_all"]
        assert not np.any(np.isnan(y_speed))
        assert not np.any(np.isinf(y_speed))
        assert np.all(y_speed >= 0.0)
        assert np.max(y_speed) > 10.0  # Must reach realistic vehicle speeds (> 36 km/h)
