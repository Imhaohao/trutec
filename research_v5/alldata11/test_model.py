"""The all-data stage-2 file loads and keeps the original feature count."""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

import unittest
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
NEW_MODEL = HERE / "stage2_alldata_hardneg_0.75.joblib"
ORIGINAL_MODEL = HERE.parent / "structure" / "cache" / "models" / "hardneg_0.75.joblib"


class ModelFileTests(unittest.TestCase):
    def test_alldata_model_has_the_original_feature_count(self) -> None:
        new = joblib.load(NEW_MODEL)
        old = joblib.load(ORIGINAL_MODEL)
        self.assertIn("model", new)
        self.assertEqual(int(new["model"].n_features_in_), int(old["model"].n_features_in_))
        self.assertEqual(len(new["record"]["feature_names"]), len(old["record"]["feature_names"]))


if __name__ == "__main__":
    unittest.main()
