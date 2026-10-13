"""The v13 stage-1 and stage-2 files load with the recipe feature counts."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

import unittest

import joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import hatch_texture  # noqa: E402
from texture_learning import pair_feature_names  # noqa: E402

STAGE1 = HERE / "stage1_alldata.joblib"
STAGE2 = HERE / "stage2_alldata_hardneg_0.75.joblib"
ORIGINAL_STAGE1 = ROOT / "models" / "models" / "texture-hgb-crop-v1.joblib"
ORIGINAL_STAGE2 = HERE.parent / "alldata12" / "stage2_alldata_hardneg_0.75.joblib"


class ModelFileTests(unittest.TestCase):
    def test_both_models_load_with_expected_feature_counts(self) -> None:
        stage1 = joblib.load(STAGE1)
        stage2 = joblib.load(STAGE2)
        original_stage1 = joblib.load(ORIGINAL_STAGE1)
        original_stage2 = joblib.load(ORIGINAL_STAGE2)
        expected_stage1 = len(pair_feature_names(hatch_texture.texture_feature_names("base"), "crop"))
        self.assertEqual(expected_stage1, 184)
        self.assertEqual(int(stage1["model"].n_features_in_), expected_stage1)
        self.assertEqual(len(stage1["record"]["feature_names"]), expected_stage1)
        self.assertEqual(int(stage1["model"].n_features_in_), int(original_stage1["model"].n_features_in_))
        self.assertEqual(int(stage2["model"].n_features_in_), 144)
        self.assertEqual(len(stage2["record"]["feature_names"]), 144)
        self.assertEqual(int(stage2["model"].n_features_in_), int(original_stage2["model"].n_features_in_))


if __name__ == "__main__":
    unittest.main()
