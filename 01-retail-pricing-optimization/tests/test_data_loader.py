"""
Tests for src/data_loader.py and the precomputed data/app/ files it reads.

Run from the project root:
    python -m unittest tests.test_data_loader -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from data_loader import (
    load_sku_master,
    load_daily_timeseries,
    load_posterior_samples,
    get_date_bounds,
    get_elasticity_samples,
    resolve_elasticity,
)

GROUND_TRUTH_COLUMN_NAMES = {"true_price_elasticity", "true_daily_demand", "base_daily_demand"}


class TestSkuMaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sku_master = load_sku_master()

    def test_fifty_skus_no_nulls(self):
        self.assertEqual(len(self.sku_master), 50)
        self.assertEqual(self.sku_master.isna().sum().sum(), 0)

    def test_no_ground_truth_columns(self):
        # sku_master.csv is read by the decision path (scenario_engine,
        # demand_model) -- it must never carry a ground-truth column, even
        # one that happens to go unused, to keep the "estimated-only"
        # boundary structurally enforced rather than just a convention.
        leaked = GROUND_TRUTH_COLUMN_NAMES & set(self.sku_master.columns)
        self.assertEqual(leaked, set(), f"Ground-truth columns leaked into sku_master.csv: {leaked}")

    def test_elasticities_are_negative(self):
        self.assertTrue((self.sku_master["elasticity_phase3_mean"] < 0).all())
        self.assertTrue((self.sku_master["elasticity_phase2"] < 0).all())

    def test_resolve_elasticity_prefers_phase3(self):
        row = self.sku_master.iloc[0]
        self.assertAlmostEqual(resolve_elasticity(row), row["elasticity_phase3_mean"])

    def test_resolve_elasticity_falls_back_to_phase2_when_phase3_missing(self):
        row = self.sku_master.iloc[0].copy()
        row["elasticity_phase3_mean"] = float("nan")
        self.assertAlmostEqual(resolve_elasticity(row), row["elasticity_phase2"])


class TestDailyTimeseries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.daily = load_daily_timeseries()

    def test_full_date_coverage_per_sku(self):
        counts = self.daily.groupby("sku").size()
        self.assertEqual(counts.min(), 731)
        self.assertEqual(counts.max(), 731)

    def test_no_nulls(self):
        self.assertEqual(self.daily.isna().sum().sum(), 0)

    def test_date_bounds_match_known_simulator_range(self):
        lo, hi = get_date_bounds(self.daily)
        self.assertEqual(lo, pd.Timestamp("2024-01-01"))
        self.assertEqual(hi, pd.Timestamp("2025-12-31"))

    def test_inventory_identity_holds(self):
        implied_ending = self.daily["starting_inventory"] - self.daily["actual_units"]
        pd.testing.assert_series_equal(
            implied_ending, self.daily["ending_inventory"], check_names=False
        )

    def test_true_price_elasticity_is_the_only_ground_truth_column(self):
        # This file DOES intentionally carry true_price_elasticity, for
        # src/replay_engine.py's outcome-realization step -- but nothing
        # else ground-truth-flavored should be present.
        other_leaks = (GROUND_TRUTH_COLUMN_NAMES - {"true_price_elasticity"}) & set(self.daily.columns)
        self.assertEqual(other_leaks, set())


class TestPosteriorSamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.posterior = load_posterior_samples()

    def test_every_sku_has_full_draw_count(self):
        counts = self.posterior.groupby("sku").size()
        self.assertEqual(counts.nunique(), 1)
        self.assertEqual(counts.iloc[0], 300)

    def test_get_elasticity_samples_matches_sku_master_mean(self):
        sku_master = load_sku_master()
        row = sku_master.iloc[0]
        samples = get_elasticity_samples(self.posterior, row["sku"])
        self.assertAlmostEqual(samples.mean(), row["elasticity_phase3_mean"], places=1)


if __name__ == "__main__":
    unittest.main()
