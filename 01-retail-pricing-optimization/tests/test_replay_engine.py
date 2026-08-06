"""
Tests for src/replay_engine.py (Phase 5d closed-loop Decision Replay).

Run from the project root:
    python -m unittest tests.test_replay_engine -v
"""

import sys
import unittest
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from data_loader import load_sku_master, load_daily_timeseries, load_posterior_samples, get_elasticity_samples
from demand_model import DemandContext
from replay_engine import run_closed_loop_replay, replay_to_frame, realize_true_outcome


class TestGroundTruthBoundary(unittest.TestCase):
    def test_demand_context_never_carries_true_elasticity(self):
        # The decision path (demand_model.build_demand_context /
        # recommend_price[_bayesian]) must never be able to see ground
        # truth -- structurally enforced by DemandContext simply having no
        # field for it, not just a convention.
        context_fields = {f.name for f in fields(DemandContext)}
        self.assertNotIn("true_price_elasticity", context_fields)
        self.assertNotIn("demand_units_uncapped", context_fields)

    def test_realize_true_outcome_is_a_pure_function_of_its_arguments(self):
        # Sanity check on the one function allowed to use true elasticity:
        # a higher candidate price should reduce realized units (a valid
        # demand curve), independent of anything else in the module.
        low_price_units = realize_true_outcome(
            candidate_price=50, actual_price=100, demand_units_uncapped=1000,
            true_price_elasticity=-2.0, available_inventory=10_000,
        )
        high_price_units = realize_true_outcome(
            candidate_price=150, actual_price=100, demand_units_uncapped=1000,
            true_price_elasticity=-2.0, available_inventory=10_000,
        )
        self.assertGreater(low_price_units, high_price_units)

    def test_realize_true_outcome_respects_inventory_cap(self):
        units = realize_true_outcome(
            candidate_price=50, actual_price=100, demand_units_uncapped=1000,
            true_price_elasticity=-2.0, available_inventory=10,
        )
        self.assertLessEqual(units, 10)


class TestClosedLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sku_master = load_sku_master()
        cls.daily = load_daily_timeseries()
        cls.posterior = load_posterior_samples()
        # A window comfortably inside the dataset's date range for every SKU.
        cls.start_date = pd.Timestamp("2025-03-03")

    def test_day_one_starting_inventory_matches_actual(self):
        days = run_closed_loop_replay(
            self.sku_master, self.daily, "SKU_003", self.start_date, n_days=5, objective="maximize_profit"
        )
        self.assertEqual(days[0].optimizer_starting_inventory, days[0].actual_starting_inventory)

    def test_inventory_override_replaces_day_one_starting_inventory(self):
        days_default = run_closed_loop_replay(
            self.sku_master, self.daily, "SKU_003", self.start_date, n_days=3, objective="maximize_profit"
        )
        days_override = run_closed_loop_replay(
            self.sku_master,
            self.daily,
            "SKU_003",
            self.start_date,
            n_days=3,
            objective="maximize_profit",
            inventory_override=500.0,
        )
        self.assertEqual(days_override[0].optimizer_starting_inventory, 500.0)
        # The actual (historical) trajectory is a pure passthrough -- must
        # be completely unaffected by the override.
        self.assertEqual(days_override[0].actual_starting_inventory, days_default[0].actual_starting_inventory)
        self.assertEqual(days_override[0].actual_units, days_default[0].actual_units)

    def test_day_two_inventory_reflects_day_one_decision_not_fixed_history(self):
        # On a day with no replenishment, optimizer Day-2 starting inventory
        # must equal Day-1's own ending inventory (the closed-loop part) --
        # not whatever the next fixed historical row says.
        days = run_closed_loop_replay(
            self.sku_master, self.daily, "SKU_003", self.start_date, n_days=5, objective="maximize_profit"
        )
        for i in range(1, len(days)):
            replenished = days[i].actual_starting_inventory > days[i - 1].actual_ending_inventory + 1e-6
            if not replenished:
                self.assertAlmostEqual(
                    days[i].optimizer_starting_inventory, days[i - 1].optimizer_ending_inventory, places=3
                )
                # And specifically NOT the fixed historical value (unless
                # they happen to coincide, vanishingly unlikely once the
                # optimizer has recommended a different price than history).
                if abs(days[i - 1].optimizer_price - days[i - 1].actual_price) > 0.01:
                    self.assertNotAlmostEqual(
                        days[i].optimizer_starting_inventory, days[i].actual_starting_inventory, places=0
                    )

    def test_replenishment_is_bounded_not_runaway(self):
        # A protect_inventory policy sells less than history most days;
        # inventory should track the real network's target level, not grow
        # without bound across a longer window (see replay_engine.py's
        # replenishment design note for the bug this guards against).
        days = run_closed_loop_replay(
            self.sku_master, self.daily, "SKU_003", self.start_date, n_days=14, objective="protect_inventory"
        )
        max_actual = max(d.actual_starting_inventory for d in days)
        max_optimizer = max(d.optimizer_starting_inventory for d in days)
        # Generous bound: optimizer inventory shouldn't blow past the
        # historical range by more than 50%.
        self.assertLess(max_optimizer, max_actual * 1.5)

    def test_ending_inventory_identity_holds_both_trajectories(self):
        days = run_closed_loop_replay(
            self.sku_master, self.daily, "SKU_003", self.start_date, n_days=7, objective="maximize_profit"
        )
        for d in days:
            self.assertAlmostEqual(
                d.optimizer_ending_inventory, d.optimizer_starting_inventory - d.optimizer_units, places=3
            )
            self.assertAlmostEqual(
                d.actual_ending_inventory, d.actual_starting_inventory - d.actual_units, places=3
            )

    def test_cumulative_profit_is_running_sum(self):
        days = run_closed_loop_replay(
            self.sku_master, self.daily, "SKU_003", self.start_date, n_days=6, objective="maximize_profit"
        )
        df = replay_to_frame(days)
        self.assertAlmostEqual(df["actual_cumulative_profit"].iloc[-1], df["actual_profit"].sum(), places=3)
        self.assertAlmostEqual(
            df["optimizer_cumulative_profit"].iloc[-1], df["optimizer_profit"].sum(), places=3
        )
        # Each day's cumulative delta must equal that day's own profit.
        import numpy as np

        deltas = df["actual_cumulative_profit"].diff().dropna().to_numpy()
        self.assertTrue(np.allclose(deltas, df["actual_profit"].iloc[1:].to_numpy()))

    def test_bayesian_mode_runs_without_exceptions(self):
        samples = get_elasticity_samples(self.posterior, "SKU_003")
        days = run_closed_loop_replay(
            self.sku_master,
            self.daily,
            "SKU_003",
            self.start_date,
            n_days=5,
            objective="maximize_profit",
            elasticity_samples=samples,
            risk_aversion=0.4,
        )
        self.assertEqual(len(days), 5)

    def test_window_past_dataset_end_raises(self):
        with self.assertRaises(ValueError):
            run_closed_loop_replay(
                self.sku_master, self.daily, "SKU_003", pd.Timestamp("2025-12-28"), n_days=10
            )

    def test_optimizer_units_never_exceed_inventory(self):
        days = run_closed_loop_replay(
            self.sku_master, self.daily, "SKU_003", self.start_date, n_days=7, objective="protect_inventory"
        )
        for d in days:
            self.assertLessEqual(d.optimizer_units, d.optimizer_starting_inventory + 1e-6)


if __name__ == "__main__":
    unittest.main()
