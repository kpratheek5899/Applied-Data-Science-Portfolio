"""
Tests for src/scenario_engine.py, src/demand_model.py, src/explanations.py.

Run from the project root:
    python -m unittest tests.test_scenarios -v
"""

import sys
import unittest
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from data_loader import load_sku_master, load_daily_timeseries
from scenario_engine import build_predefined_scenarios, build_manual_scenario, Scenario
from demand_model import build_demand_context, recommend_price
from explanations import generate_explanation


class TestPresetsArePrefillsOnly(unittest.TestCase):
    """Modification 1: presets must carry only input state, never a precomputed result."""

    @classmethod
    def setUpClass(cls):
        cls.sku_master = load_sku_master()
        cls.daily = load_daily_timeseries()
        cls.scenarios = build_predefined_scenarios(cls.sku_master, cls.daily)

    def test_six_predefined_scenarios(self):
        self.assertEqual(len(self.scenarios), 6)

    def test_scenario_has_no_result_fields(self):
        result_like_names = {"recommended_price", "profit", "revenue", "expected_units", "curve"}
        scenario_field_names = {f.name for f in fields(Scenario)}
        self.assertEqual(result_like_names & scenario_field_names, set())

    def test_every_scenario_sku_exists_in_sku_master(self):
        valid_skus = set(self.sku_master["sku"])
        for scenario in self.scenarios.values():
            self.assertIn(scenario.sku, valid_skus)

    def test_every_scenario_produces_a_valid_recommendation(self):
        # The whole point of Modification 1: presets flow through the exact
        # same build_demand_context -> recommend_price path as manual input.
        for name, scenario in self.scenarios.items():
            with self.subTest(scenario=name):
                ctx = build_demand_context(
                    self.sku_master, self.daily, scenario.sku, scenario.start_date, scenario.end_date
                )
                result = recommend_price(
                    ctx,
                    scenario.objective,
                    price_min=scenario.price_min,
                    price_max=scenario.price_max,
                    min_margin=scenario.min_margin,
                    max_price_change_pct=scenario.max_price_change_pct,
                )
                self.assertGreater(result["recommended_price"], 0)
                self.assertGreaterEqual(result["margin_pct"], scenario.min_margin - 1e-6)


class TestManualScenarioSharesCodePath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sku_master = load_sku_master()
        cls.daily = load_daily_timeseries()

    def test_manual_scenario_same_shape_as_preset(self):
        preset = build_predefined_scenarios(self.sku_master, self.daily)["Inventory Overstock"]
        manual = build_manual_scenario(
            sku=preset.sku, start_date=preset.start_date, end_date=preset.end_date, objective=preset.objective
        )
        self.assertEqual({f.name for f in fields(preset)}, {f.name for f in fields(manual)})

    def test_manual_single_day_matches_preset_recommendation_for_same_inputs(self):
        preset = build_predefined_scenarios(self.sku_master, self.daily)["Inventory Overstock"]
        manual = build_manual_scenario(
            sku=preset.sku,
            start_date=preset.start_date,
            end_date=preset.end_date,
            objective=preset.objective,
            price_min=preset.price_min,
            price_max=preset.price_max,
            min_margin=preset.min_margin,
            max_price_change_pct=preset.max_price_change_pct,
        )
        for scenario in (preset, manual):
            ctx = build_demand_context(self.sku_master, self.daily, scenario.sku, scenario.start_date, scenario.end_date)
            result = recommend_price(
                ctx,
                scenario.objective,
                price_min=scenario.price_min,
                price_max=scenario.price_max,
                min_margin=scenario.min_margin,
                max_price_change_pct=scenario.max_price_change_pct,
            )
            if scenario is preset:
                preset_price = result["recommended_price"]
            else:
                manual_price = result["recommended_price"]
        self.assertAlmostEqual(preset_price, manual_price, places=6)


class TestDateRangeAggregation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sku_master = load_sku_master()
        cls.daily = load_daily_timeseries()

    def test_range_current_units_equals_manual_daily_sum(self):
        sku = self.sku_master.iloc[0]["sku"]
        window = self.daily[(self.daily["sku"] == sku)].sort_values("date").iloc[10:17]
        start, end = window["date"].iloc[0], window["date"].iloc[-1]

        ctx = build_demand_context(self.sku_master, self.daily, sku, start, end)
        result = recommend_price(ctx, "maximize_profit")

        self.assertEqual(ctx.is_multi_day, True)
        self.assertEqual(len(ctx.day_dates), 7)
        self.assertAlmostEqual(result["current_units"], window["actual_units"].sum(), places=3)
        self.assertAlmostEqual(
            result["current_revenue"], (window["actual_price"] * window["actual_units"]).sum(), places=1
        )

    def test_single_day_is_not_flagged_multi_day(self):
        sku = self.sku_master.iloc[0]["sku"]
        first_date = self.daily[self.daily["sku"] == sku]["date"].min()
        ctx = build_demand_context(self.sku_master, self.daily, sku, first_date)
        self.assertFalse(ctx.is_multi_day)


class TestExplanationsVaryWithDirection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sku_master = load_sku_master()
        cls.daily = load_daily_timeseries()

    def test_explanation_mentions_reduce_when_price_drops(self):
        scenario = build_predefined_scenarios(self.sku_master, self.daily)["Inventory Overstock"]
        ctx = build_demand_context(self.sku_master, self.daily, scenario.sku, scenario.start_date, scenario.end_date)
        result = recommend_price(ctx, scenario.objective, max_price_change_pct=scenario.max_price_change_pct)
        self.assertLess(result["price_change_pct"], 0)
        explanation = generate_explanation(ctx, result)
        self.assertIn("Reduce", explanation)

    def test_explanation_mentions_increase_when_price_rises(self):
        scenario = build_predefined_scenarios(self.sku_master, self.daily)["Low-Elasticity Premium Product"]
        ctx = build_demand_context(self.sku_master, self.daily, scenario.sku, scenario.start_date, scenario.end_date)
        result = recommend_price(ctx, scenario.objective, max_price_change_pct=scenario.max_price_change_pct)
        self.assertGreater(result["price_change_pct"], 0)
        explanation = generate_explanation(ctx, result)
        self.assertIn("Increase", explanation)

    def test_explanation_cites_real_elasticity_value(self):
        scenario = build_predefined_scenarios(self.sku_master, self.daily)["High-Elasticity Commodity Product"]
        ctx = build_demand_context(self.sku_master, self.daily, scenario.sku, scenario.start_date, scenario.end_date)
        result = recommend_price(ctx, scenario.objective, max_price_change_pct=scenario.max_price_change_pct)
        explanation = generate_explanation(ctx, result)
        self.assertIn(f"{ctx.elasticity:.2f}", explanation)


if __name__ == "__main__":
    unittest.main()
