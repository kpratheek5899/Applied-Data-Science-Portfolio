"""
Tests for src/optimization.py, runnable independently of any Streamlit
interface (no pytest in the project venv -- plain unittest, no new
dependency).

Run from the project root:
    python -m unittest tests.test_optimizer -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optimization import optimize_price, optimize_price_gp, resolve_price_bounds


# A deterministic "elastic" scenario (e < -1, e.g. Commodity/Seasonal/Promo
# Sensitive) and a deterministic "inelastic" scenario (e > -1, e.g. Premium)
# reused across tests so the expected direction of each result is known
# up front rather than just asserted as "some price came back".
ELASTIC = dict(base_price=100.0, base_units=1000.0, cost=50.0, elasticity=-2.0)
INELASTIC = dict(base_price=100.0, base_units=1000.0, cost=50.0, elasticity=-0.5)


class TestPriceBounds(unittest.TestCase):
    def test_recommended_price_within_bounds(self):
        for objective in ["maximize_profit", "maximize_revenue"]:
            result = optimize_price(objective, **ELASTIC, price_min=60, price_max=150)
            self.assertGreaterEqual(result["recommended_price"], 60)
            self.assertLessEqual(result["recommended_price"], 150)

    def test_minimum_margin_respected(self):
        result = optimize_price("maximize_revenue", **ELASTIC, min_margin=0.4, price_max=500)
        self.assertGreaterEqual(result["margin_pct"], 0.4 - 1e-6)

    def test_infeasible_bounds_raise(self):
        # min_margin of 0.9 on cost=50 implies price >= 500, but price_max=100
        with self.assertRaises(ValueError):
            resolve_price_bounds(base_price=100, cost=50, price_max=100, min_margin=0.9)

    def test_expected_units_do_not_exceed_inventory(self):
        result = optimize_price("protect_inventory", **ELASTIC, inventory=200, price_max=1000)
        self.assertLessEqual(result["expected_units"], 200 + 1e-6)
        self.assertFalse(result["stockout_risk"])


class TestInternalConsistency(unittest.TestCase):
    def test_revenue_and_profit_match_price_and_units(self):
        result = optimize_price("maximize_profit", **ELASTIC, price_max=300)
        self.assertAlmostEqual(
            result["revenue"], result["recommended_price"] * result["expected_units"], places=6
        )
        self.assertAlmostEqual(
            result["profit"],
            (result["recommended_price"] - ELASTIC["cost"]) * result["expected_units"],
            places=6,
        )

    def test_current_vs_recommended_pct_changes_are_consistent(self):
        result = optimize_price("maximize_profit", **ELASTIC, price_max=300)
        implied_price = result["current_price"] * (1 + result["price_change_pct"] / 100)
        self.assertAlmostEqual(implied_price, result["recommended_price"], places=6)


class TestObjectivesDiffer(unittest.TestCase):
    def test_profit_and_revenue_differ_for_elastic_product(self):
        # Elastic demand (e < -1): revenue is maximized by pushing price to
        # the floor, profit has an interior optimum -- these must differ.
        profit_result = optimize_price("maximize_profit", **ELASTIC, price_min=10, price_max=300)
        revenue_result = optimize_price("maximize_revenue", **ELASTIC, price_min=10, price_max=300)
        self.assertNotAlmostEqual(
            profit_result["recommended_price"], revenue_result["recommended_price"], places=1
        )

    def test_high_elasticity_reacts_more_to_price_change_than_low_elasticity(self):
        from optimization import constant_elasticity_units

        elastic_units = constant_elasticity_units(120, base_price=100, base_units=1000, elasticity=-2.0)
        inelastic_units = constant_elasticity_units(120, base_price=100, base_units=1000, elasticity=-0.5)

        elastic_drop_pct = (1000 - elastic_units) / 1000
        inelastic_drop_pct = (1000 - inelastic_units) / 1000
        self.assertGreater(elastic_drop_pct, inelastic_drop_pct)


class TestInventoryProtection(unittest.TestCase):
    def test_protect_inventory_price_is_geq_profit_price_when_constrained(self):
        profit_result = optimize_price("maximize_profit", **ELASTIC, price_max=1000)
        protect_result = optimize_price(
            "protect_inventory", **ELASTIC, inventory=150, price_max=1000
        )
        self.assertGreaterEqual(
            protect_result["recommended_price"], profit_result["recommended_price"] - 1e-6
        )

    def test_protect_inventory_matches_profit_when_unconstrained(self):
        # Baseline demand at current price is 1000 units; inventory of 5000
        # is never binding, so protect_inventory should collapse to plain
        # profit maximization.
        profit_result = optimize_price("maximize_profit", **ELASTIC, price_max=1000)
        protect_result = optimize_price(
            "protect_inventory", **ELASTIC, inventory=5000, price_max=1000
        )
        self.assertAlmostEqual(
            protect_result["recommended_price"], profit_result["recommended_price"], places=0
        )


class TestBusinessScenarios(unittest.TestCase):
    def test_overstock_scenario_can_recommend_lower_price(self):
        # Overstock: plenty of inventory, elastic product -- maximize_revenue
        # should recommend cutting price toward the floor to move volume.
        result = optimize_price("maximize_revenue", **ELASTIC, inventory=5000, price_min=40, price_max=150)
        self.assertLess(result["recommended_price"], ELASTIC["base_price"])

    def test_holiday_shortage_scenario_can_recommend_higher_price(self):
        # Shortage during a demand surge: inventory well below what baseline
        # demand would consume -- protect_inventory should raise price.
        result = optimize_price(
            "protect_inventory", **ELASTIC, inventory=100, price_max=1000
        )
        self.assertGreater(result["recommended_price"], ELASTIC["base_price"])


class TestGeometricProgrammingCrossCheck(unittest.TestCase):
    def test_grid_search_matches_cvxpy_gp_for_revenue(self):
        grid_result = optimize_price(
            "maximize_revenue", **ELASTIC, price_min=10, price_max=300, n_points=2000
        )
        gp_price = optimize_price_gp(
            base_price=ELASTIC["base_price"],
            base_units=ELASTIC["base_units"],
            elasticity=ELASTIC["elasticity"],
            price_low=10,
            price_high=300,
        )
        # Grid search is discretized (n_points over [10, 300]); GP solves
        # exactly. They should agree within roughly one grid step.
        grid_step = (300 - 10) / 2000
        self.assertAlmostEqual(grid_result["recommended_price"], gp_price, delta=grid_step * 2)


if __name__ == "__main__":
    unittest.main()
