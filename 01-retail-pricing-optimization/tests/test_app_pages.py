"""
Smoke tests for the Streamlit pages themselves, using Streamlit's own
`AppTest` headless runner (no browser needed, but it actually executes the
page scripts the way a real session would -- unlike importing the module
directly, this catches Streamlit-API-level bugs, e.g. dtype/argument
mismatches inside widget calls).

Run from the project root:
    python -m unittest tests.test_app_pages -v
"""

import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LANDING_PAGE = str(PROJECT_ROOT / "app" / "streamlit_app.py")
SCENARIO_EXPLORER_PAGE = str(PROJECT_ROOT / "app" / "pages" / "1_Scenario_Explorer.py")


class TestLandingPage(unittest.TestCase):
    def test_loads_without_exceptions(self):
        at = AppTest.from_file(LANDING_PAGE)
        at.run(timeout=30)
        self.assertEqual(list(at.exception), [])


class TestScenarioExplorer(unittest.TestCase):
    def test_loads_without_exceptions_in_custom_mode(self):
        at = AppTest.from_file(SCENARIO_EXPLORER_PAGE)
        at.run(timeout=30)
        self.assertEqual(list(at.exception), [])
        self.assertGreater(len(at.dataframe), 0)
        self.assertGreater(len(at.get("metric")), 0)

    def test_every_preset_loads_without_exceptions(self):
        at = AppTest.from_file(SCENARIO_EXPLORER_PAGE)
        at.run(timeout=30)
        starting_point = at.selectbox[0]
        preset_names = [o for o in starting_point.options if o != "Custom"]
        self.assertEqual(len(preset_names), 6)

        for name in preset_names:
            with self.subTest(preset=name):
                at.selectbox[0].select(name).run(timeout=30)
                self.assertEqual(list(at.exception), [], f"Preset {name!r} raised an exception")

    def test_date_range_mode_loads_without_exceptions(self):
        at = AppTest.from_file(SCENARIO_EXPLORER_PAGE)
        at.run(timeout=30)
        at.selectbox[0].select("Inventory Overstock").run(timeout=30)
        range_checkbox = at.checkbox[0]
        range_checkbox.set_value(True).run(timeout=30)
        self.assertEqual(list(at.exception), [])

    def test_objective_switch_changes_recommendation(self):
        # "High-Elasticity Commodity Product" defaults to maximize_profit;
        # switching it to maximize_revenue should change the recommendation
        # (elastic demand -> the two objectives push price in different
        # directions, per Phase 4's tested behavior).
        at = AppTest.from_file(SCENARIO_EXPLORER_PAGE)
        at.run(timeout=30)
        at.selectbox[0].select("High-Elasticity Commodity Product").run(timeout=30)
        profit_price = at.get("metric")[0].value

        objective_box = [sb for sb in at.selectbox if sb.key and sb.key.startswith("objective_")][0]
        self.assertEqual(objective_box.value, "maximize_profit")
        objective_box.select("maximize_revenue").run(timeout=30)
        self.assertEqual(list(at.exception), [])
        revenue_price = at.get("metric")[0].value

        self.assertNotEqual(profit_price, revenue_price)

    def test_risk_aversion_slider_changes_recommendation(self):
        # protect_inventory uses a hard inventory floor and is deliberately
        # unaffected by risk_aversion (see optimize_price_bayesian's
        # docstring). And for maximize_profit, risk_aversion only has
        # something to trade off against if the unconstrained profit-optimal
        # price sits *inside* the allowed price-change band with genuine
        # stockout risk there -- too tight a band (e.g. the default 50%) or
        # too loose/too tight an inventory override just saturates one
        # extreme regardless of risk_aversion. inventory=6200 with a 200%
        # max change band was verified directly against demand_model to put
        # the profit-optimal point mid-distribution (~42% stockout
        # probability at risk_aversion=0), where the tradeoff is real.
        at = AppTest.from_file(SCENARIO_EXPLORER_PAGE)
        at.run(timeout=30)
        at.selectbox[0].select("Inventory Shortage").run(timeout=30)

        objective_box = [sb for sb in at.selectbox if sb.key and sb.key.startswith("objective_")][0]
        objective_box.select("maximize_profit").run(timeout=30)

        max_change_slider = [s for s in at.slider if s.key and s.key.startswith("maxchange_")][0]
        max_change_slider.set_value(2.0).run(timeout=30)

        inventory_input = [n for n in at.number_input if n.key and n.key.startswith("inventory_")][0]
        inventory_input.set_value(6200.0).run(timeout=30)
        self.assertEqual(list(at.exception), [])

        risk_slider = [s for s in at.slider if s.key and s.key.startswith("risk_")][0]
        risk_slider.set_value(0.0).run(timeout=30)
        self.assertEqual(list(at.exception), [])
        low_risk_price = at.get("metric")[0].value

        risk_slider.set_value(1.0).run(timeout=30)
        self.assertEqual(list(at.exception), [])
        high_risk_price = at.get("metric")[0].value

        self.assertNotEqual(low_risk_price, high_risk_price)

    def test_disabling_bayesian_mode_falls_back_without_exceptions(self):
        at = AppTest.from_file(SCENARIO_EXPLORER_PAGE)
        at.run(timeout=30)
        at.selectbox[0].select("Inventory Overstock").run(timeout=30)

        bayes_checkbox = [c for c in at.checkbox if c.key and c.key.startswith("bayes_")][0]
        bayes_checkbox.set_value(False).run(timeout=30)
        self.assertEqual(list(at.exception), [])


if __name__ == "__main__":
    unittest.main()
