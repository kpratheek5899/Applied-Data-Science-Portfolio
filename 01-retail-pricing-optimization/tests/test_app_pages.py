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


if __name__ == "__main__":
    unittest.main()
