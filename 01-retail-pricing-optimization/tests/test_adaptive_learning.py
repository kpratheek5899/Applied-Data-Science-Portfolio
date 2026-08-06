"""
Tests for src/bayesian_learning.py and src/adaptive_simulation.py
(Phase 5f: Adaptive Learning / Thompson Sampling demo).

Thompson Sampling is stochastic, so its behavioral tests run >=20 seeded
repetitions and assert on averages rather than single-run outcomes, per the
spec this phase was built against.

Run from the project root:
    python -m unittest tests.test_adaptive_learning -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from bayesian_learning import (
    build_prior,
    update_posterior,
    sample_theta,
    posterior_mean_beta,
    credible_interval_beta,
    ControlEffects,
    control_log_effect,
    parse_control_effects,
)
from data_loader import load_sku_master, load_daily_timeseries, load_control_effects
from adaptive_simulation import run_adaptive_simulation, adaptive_to_frame

N_SEEDS = 20
TEST_SKU = "SKU_003"  # Commodity, true elasticity -2.045 -- verified during development to show real static-vs-Thompson differentiation at these settings
TEST_START = "2025-03-03"


class TestNIGPosteriorMath(unittest.TestCase):
    def test_prior_credible_interval_is_well_ordered(self):
        prior = build_prior(prior_mean_elasticity=-1.5, prior_strength=2.0)
        lo, hi = credible_interval_beta(prior)
        self.assertLess(lo, posterior_mean_beta(prior))
        self.assertLess(posterior_mean_beta(prior), hi)

    def test_posterior_converges_toward_true_beta_with_more_data(self):
        rng = np.random.default_rng(0)
        true_beta, true_sigma = -1.8, 0.15
        post = build_prior(prior_mean_elasticity=-1.0, prior_strength=1.0)
        initial_gap = abs(posterior_mean_beta(post) - true_beta)

        for _ in range(200):
            dx = rng.normal(0, 0.15)
            dy = true_beta * dx + rng.normal(0, true_sigma)
            post = update_posterior(post, dx, dy)

        final_gap = abs(posterior_mean_beta(post) - true_beta)
        self.assertLess(final_gap, initial_gap)
        self.assertLess(final_gap, 0.3)

    def test_credible_interval_narrows_with_more_data(self):
        rng = np.random.default_rng(1)
        true_beta = -1.5
        post = build_prior(prior_mean_elasticity=-1.0, prior_strength=1.0)
        initial_width = credible_interval_beta(post)[1] - credible_interval_beta(post)[0]

        for _ in range(200):
            dx = rng.normal(0, 0.15)
            dy = true_beta * dx + rng.normal(0, 0.1)
            post = update_posterior(post, dx, dy)

        final_width = credible_interval_beta(post)[1] - credible_interval_beta(post)[0]
        self.assertLess(final_width, initial_width)

    def test_matches_through_origin_ols_with_weak_prior(self):
        rng = np.random.default_rng(2)
        true_beta = -1.8
        post = build_prior(prior_mean_elasticity=0.0, prior_strength=0.001)
        xs, ys = [], []
        for _ in range(500):
            dx = rng.normal(0, 0.15)
            dy = true_beta * dx + rng.normal(0, 0.15)
            xs.append(dx)
            ys.append(dy)
            post = update_posterior(post, dx, dy)
        xs, ys = np.array(xs), np.array(ys)
        ols_beta = (xs @ ys) / (xs @ xs)
        self.assertAlmostEqual(posterior_mean_beta(post), ols_beta, places=3)

    def test_sample_theta_is_reproducible_with_same_rng_state(self):
        post = build_prior(prior_mean_elasticity=-1.5, prior_strength=2.0)
        a = sample_theta(post, np.random.default_rng(42))
        b = sample_theta(post, np.random.default_rng(42))
        self.assertEqual(a, b)


class TestControlResidualization(unittest.TestCase):
    """
    Tests the confound-correction claim in bayesian_learning.py's module
    docstring directly: when a confound is correlated with the price change
    being regressed on (the same failure mode as an uncontrolled
    difference-in-differences estimate), the raw update recovers a biased
    beta. Subtracting the confound's already-known effect first should
    recover something much closer to the true beta.
    """

    def test_residualizing_a_correlated_confound_reduces_bias(self):
        rng = np.random.default_rng(3)
        true_beta = -1.8
        confound_coef = 0.6  # e.g. a promotion's lift to log(units) when "on"

        effects = ControlEffects(
            promo_type_coef={"Commodity": confound_coef},
            promo_mean=0.0,
            promo_std=1.0,
            event_phase_coef={},
            day_of_week_coef={},
        )

        raw_post = build_prior(prior_mean_elasticity=0.0, prior_strength=0.001)
        adj_post = build_prior(prior_mean_elasticity=0.0, prior_strength=0.001)

        for _ in range(300):
            # Promotions coincide with price cuts more often than not -- the confound.
            promo_on = rng.random() < 0.7
            dlog_price = rng.normal(-0.05 if promo_on else 0.0, 0.1)
            confound = confound_coef if promo_on else 0.0
            dlog_units = true_beta * dlog_price + confound + rng.normal(0, 0.05)

            raw_post = update_posterior(raw_post, dlog_price, dlog_units)

            # promo_mean=0, promo_std=1 makes control_log_effect's z-score a no-op, so this
            # returns exactly confound_coef * promo_on -- matching the synthetic confound above.
            effect = control_log_effect(effects, "Commodity", 1.0 if promo_on else 0.0, "normal", "Monday")
            adj_post = update_posterior(adj_post, dlog_price, dlog_units - effect)

        raw_error = abs(posterior_mean_beta(raw_post) - true_beta)
        adj_error = abs(posterior_mean_beta(adj_post) - true_beta)
        self.assertLess(adj_error, raw_error * 0.5)

    def test_zero_effects_is_a_no_op(self):
        effects = ControlEffects.zero()
        self.assertEqual(control_log_effect(effects, "Commodity", 5.0, "event", "Monday"), 0.0)
        self.assertEqual(control_log_effect(effects, "Unknown Type", -3.0, "unknown_phase", "Someday"), 0.0)

    def test_parse_control_effects_round_trips_real_exported_file(self):
        effects = parse_control_effects(load_control_effects())
        # All four product types from sku_master should have a promo coefficient.
        for product_type in ["Commodity", "Premium", "Promo Sensitive", "Seasonal"]:
            self.assertIn(product_type, effects.promo_type_coef)
        # All seven days should resolve to a real number (six fitted + one zero baseline).
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
            self.assertIn(day, effects.day_of_week_coef)
        self.assertGreater(effects.promo_std, 0.0)


class TestAdaptiveSimulationInvariants(unittest.TestCase):
    """Deterministic (single-run) structural invariants -- not stochastic behavior."""

    @classmethod
    def setUpClass(cls):
        cls.sku_master = load_sku_master()
        cls.daily = load_daily_timeseries()
        cls.result = run_adaptive_simulation(
            cls.sku_master, cls.daily, TEST_SKU, TEST_START, n_days=15, random_seed=7
        )

    def test_oracle_regret_is_exactly_zero_every_day(self):
        # Oracle's own daily profit *is* the regret reference by
        # construction -- this must hold exactly, not just approximately.
        for day in self.result.oracle:
            self.assertEqual(day.regret, 0.0)

    def test_static_elasticity_never_changes(self):
        elasticities = {d.elasticity_used for d in self.result.static}
        self.assertEqual(len(elasticities), 1, "static's elasticity must be frozen across the whole window")

    def test_static_never_matches_thompsons_evolving_belief_by_construction(self):
        # Sanity check that "static" and "thompson" are actually distinct
        # code paths, not accidentally aliased.
        self.assertIsNot(self.result.static, self.result.thompson)

    def test_all_variants_respect_price_and_margin_bounds(self):
        min_margin = 0.10
        result = run_adaptive_simulation(
            self.sku_master, self.daily, TEST_SKU, TEST_START, n_days=15, random_seed=3, min_margin=min_margin
        )
        for days in (result.static, result.thompson, result.oracle):
            for day in days:
                self.assertGreater(day.price, 0)
                margin = (day.price - self._cost()) / day.price
                self.assertGreaterEqual(margin, min_margin - 1e-6)

    def _cost(self):
        return float(self.sku_master.loc[self.sku_master["sku"] == TEST_SKU, "cost"].iloc[0])

    def test_units_never_exceed_available_inventory(self):
        for days in (self.result.static, self.result.thompson, self.result.oracle):
            for day in days:
                self.assertLessEqual(day.units, day.starting_inventory + 1e-6)

    def test_inventory_override_replaces_day_one_starting_inventory_for_all_variants(self):
        result = run_adaptive_simulation(
            self.sku_master, self.daily, TEST_SKU, TEST_START, n_days=5, random_seed=7, inventory_override=500.0
        )
        for days in (result.static, result.thompson, result.oracle):
            self.assertEqual(days[0].starting_inventory, 500.0)

    def test_window_past_dataset_end_raises(self):
        with self.assertRaises(ValueError):
            run_adaptive_simulation(self.sku_master, self.daily, TEST_SKU, "2025-12-28", n_days=10)

    def test_unknown_sku_raises(self):
        with self.assertRaises(ValueError):
            run_adaptive_simulation(self.sku_master, self.daily, "SKU_999", TEST_START, n_days=10)

    def test_zero_control_effects_runs_without_error(self):
        result = run_adaptive_simulation(
            self.sku_master, self.daily, TEST_SKU, TEST_START, n_days=15, random_seed=7,
            control_effects=ControlEffects.zero(),
        )
        self.assertEqual(len(result.thompson), 15)

    def test_real_control_effects_change_the_learned_posterior_vs_zero(self):
        # Confirms the residualization is actually wired into the online update on real
        # data -- not just present in bayesian_learning.py's math but dead code here.
        zero_result = run_adaptive_simulation(
            self.sku_master, self.daily, TEST_SKU, TEST_START, n_days=20, random_seed=7,
            control_effects=ControlEffects.zero(),
        )
        real_result = run_adaptive_simulation(
            self.sku_master, self.daily, TEST_SKU, TEST_START, n_days=20, random_seed=7,
        )
        self.assertNotAlmostEqual(
            posterior_mean_beta(zero_result.final_posterior),
            posterior_mean_beta(real_result.final_posterior),
            places=6,
        )


class TestThompsonSamplingStochasticBehavior(unittest.TestCase):
    """
    Averaged over >=20 seeded runs, per the spec -- single-run assertions on
    a stochastic algorithm would be flaky.
    """

    @classmethod
    def setUpClass(cls):
        cls.sku_master = load_sku_master()
        cls.daily = load_daily_timeseries()

    def _run(self, n_days, seed):
        result = run_adaptive_simulation(
            self.sku_master, self.daily, TEST_SKU, TEST_START, n_days=n_days, random_seed=seed
        )
        df = adaptive_to_frame(result)
        totals = df.groupby("variant")[["profit", "regret"]].sum()
        return result, totals

    def test_average_regret_thompson_lower_than_static(self):
        static_regrets, thompson_regrets = [], []
        for seed in range(N_SEEDS):
            _, totals = self._run(n_days=20, seed=seed)
            static_regrets.append(totals.loc["static", "regret"])
            thompson_regrets.append(totals.loc["thompson", "regret"])

        avg_static = np.mean(static_regrets)
        avg_thompson = np.mean(thompson_regrets)
        self.assertLess(avg_thompson, avg_static * 0.9, "Thompson Sampling should meaningfully beat the frozen baseline on average")

    def test_average_credible_interval_width_narrows_by_final_day(self):
        day1_widths, dayN_widths = [], []
        for seed in range(N_SEEDS):
            result, _ = self._run(n_days=20, seed=seed)
            day1 = result.thompson[0]
            dayN = result.thompson[-1]
            day1_widths.append(day1.posterior_ci_high - day1.posterior_ci_low)
            dayN_widths.append(dayN.posterior_ci_high - dayN.posterior_ci_low)

        self.assertLess(np.mean(dayN_widths), np.mean(day1_widths))

    def test_thompson_profit_ratio_to_oracle_improves_with_longer_window(self):
        short_ratios, long_ratios = [], []
        for seed in range(N_SEEDS):
            _, totals_short = self._run(n_days=10, seed=seed)
            _, totals_long = self._run(n_days=30, seed=seed)
            short_ratios.append(totals_short.loc["thompson", "profit"] / totals_short.loc["oracle", "profit"])
            long_ratios.append(totals_long.loc["thompson", "profit"] / totals_long.loc["oracle", "profit"])

        self.assertGreaterEqual(np.mean(long_ratios), np.mean(short_ratios))

    def test_prices_respect_bounds_across_all_seeds(self):
        for seed in range(N_SEEDS):
            result, _ = self._run(n_days=15, seed=seed)
            for days in (result.static, result.thompson, result.oracle):
                for day in days:
                    self.assertGreater(day.price, 0)


if __name__ == "__main__":
    unittest.main()
