"""
Sequential conjugate Bayesian regression (Normal-Inverse-Gamma) for the
Adaptive Learning demo (Phase 5f).

Model: dlog_units_t = beta * dlog_price_t + eps_t, eps_t ~ N(0, sigma^2),
where dlog_price_t/dlog_units_t are log-differences *from a reference
anchor point* (typically the previous day's own realized (price, units)),
not absolute log levels. beta | sigma^2 ~ N(mu, sigma^2 * v); sigma^2 ~
InvGamma(a, b).

Why differences-from-anchor rather than a standard `log_units ~ alpha +
beta*log_price` regression with a free intercept: an earlier version of
this module used exactly that 2-parameter form, and it broke in practice.
Once the pricing policy converges toward a narrow price range (which it
does quickly once it starts favoring a particular price), the sequence of
observed log_price values stops varying, and a 2D regression can't
separately identify alpha from beta when its one predictor is nearly
constant -- a single noisy sequential update in that near-unidentified
regime pushed the fitted elasticity to an economically nonsensical
*positive* value within a handful of days (verified against real data
while building this). Regressing log-differences from an anchor instead
collapses the model to a single parameter (beta alone, the elasticity),
matching exactly how every other demand curve in this app already works
(`optimization.constant_elasticity_units`: swing from an anchor point by
`elasticity`, never an absolute intercept) -- there is no second parameter
left to be confounded with beta, and by construction dx=dy=0 reproduces
the anchor exactly, so no intercept term is needed at all.

Deliberately closed-form, not full PyMC/MCMC: Phase 3's hierarchical model
is far too slow to refit every simulated day inside an interactive
Streamlit loop. This is the standard 1D Bayesian linear regression NIG
conjugacy (e.g. Murphy, "Machine Learning: A Probabilistic Perspective"
sec 7.6), fast enough to run hundreds of times per page load.

Promotion / event / day-of-week confounding, and how it's handled without
adding free parameters: `dlog_units_t` above is *not* driven by price alone
-- if a promotion or event also moved between the anchor day and today,
that shows up in `dlog_units_t` too, and gets misread as price sensitivity
whenever it's correlated with `dlog_price_t` (the same mechanism as an
uncontrolled difference-in-differences estimate). Rather than adding those
as free parameters here (reintroducing the near-unidentified-regression
risk described above), `control_log_effect` below subtracts Phase 2/3's
*already-fitted* promotion/event/day-of-week effect sizes from
`dlog_units_t` before it reaches `update_posterior` -- those coefficients
were estimated once, offline, over the full two-year panel (ample data,
already validated -- see PROGRESS.md), so only price elasticity itself is
still being learned online.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class NIGPosterior:
    """beta | sigma^2 ~ N(mu, sigma^2 * v); sigma^2 ~ InvGamma(a, b)."""

    mu: float
    v: float
    a: float
    b: float


def build_prior(
    prior_mean_elasticity: float,
    prior_strength: float,
    generic_sigma2_guess: float = 0.05,
) -> NIGPosterior:
    """
    A deliberately weak starting belief about beta (elasticity), parameterized
    by an equivalent-sample-size `prior_strength` (n0): roughly "how many
    pseudo-observations worth of confidence" the prior represents. Smaller n0
    -> wider day-1 credible interval -> more visible learning over the window.

    `prior_mean_elasticity` should come from a *generic* source (e.g. the
    SKU's product-type-level Phase 2 estimate, `sku_master.csv`'s
    `elasticity_phase2`) -- never the SKU's own well-fit Phase 3 posterior or
    the true elasticity, or there would be nothing left to learn.

    `generic_sigma2_guess` is on the scale of squared log-differences (this
    model regresses log-unit differences on log-price differences, both
    typically small in magnitude day to day), much smaller than a plausible
    residual variance for absolute log-units -- 0.05 is a reasonable default
    order of magnitude, not fit to any particular SKU.
    """
    n0 = max(float(prior_strength), 1e-3)
    mu = float(prior_mean_elasticity)
    v = 1.0 / n0
    a0 = max(1.0, n0 / 2)
    b0 = a0 * generic_sigma2_guess
    return NIGPosterior(mu=mu, v=v, a=a0, b=b0)


def update_posterior(prior: NIGPosterior, dlog_price: float, dlog_units: float) -> NIGPosterior:
    """Closed-form NIG update after observing one (dlog_price, dlog_units) pair."""
    x, y = float(dlog_price), float(dlog_units)

    v0_inv = 1.0 / prior.v
    vn_inv = v0_inv + x * x
    v_n = 1.0 / vn_inv
    mu_n = v_n * (v0_inv * prior.mu + x * y)

    a_n = prior.a + 0.5
    prior_quad = prior.mu * v0_inv * prior.mu
    new_quad = mu_n * vn_inv * mu_n
    b_n = prior.b + 0.5 * (y**2 + prior_quad - new_quad)

    return NIGPosterior(mu=mu_n, v=v_n, a=a_n, b=b_n)


def sample_theta(posterior: NIGPosterior, rng: np.random.Generator) -> float:
    """Draw one beta sample from the posterior -- the Thompson Sampling step."""
    sigma2 = 1.0 / rng.gamma(shape=posterior.a, scale=1.0 / posterior.b)
    return float(rng.normal(posterior.mu, np.sqrt(sigma2 * posterior.v)))


def posterior_mean_beta(posterior: NIGPosterior) -> float:
    return float(posterior.mu)


def posterior_sd_beta(posterior: NIGPosterior) -> float:
    """Marginal posterior SD of beta (using the posterior mean of sigma^2)."""
    mean_sigma2 = posterior.b / (posterior.a - 1) if posterior.a > 1 else posterior.b / posterior.a
    return float(np.sqrt(mean_sigma2 * posterior.v))


def credible_interval_beta(posterior: NIGPosterior, prob: float = 0.9) -> tuple[float, float]:
    """
    Marginal `prob`-credible interval for beta: integrating out sigma^2
    gives a Student-t distribution with 2a degrees of freedom, location mu,
    scale sqrt((b/a) * v) -- the standard NIG marginal result, closed form.
    """
    df = 2 * posterior.a
    scale = np.sqrt((posterior.b / posterior.a) * posterior.v)
    lo, hi = stats.t.interval(prob, df, loc=posterior.mu, scale=scale)
    return float(lo), float(hi)


@dataclass(frozen=True)
class ControlEffects:
    """
    Phase 3's fitted promotion/event/day-of-week effect sizes (posterior
    means from `src/modeling.py`'s hierarchical model), used by
    `control_log_effect` to residualize Adaptive Learning's online update.
    Exported once, offline, by `scripts/build_app_data.py` into
    `data/app/control_effects.csv` -- nothing at runtime refits these.
    """

    promo_type_coef: dict[str, float]
    promo_mean: float
    promo_std: float
    event_phase_coef: dict[str, float]
    day_of_week_coef: dict[str, float]

    @classmethod
    def zero(cls) -> "ControlEffects":
        """All coefficients zero -- residualization becomes a no-op. Used in tests to
        confirm this addition doesn't silently move behavior when there's nothing to correct for."""
        return cls(promo_type_coef={}, promo_mean=0.0, promo_std=1.0, event_phase_coef={}, day_of_week_coef={})


def parse_control_effects(effects_df: pd.DataFrame) -> ControlEffects:
    """Build a ControlEffects from the raw rows of data/app/control_effects.csv."""

    def _lookup(kind: str) -> dict[str, float]:
        sub = effects_df[effects_df["kind"] == kind]
        return dict(zip(sub["key"], sub["coefficient"].astype(float)))

    promo_standardization = _lookup("promo_standardization")
    return ControlEffects(
        promo_type_coef=_lookup("promo_type"),
        promo_mean=promo_standardization.get("mean", 0.0),
        promo_std=promo_standardization.get("std", 1.0),
        event_phase_coef=_lookup("event_phase"),
        day_of_week_coef=_lookup("day_of_week"),
    )


def control_log_effect(
    effects: ControlEffects,
    product_type: str,
    promotion_depth: float,
    event_phase: str,
    day_of_week: str,
) -> float:
    """
    Phase 3's fitted log(units) contribution from promotion depth, event
    window, and day-of-week -- the confounders `dlog_units_t` would
    otherwise silently carry (see module docstring). Standardizes
    `promotion_depth` exactly as Phase 3's fit did (z-scored against that
    fitting panel's own mean/std, not the online window's), and matches
    `gamma_promo` to the SKU's own product type. Unknown category values
    (should not occur for in-sample data) fall back to 0.0 rather than
    raising.
    """
    promo_std = effects.promo_std if effects.promo_std else 1.0
    promo_z = (promotion_depth - effects.promo_mean) / promo_std
    return (
        effects.promo_type_coef.get(product_type, 0.0) * promo_z
        + effects.event_phase_coef.get(event_phase, 0.0)
        + effects.day_of_week_coef.get(day_of_week, 0.0)
    )
