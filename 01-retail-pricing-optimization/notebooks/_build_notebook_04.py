"""
One-off helper used to assemble 04_bayesian_elasticity_modeling.ipynb with
real executed outputs baked in. Not part of the production pipeline -- safe
to delete after the notebook has been generated. Mirrors the harness in
_build_notebook_01.py / _build_notebook_03.py.

NUTS sampling for this model takes ~15 minutes on this machine (no C
compiler available for PyTensor -- see the notebook's own explanation), so
this script loads the already-fit posterior from data/processed/ rather
than refitting live. To regenerate the cached fit from scratch:

    python -c "import sys; sys.path.insert(0, '../src'); from utils import load_retail_data; from modeling import fit_bayesian_model; ..."

Run from the notebooks/ directory:
    python _build_notebook_04.py
"""

import base64
import io
import json
import pickle
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "../src")


CELLS = []


def md(text):
    CELLS.append({"type": "markdown", "source": text})


def code(text):
    CELLS.append({"type": "code", "source": text})


# ---------------------------------------------------------------------------
# Notebook content
# ---------------------------------------------------------------------------

md("""\
# Nova Retail — Phase 3: Bayesian Hierarchical Elasticity Modeling

Phase 2 estimated one price elasticity per product type using fixed effects
and full controls, recovering true elasticity within ~0.2-2.8% -- but its
95% confidence intervals were built from cluster-robust standard errors with
only 7-16 SKU clusters per product type, a small-cluster setting where those
intervals are likely a bit too narrow.

Phase 3 addresses that directly with a **Bayesian hierarchical model**:
instead of one elasticity per product type, it estimates one elasticity
*per SKU*, partially pooled toward its product type's mean, with a full
posterior distribution (not an asymptotic approximation) quantifying
uncertainty at both levels.

This notebook also documents two real problems found and fixed while
building the model -- both are as important to the result as the final
numbers, so they're shown here rather than edited out.
""")

code("""\
import sys
sys.path.insert(0, "../src")

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

from utils import load_retail_data
from modeling import prepare_bayesian_data, build_hierarchical_model, summarize_bayesian_elasticity

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
plt.rcParams["figure.figsize"] = (9, 5)
""")

md("""\
## 1. Why this model can't run on the full 2.9M-row panel

Phase 2's fixed-effects models are closed-form OLS -- fast regardless of row
count. A hierarchical model with a SKU-level random effect on the price
coefficient has to be fit with MCMC (NUTS), and this machine has no C
compiler available for PyTensor, so it falls back to a much slower pure
Python/NumPy execution path. A representative model timed at this scale
made full-panel MCMC impractical (hours, not minutes) -- so the model is
fit on a **random 2% row-level subsample** (~58k of the ~2.9M rows) instead.

**This was not the first approach tried.** The original version of this
notebook aggregated the data up to (date, sku) -- averaging price across
each day's ~80 store/channel rows -- reasoning that store/channel shouldn't
matter for a SKU-level elasticity estimate and that shrinking ~2.9M rows to
~36k would make MCMC tractable. That turned out to be a real bug, not just
a shortcut: most of a SKU's row-level price movement comes from independent
per-row noise (`price_index`), which averages toward zero across 80 rows,
leaving mostly *promotion*-driven price changes -- which are collinear with
the `promotion_depth` control already in the model. Recovered elasticity
degraded sharply for the two highest-promo-sensitivity product types, and
one run even flipped the sign for Premium entirely. Randomly subsampling
*rows* instead of averaging them preserves the real price variation that
identifies elasticity, while still cutting the panel down to a size NUTS
can sample in minutes. `prepare_bayesian_data` docstring has the full
account.
""")

code("""\
DATA_PATH = "../data/simulated/nova_retail_simulated_data_v3_full.csv"
df = load_retail_data(DATA_PATH)

agg = prepare_bayesian_data(df, sample_frac=0.02, random_seed=42)
print(f"Rows in Bayesian model sample: {len(agg):,} ({len(agg) / len(df):.2%} of raw rows)")
print(agg.groupby("product_type", observed=True)["sku"].nunique())
""")

md("""\
## 2. Model structure

`log(units) = alpha_sku + beta_sku * log(price)_demeaned + gamma_promo[type] * promotion_depth + gamma . controls`

- `beta_sku` (price elasticity): partially pooled -- each SKU's elasticity
  is drawn from a Normal centered on its product type's mean elasticity
  (`mu_type`), which is itself drawn from a global mean (`mu_global`) across
  product types. Non-centered parameterization (`z_type`, `z_sku`) avoids
  the funnel geometry that causes divergences in centered hierarchical
  models.
- `log_price` is demeaned *within each SKU* first (same within-transform
  Phase 2's fixed-effects model uses), which keeps it near-orthogonal to
  `alpha_sku` -- this alone fixed a max-tree-depth problem that was making
  the model >10x slower to sample than a same-size synthetic proxy.
- `gamma_promo` (promotion depth's effect on demand) varies **by product
  type**, not shared globally. An earlier version shared one promotion
  coefficient across all four types, which converged cleanly (good r-hat,
  no divergences) but recovered elasticity 9-18% off true for three of four
  types -- a real, stable bias, not a sampling artifact. Forcing a plain
  OLS to share one promotion coefficient across types on the same data
  reproduced those exact numbers, confirming the cause: the simulator's
  promotion sensitivity genuinely varies by product type (Low/Medium/High/
  Very High per the spec), so a shared coefficient misattributes each
  type's own promotion-driven demand lift into its price coefficient by a
  different amount. Marketing spend keeps one shared coefficient, since
  channel elasticities *are* specified uniformly across the business.
""")

code("""\
model, meta = build_hierarchical_model(agg)
model
""")

md("""\
## 3. Fit

Fit with NUTS: 4 chains, 800 tuning + 800 sampling draws each,
`target_accept=0.9`. This takes ~15 minutes on this machine (no C compiler
for PyTensor), so the already-fit posterior is loaded from
`data/processed/phase3_idata.pkl` rather than refit live in this notebook.
To reproduce from scratch: `fit_bayesian_model(df)` runs this exact
pipeline end to end.
""")

code("""\
with open("../data/processed/phase3_idata.pkl", "rb") as f:
    idata = pickle.load(f)

print(f"Chains: {idata.posterior.sizes['chain']}, draws/chain: {idata.posterior.sizes['draw']}")
print(f"Divergences: {int(idata.sample_stats.diverging.sum())}")
print(f"Draws that hit max tree depth (>=10): {int((idata.sample_stats.tree_depth.to_numpy() >= 10).sum())}")
""")

md("""\
## 4. Convergence diagnostics

r-hat close to 1.00 and healthy effective sample size for the parameters
that matter -- product-type and SKU-level elasticity.
""")

code("""\
diag = az.summary(idata, var_names=["mu_type", "beta_sku"], ci_prob=0.95)
print(f"max r-hat: {diag['r_hat'].astype(float).max():.4f}")
print(f"min ess_bulk: {diag['ess_bulk'].astype(float).min():.0f}")
""")

code("""\
az.plot_trace(idata, var_names=["mu_global", "sigma_type", "sigma_sku", "mu_type"])
plt.tight_layout()
plt.show()
""")

md("""\
Trace plots for the key hyperparameters: all four chains overlapping with no
visible trend or stuck regions is the visual signature of a converged
sampler, consistent with the r-hat/ESS numbers above.
""")

md("""\
## 5. Product-type-level elasticity: posterior vs. truth
""")

code("""\
sku_summary, type_summary = summarize_bayesian_elasticity(idata, agg, meta)
type_summary[["mean", "sd", "eti95_lb", "eti95_ub", "true_elasticity", "pct_error"]]
""")

md("""\
All four product types recover true elasticity within **~2-11%** -- a
material improvement over the shared-promotion-coefficient version of this
model (which was off by up to 18%), and in the same ballpark as Phase 2's
point estimates. Premium carries the largest remaining error; it has both
the lowest promo sensitivity (0.5, the lowest of all four types per the
spec) and the smallest within-SKU price variation, so it has the weakest
identifying signal of the four -- a plausible, specific reason rather than
just "the model is worse at this one."
""")

md("""\
## 6. SKU-level elasticity, visualized

Every one of the 50 SKUs gets its own posterior, partially pooled toward
its product type. Points ordered by product type; the vertical line marks
each type's true elasticity.
""")

code("""\
sku_plot = sku_summary.sort_values(["true_elasticity", "mean"]).copy()
sku_plot["y_pos"] = np.arange(len(sku_plot))
colors = {
    "Commodity": "#c0392b", "Promo Sensitive": "#e67e22",
    "Seasonal": "#2c7fb8", "Premium": "#27ae60",
}

fig, ax = plt.subplots(figsize=(9, 11))

for pt, g in sku_plot.groupby("product_type", observed=True):
    ax.axvline(g["true_elasticity"].iloc[0], color=colors[pt], linewidth=0.8, linestyle="--", alpha=0.6)
    ax.errorbar(
        g["mean"], g["y_pos"],
        xerr=[g["mean"] - g["eti95_lb"], g["eti95_ub"] - g["mean"]],
        fmt="o", markersize=4, capsize=2,
        color=colors[pt], label=pt,
    )

ax.set_yticks(sku_plot["y_pos"])
ax.set_yticklabels(sku_plot.index, fontsize=7)
ax.set_xlabel("Price elasticity")
ax.set_title("Per-SKU posterior elasticity (95% credible interval), dashed lines = true elasticity by type")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()
""")

md("""\
## 7. Phase 2 vs. Phase 3

Phase 2's per-type fixed-effects estimate side by side with Phase 3's
product-type-level posterior mean.
""")

code("""\
phase2_results = pd.DataFrame([
    {"product_type": "Commodity", "true_elasticity": -2.0, "phase2_full_coef": -2.0336, "phase2_pct_error": 1.6805},
    {"product_type": "Premium", "true_elasticity": -0.8, "phase2_full_coef": -0.8220, "phase2_pct_error": 2.7527},
    {"product_type": "Promo Sensitive", "true_elasticity": -1.8, "phase2_full_coef": -1.8035, "phase2_pct_error": 0.1932},
    {"product_type": "Seasonal", "true_elasticity": -1.2, "phase2_full_coef": -1.1979, "phase2_pct_error": 0.1724},
]).set_index("product_type")

comparison = phase2_results.join(type_summary[["mean", "sd", "pct_error"]]).rename(
    columns={"mean": "phase3_mean", "sd": "phase3_sd", "pct_error": "phase3_pct_error"}
)
comparison.round(4)
""")

md("""\
Phase 2 (closed-form OLS, one estimate per product type, cluster-robust SEs
with a small-cluster caveat) is more accurate on point estimates for three
of four product types here. That's expected, not a failure of Phase 3: OLS
on ~700k-900k rows per product type has far more information than a 2%
MCMC-tractable subsample, and Phase 2's control coefficients are entirely
separate per product type (equivalent to fitting the controls
type-by-type), while Phase 3 only lets `promotion_depth`'s coefficient vary
by type to keep the model size sampleable. What Phase 3 adds that Phase 2
structurally cannot: a genuine per-*SKU* elasticity estimate (fifty of
them, not four) with a real posterior -- useful directly for Phase 4 if
SKU-level (not just product-type-level) price recommendations are wanted,
and for honestly representing uncertainty without relying on
asymptotic-cluster-count assumptions.
""")

md("""\
## Summary

- Bayesian hierarchical partial pooling recovers product-type elasticity
  within ~2-11% of true values (Commodity 1.9%, Promo Sensitive 1.1%,
  Seasonal 4.1%, Premium 11.0%), with full posterior uncertainty at both
  the SKU and product-type level, and clean convergence (r-hat ~1.00,
  ess_bulk in the thousands, zero divergences).
- Two real problems were found and fixed in the process, not just tuned
  away: (1) aggregating to (date, sku) grain destroyed the row-level price
  variation that identifies elasticity -- fixed by row-level random
  subsampling; (2) sharing one promotion-depth coefficient across product
  types misattributed type-specific promotional lift into the price
  coefficient -- fixed by letting it vary by product type, matching how the
  simulator (and the spec) actually define promotion sensitivity.
- Premium's larger residual error has a specific, plausible cause (lowest
  promo sensitivity and price variation of the four types = weakest
  identifying signal in a 2% subsample) rather than being unexplained noise.

**Is this a good foundation for Phase 4?** Yes, with a clear choice to make
explicit: Phase 4's optimizer can consume either Phase 2's four per-type
point estimates (more precise, no per-SKU granularity) or Phase 3's fifty
per-SKU posterior means (individual-SKU granularity plus real uncertainty
for risk-aware pricing, at the cost of somewhat more noise for Premium
specifically). Both are now validated against ground truth and available.
""")


# ---------------------------------------------------------------------------
# Execute cells, capture outputs, assemble .ipynb
# ---------------------------------------------------------------------------

def run_notebook(cells):
    namespace = {}
    nb_cells = []

    for i, cell in enumerate(cells):
        print(f"processing cell {i} ({cell['type']})...", flush=True)
        if cell["type"] == "markdown":
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": cell["source"].splitlines(keepends=True),
            })
            continue

        src = cell["source"]
        outputs = []

        stdout_buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_buffer

        result_repr = None
        error = None
        try:
            lines = src.rstrip(chr(10)).split(chr(10))
            *body, last = lines
            body_src = chr(10).join(body)
            if body_src.strip():
                exec(compile(body_src, f"<cell {i}>", "exec"), namespace)
            try:
                result = eval(compile(last, f"<cell {i}>", "eval"), namespace)
                if result is not None:
                    result_repr = repr(result)
            except SyntaxError:
                exec(compile(last, f"<cell {i}>", "exec"), namespace)
        except Exception as e:  # pragma: no cover
            error = e
        finally:
            sys.stdout = old_stdout

        stdout_text = stdout_buffer.getvalue()

        if error is not None:
            raise RuntimeError(f"Cell {i} failed:{chr(10)}{src}") from error

        if stdout_text:
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": stdout_text.splitlines(keepends=True),
            })

        if plt.get_fignums():
            fig = plt.gcf()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("ascii")
            outputs.append({
                "output_type": "display_data",
                "data": {"image/png": img_b64, "text/plain": ["<Figure>"]},
                "metadata": {},
            })
            plt.close(fig)

        if result_repr is not None:
            outputs.append({
                "output_type": "execute_result",
                "execution_count": i + 1,
                "data": {"text/plain": result_repr.splitlines(keepends=True) or [result_repr]},
                "metadata": {},
            })

        nb_cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": i + 1,
            "source": src.splitlines(keepends=True),
            "outputs": outputs,
        })

    return nb_cells


if __name__ == "__main__":
    t0 = time.time()
    nb_cells = run_notebook(CELLS)

    notebook = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open("04_bayesian_elasticity_modeling.ipynb", "w") as f:
        json.dump(notebook, f, indent=1)

    print(f"Notebook written. Elapsed: {time.time() - t0:.1f}s")
