# Deploying to Streamlit Community Cloud

The app is self-contained: it only reads the small precomputed CSVs under
`data/app/` (already committed to git, ~4.7MB total) and the four packages
in `01-retail-pricing-optimization/requirements.txt` (`streamlit`, `pandas`,
`numpy`, `matplotlib`). It never needs the ~970MB simulated dataset, never
refits the Bayesian model, and never imports `pymc`/`cvxpy`/`statsmodels`
at runtime — verified by running the app against an isolated virtualenv
containing *only* those four packages before writing this doc.

## One thing to get right: which `requirements.txt`

The **repo root** also has a `requirements.txt` — a raw `pip freeze` dump
(UTF-16, ~164 lines, covers all three portfolio projects including
`pymc`/`cvxpy`/`statsmodels`/etc.). That is **not** the one this app should
build from. Streamlit Community Cloud can be told explicitly which
requirements file to use (see step 4) — set it to
`01-retail-pricing-optimization/requirements.txt` explicitly rather than
relying on auto-detection, so the build never picks up the root file by
accident (which would at minimum waste build minutes pulling in every other
project's dependencies, and the UTF-16 encoding may not even parse as a
valid requirements file).

## Steps

1. **Push to GitHub.** All Phase 1-5 work has been committed locally but
   not pushed — confirm with whoever's driving this before running
   `git push`, since it's a shared/visible action.

2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in
   with the GitHub account that owns `kpratheek5899/Applied-Data-Science-Portfolio`.

3. Click **"Create app"** → **"Deploy a public app from GitHub"**.

4. Fill in:
   - **Repository:** `kpratheek5899/Applied-Data-Science-Portfolio`
   - **Branch:** `main`
   - **Main file path:** `01-retail-pricing-optimization/app/streamlit_app.py`
   - Under **"Advanced settings"**: set **Python version to 3.12** (what
     this app was built and tested against), and if there's a field for
     the requirements/dependencies file path, set it explicitly to
     `01-retail-pricing-optimization/requirements.txt`.

5. Optionally customize the subdomain (the app URL is
   `https://<your-choice>.streamlit.app`).

6. Click **Deploy**. First build takes a few minutes (installing 4 small
   packages, no compilation needed — much faster than anything involving
   `pymc`/`cvxpy`).

7. **Verify the live app**, not just that it loads:
   - Landing page renders with the sidebar GitHub link.
   - Scenario Explorer: try at least one preset and manual/date-range mode,
     toggle Bayesian optimization on/off, move the risk-aversion slider.
   - Decision Replay: step through a few days, try both a short and a long
     window (confirm the date picker won't let you pick an invalid
     start-date/window combination).

## If the build fails

- **Dependency error mentioning `pymc`, `cvxpy`, or an encoding issue** —
  the wrong `requirements.txt` was picked up; re-check step 4.
- **`ModuleNotFoundError` for something in `src/`** — check the main file
  path is exactly `01-retail-pricing-optimization/app/streamlit_app.py`;
  the app pages resolve `src/` relative to their own file location, which
  depends on the repo structure staying intact.
- **Missing data files** — confirm `01-retail-pricing-optimization/data/app/*.csv`
  are actually present in the GitHub repo (`git status`/`git log` should
  show them tracked, not gitignored — only `data/simulated`, `data/raw`,
  `data/processed` are excluded).
