# Deploying to Streamlit Community Cloud

The app is self-contained: it only reads the small precomputed CSVs under
`data/app/` (already committed to git, ~4.7MB total) and the five packages
in `01-retail-pricing-optimization/app/requirements.txt` (`streamlit`,
`pandas`, `numpy`, `matplotlib`, `scipy`). It never needs the ~970MB
simulated dataset, never refits the Bayesian model, and never imports
`pymc`/`cvxpy`/`statsmodels` at runtime — verified by running the app
against an isolated virtualenv containing *only* those five packages
before writing this doc.

## One thing to get right: which `requirements.txt`

The **repo root** also has a `requirements.txt` — a raw `pip freeze` dump
(UTF-16, ~164 lines, covers all three portfolio projects including
`pymc`/`cvxpy`/`statsmodels`/a Windows-only `pywinpty` package that can't
even build on Linux). That is **not** the one this app should build from.

Streamlit Community Cloud auto-detects the requirements file by looking
**in the same directory as the main script**, and only falls back to the
repo root if it finds nothing there. That's why this project's
`requirements.txt` lives at `01-retail-pricing-optimization/app/requirements.txt`
— right next to `streamlit_app.py` — rather than one level up. (An earlier
version of this file lived one directory up and was silently skipped in
favor of the root file, which broke a real deploy: the build tried to
compile `pywinpty` on Streamlit Cloud's Linux servers and failed outright.)
Don't move it back up a level, and don't rely on an "Advanced settings"
field to override the path — Community Cloud does not reliably expose one;
directory placement is the mechanism that actually works.

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
     this app was built and tested against). No dependencies-file field
     needs setting — `requirements.txt` living next to `streamlit_app.py`
     is what makes Community Cloud find the right one automatically.

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
