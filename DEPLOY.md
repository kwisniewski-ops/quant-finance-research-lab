# Deployment Guide

## 1. Push to GitHub

```bash
# from the repo root (git is already initialized with an initial commit)
gh repo create quant-research-lab --public --source=. --push
# or manually:
#   create an empty repo named quant-research-lab on github.com, then:
git remote add origin https://github.com/<your-username>/quant-research-lab.git
git push -u origin main
```

Two GitHub Actions ship with the repo and run automatically on push:

- `tests.yml` — full pytest suite on Python 3.10/3.11/3.12 (a green badge-worthy CI run on every commit)
- `pages.yml` — deploys `app/frontend/` to GitHub Pages

## 2. Enable GitHub Pages (one-time)

Repo → **Settings → Pages → Source: GitHub Actions**. The next push to `main`
(or a manual run of the `deploy-website` workflow) publishes the site at
`https://<your-username>.github.io/quant-research-lab/`.

All site links are relative, so it works at any subpath — and you can also just
open `app/frontend/index.html` locally in a browser.

## 3. After pushing, update two placeholders

- README + site link GitHub at `https://github.com/kylewisniewski/quant-research-lab` —
  adjust if your username/repo name differs (`grep -rn "kylewisniewski/quant-research-lab" README.md app/frontend/`).
- Optionally add the live Pages URL to the repo description and README.

## 4. Custom domain (optional)

Settings → Pages → Custom domain (e.g., `lab.yourdomain.com`), add a CNAME DNS
record pointing to `<your-username>.github.io`, and commit the `CNAME` file
GitHub creates into `app/frontend/`.

## 5. Refreshing data

Cached snapshots in `data/snapshots/` keep everything reproducible offline.
To refresh and re-execute the notebooks:

```bash
python -m src.data.market_data_loader --refresh --tickers SPY QQQ IWM EFA EEM AGG TLT LQD GLD DBC VNQ USMV MTUM VLUE QUAL
python -m src.data.factor_data_loader --refresh
python notebooks/_build/execute.py     # rebuilds all six notebooks
```
