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
- `pages.yml` — temporarily mirrors `app/frontend/` to the legacy lab subdomain during migration

## 2. Publish the canonical site

The canonical publication lives at **https://www.kylewisniewski.com/lab/**. The personal-site
repository vendors `app/frontend/` with `scripts/sync-lab.mjs`, publishes those files under
`public/lab/`, and includes every lab page in the main sitemap. Run the personal site's lab
sync before its production build whenever this repository changes.

All internal lab links are relative, so the same source can also be opened locally.

## 3. After pushing, update two placeholders

- README + site link GitHub at `https://github.com/kwisniewski-ops/quant-finance-research-lab` —
  adjust if your username/repo name differs (`grep -rn "kwisniewski-ops/quant-finance-research-lab" README.md app/frontend/`).
- Optionally add the live Pages URL to the repo description and README.

## 4. Retired: the lab.kylewisniewski.com subdomain

`lab.kylewisniewski.com` was a temporary GitHub Pages mirror during the move to the
canonical `/lab` path. It was retired in August 2026 and **must not be recreated**:
the lab is published at one address only.

What was removed, and where it lived, in case any of it resurfaces:

- `app/frontend/CNAME` (claimed the subdomain for GitHub Pages)
- `.github/workflows/pages.yml` (deployed the mirror on every push)
- the GitHub Pages site and its custom domain, in this repository's Settings
- the `lab` CNAME record in the domain's DNS

No redirect was configured. The mirror was public only briefly, every page declared
`www.kylewisniewski.com/lab` as canonical throughout, and nothing linked to the
subdomain, so the address was removed outright rather than forwarded.

## 5. Refreshing data

Cached snapshots in `data/snapshots/` keep everything reproducible offline.
To refresh and re-execute the notebooks:

```bash
python -m src.data.market_data_loader --refresh --tickers SPY QQQ IWM EFA EEM AGG TLT LQD GLD DBC VNQ USMV MTUM VLUE QUAL
python -m src.data.factor_data_loader --refresh
python notebooks/_build/execute.py     # rebuilds all six notebooks
```
