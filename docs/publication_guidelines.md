# Publication guidelines — Quantitative Markets & Institutions Lab

These are the binding rules for everything the lab publishes: new findings, new
models, new syntheses, and edits to existing pages. They encode the decisions
made during the 2026-08 "working observatory" redesign so future work lands
consistent without re-deriving the reasoning. The enforceable subset lives in
`tests/test_frontend_content.py` — **the test suite is the contract; this
document is the intent behind it.** When the two disagree, fix one of them
deliberately, in the same commit.

---

## 1. The organizing idea

**The lab's own computed evidence is the ornament.** No stock imagery, no
decorative graphics, no framework runtime. Every visual on the site must be
derived from a number the lab actually produced — and every number on the site
should be traceable to its artifact (notebook, JSON snapshot, commit).

A visitor should be able to tell what the lab found without reading a full
page, and a skeptic should be able to find the boundary of every claim.

## 2. Voice and register

- Impersonal and declarative. Never first person ("I", "we", "our"), never
  "Kyle" on a lab surface, no question marks in titles or H1s. (Test-enforced.)
- Density belongs at depth. Finding pages may be maximally precise from
  depth 02 (Evidence) downward. Front doors — home, hub ledes, card answers —
  get one idea per sentence and one sentence where one will do.
- **Register variation rule:** every major surface carries at least one
  plain-speed sentence a tired reader absorbs at full pace. Dense phrasing
  only lands when something plain sits beside it.
- **Say it once.** One methodological commitment statement per page, maximum.
  The homepage carries exactly two flags: the thesis blockquote and the
  closing standard band headline. Do not add a third.
- **Voice examples** (from the 2026-08 register pass — imitate these moves):
  - *Dense only:* "The lab examines how quantitative methods behave when
    estimates leave sample, correlations change sign, liquidity deteriorates,
    and institutional constraints reshape market transmission."
    → *Anchored:* "Models hold until conditions change. The lab measures when,
    how, and what it costs a decision."
  - *Dense only:* "Activity, inflation, market liquidity, and institutional
    response alter how shocks propagate through asset prices and portfolios."
    → *Anchored:* "Shocks do not travel one fixed road; activity, inflation,
    liquidity, and institutional response reroute them."
  - *The house standard, already in the record:* "A result that cannot be
    reproduced from a clean clone is an anecdote."
- Never market the simplicity ("plain-language", "simple first", "short
  answer" are prohibited strings — test-enforced). The lab is allowed to BE
  simple; it is not allowed to advertise it.

## 3. Word budgets

| Surface | Budget |
|---|---|
| Homepage prose (ledes, answers, headings, agenda bodies, the quote) | ≤ 250 words |
| Hub-page lede (`findings`, `tools`, `method`, `research-log`) | 1 sentence |
| Finding card answer (home + findings hub) | 1 sentence |
| Finding page, depth 01 (Principal finding) | ≤ 120 words |
| Finding page, depths 02–05 | as long as the evidence requires |
| Figure caption | ≤ 60 words, and it must earn them (see §5) |

Prose means sentences a visitor reads. Mono navigation labels, metrics,
kickers, and card links are scannable structure and sit outside the budget —
but they are not a loophole: a "label" long enough to read as a sentence is one.

If new content pushes a surface over budget, something else on that surface
moves down a level or is cut — the budget does not stretch.

## 4. Layout grammar

- **Vertical flow.** Content stacks; the site does not place paragraph beside
  paragraph. Grids of prose are converted to single-column hairline rows.
  (Numbers, labels, and buttons may sit side by side; sentences may not.)
- **Hairline rules, never boxes.** A component gets a `border-top` (or a
  left rule for callouts), not a four-sided frame. Hover reveals the gold
  accent rule growing across the row.
- **Let it breathe.** Section spacing uses the `clamp(4rem, 9vw, 7rem)` scale;
  do not compress it to fit more above the fold. The fold shows: headline,
  lede, both actions, meter. Nothing else is entitled to the first screen.
- Both themes always. Every new component must be checked in light and dark
  (`prefers-color-scheme`); colors come only from the custom properties in
  `css/main.css` — never hex values in markup.

## 5. Evidence visuals

Every published finding ships two graphics, both derived only from numbers in
the public record:

1. **A spark** — the ≤40-element inline SVG on its card (home + findings hub),
   showing the headline result. Classes: `.finding-spark` vocabulary only.
2. **A figure** — one `<figure class="figure">` at the top of the finding page
   (after the proof strip), using the `.fig-*` class vocabulary.

Rules for both:

- **Exact vs. schematic is a hard boundary.** If the record stores the series,
  draw the series. If it stores only summary statistics, the caption MUST say
  the drawing is schematic and name what is exact (e.g. the VaR breach strip:
  run-length structure exact from transition counts, placement schematic).
  A figure that overstates the record is a defect, not a style choice.
- Chart colors are the validated series `--chart-1` … `--chart-5`, assigned in
  fixed order, first slot to the finding's own result. Before changing the
  palette, re-run the dataviz validator against BOTH surface colors
  (`#faf8f4` light, `#121a24` dark) — all five checks must pass.
- Every SVG `<title>` and `<desc>` carries an `id` attribute. A bare
  `<title>` string breaks the metadata test, which counts them.
- No text in a visual that violates §2 (no "?", no first person).
- Accessibility: `role="img"` + `aria-labelledby` on figures; sparks are
  `aria-hidden="true"` because their card text carries the content.

## 6. Adding a new finding — checklist

1. Produce the research artifact first: executed notebook, JSON snapshot in
   `app/frontend/data/` (via `notebooks/_build/`), entry in the research
   record with status and provenance (commit hash, data vintage).
2. Copy the structure of `finding-var-backtest.html` — it is the canonical
   template: head metadata, breadcrumb schema, depth-nav (exact five anchors
   `#bottom-line #evidence #practice #limitations #explore` and labels),
   `finding-article` class, proof strip, figure, regime lens only if the
   dossier genuinely supports it (max three findings carry one).
3. Write depth 01 inside its word budget; put the density below it.
4. Build the spark and the figure per §5; add the spark to BOTH card surfaces
   (index and findings hub) with a one-sentence answer.
5. Add the Atom feed entry (`feed.xml`) — declarative title, no "?", updated
   feed timestamp ordering.
6. Run `python3 scripts/normalize_frontend_shell.py` to rebuild the shared
   shell, breadcrumbs, and social-image references.
7. `python3 -m pytest tests/test_frontend_content.py -q` must pass 25/25
   (or more — add assertions for the new page's contract in the same commit).
8. Screenshot the new page at 1440px and 390px, light and dark, with and
   without reduced motion. Look at them. Label collisions and blank reveals
   are defects.
9. Update counts that the record asserts (investigation counts on the
   research log and proof strips) — the tests will catch a mismatch.

## 7. Publishing pipeline

- **Source of truth:** this repository. The site at
  `www.kylewisniewski.com/lab` is a vendored copy inside the
  `kyle-wisniewski-site` repository (`public/lab/`), refreshed by that repo's
  `npm run sync:lab` (schedule/dispatch workflow) and deployed by Vercel.
- There is exactly one published address: `www.kylewisniewski.com/lab`. The
  `lab.kylewisniewski.com` subdomain and its GitHub Pages deployment were retired
  in August 2026 (see `DEPLOY.md`); do not reintroduce a second host.
- **Social card:** one PNG, exactly 1200×630, named `og-lab-v<N>.png`. Its
  URL and alt text live in TWO places that must move together:
  `scripts/normalize_frontend_shell.py` (`SOCIAL_IMAGE_URL`,
  `SOCIAL_IMAGE_ALT`) and `tests/test_frontend_content.py`
  (`SOCIAL_IMAGE_URL`, `SOCIAL_IMAGE_ALT_HTML`). Bump the version number when
  the design changes materially; never edit the file in place (caches).
- Stylesheet changes bump the `?v=` query on the `main.css` link so caches
  release the old design.

## 8. What "done" means

A change is publishable when: tests pass; both themes and both widths have
been looked at by a human (or an agent that read the screenshots); no surface
exceeded its word budget; every new number is traceable; and every schematic
visual says it is schematic. If any of those is uncomfortable to satisfy, the
content is not ready — the standard does not bend to the deadline.
