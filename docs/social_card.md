# Social Card Production Note

The public share card is `app/frontend/og-lab-v4.png` at exactly 1200 × 630 pixels.
Every indexable lab page references this cache-busted filename and uses one accurate
alternate description. Older files remain historical assets and are not referenced.

## Source prompt

The v4 card was created in built-in image-edit mode from `og-lab-v3.png` with this
production specification:

> Preserve the large title “QUANTITATIVE MARKETS & INSTITUTIONS LAB,” editorial serif
> typography, left-aligned hierarchy, ivory research-paper background, thin oxblood
> rules, navy/ochre scientific chart, open research notebook, and landscape composition.
> Replace only the existing tagline with the exact text “INDEPENDENT RESEARCH ON
> MODELS, REGIMES & INSTITUTIONS.” Add no other text or watermark.

The selected result was resized to the Open Graph standard of 1200 × 630 pixels.
`tests/test_frontend_content.py` verifies the output format, dimensions, shared URL,
and matching alternate text.
