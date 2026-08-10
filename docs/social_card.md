# Social Card Production Note

The public share card is `app/frontend/og-lab-v3.png` at exactly 1200 × 630 pixels.
Every indexable lab page references this cache-busted filename and uses one accurate
alternate description. Older files remain historical assets and are not referenced.

## Source prompt

The v3 card was created in built-in image-edit mode from `og-lab-v2.png` with this
production specification:

> Change only the large title text from “QUANTITATIVE MARKETS RESEARCH LAB” to
> “QUANTITATIVE MARKETS & INSTITUTIONS LAB.” Preserve the high-contrast editorial
> serif typography, left-aligned hierarchy, ivory research-paper background, thin
> oxblood rules, navy/ochre scientific chart, open research ledger, and landscape
> composition. Reflow the title as needed so every word and the ampersand remain
> legible and balanced. Keep the subtitle exactly: “EVIDENCE FOR DECISIONS UNDER
> UNCERTAINTY.” Add no other text or watermark.

The selected result was resized to the Open Graph standard of 1200 × 630 pixels.
`tests/test_frontend_content.py` verifies the output format, dimensions, shared URL,
and matching alternate text.
