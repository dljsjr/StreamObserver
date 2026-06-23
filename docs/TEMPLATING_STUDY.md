# TEMPLATING_STUDY.md — why the interjections felt "templated," and the fix

**Symptom (user, watching the Herzog demo over a long stretch):** every aside marched to the same
shape — `The <noun>. <fragment>. <grave void-elaboration>.` — and felt structurally predictable. A
first instinct (turn temperature *down*) was the wrong direction; this is the controlled study that
found the real cause.

## Apparatus (non-destructive, one variable per arm)

`herzog.txt` stayed the **control**; every lever became an independent knob so each hypothesis is
testable in isolation (the control path is byte-identical to the original ask):
- **H1 (persona prompt)** → sibling files: `herzog_norecipe.txt` (drop the "small move → overwhelming"
  recipe), `herzog_broad.txt` (broaden theme beyond the void), `herzog_varyform.txt` (control + "vary
  your openings, never a fixed formula").
- **H2 (ask framing)** → `--ask-mode {passage|continuous}`.
- **H3 (temperature)** → `--interject-temp`.
- **H4 (novelty memory)** → `--novelty {fresh|form|off}`.

Why it's clean: the **surprisal pass is untouched by interjection settings**, so every arm fires on
the *identical spans* — only the one varied generation variable differs. `--dedup 0` shows the raw
generation; constant-seeded RNG → reproducible. (Code: `AskMode`/`NoveltyMode` in `src/lobe.rs`;
`interject_ask_context` branches on them; flags in `src/main.rs`.)

## Round 1 results (8 arms, E4B, ~22k-token Moby Dick chunk)

**The metric lied.** "Distinct 3-word opening" was **7/7 in every arm** (openings literally always
differ) yet control is rigidly templated. Only *reading the asides* revealed the shape — vindicating
the [[interjection-eval-is-judgment-based]] rule.

**Refuted as causes:**
- **H1a (remove recipe):** template persists. Not the recipe.
- **H1b (broaden theme):** the model snapped right back to "the void/abyss" regardless. The **"Werner
  Herzog" name anchors the theme**, not the prompt's adjectives.
- **H2 (continuous framing):** swapped `The`→`A`, same shape. Refuted.

**Two separable problems:**
1. **Staccato noun-fragment *shape*** — a deep stylistic default of E4B's dramatic register; survives
   every structural/theme/framing change. Partially loosened by explicit form-variety instruction
   (H1c, H4a) and removing self-examples (H4b); mildly by temp (H3). No single lever cures it.
2. **Void *monotony*** — driven by the persona anchor (the name), independent of the shape.

**What moved it (by reading):** H4a (form-framed novelty) > H1c (vary-form prompt) > H4b (novelty off)
> H3 (temp). The original instinct (temp) was a real but minor lever — and inverted.

## The decision: `herzog_varyform.txt` @ temp 0.6

A 4-arm depth read (original / 3-mover-combo / varyform@0.7 / varyform@0.6) settled it. The combo
(varyform + form-novelty + temp 0.9) had the most variety but went **shallow** — and at 0.9 even
glitched ("thump"→"thumb"). The key principle:

> **Temperature owns depth; the prompt owns structure.** Low temp = commitment = the rich,
> develop-one-image depth (the "butter to fowl" quality). The vary-openings *prompt* nudge breaks the
> lockstep at ~zero depth cost. You get both by keeping temp LOW and nudging form in the PROMPT —
> never by cranking temp (that buys variety by spending the voice).

**Shipped:** `--preamble-file personas/herzog_varyform.txt --interject-temp 0.6` (the demo command in
CLAUDE.md). Original-grade depth, lockstep broken.

## Round 2 results (3 arms at temp 0.6 vs the varyform@0.6 baseline)

- **H5 — `herzog_flowing.txt` (anti-fragment): a real RHYTHM win.** It actually broke the staccato that
  survived *every* Round-1 arm — output became connected prose ("…these little anchors of despair—the
  grim mouth, the drizzly November within—they are merely the predictable tremors before the
  inevitable…") instead of `The X. A Y. The Z.` fragments. Two tradeoffs: (a) flowing = longer
  generation → **fewer asides** (4 vs 7 over the same chunk; longer asides occupy more stream-ticks, so
  some fires are skipped while one is in flight) — which actually *aligns* with the "sparse, lit-class
  cadence" preference; (b) a faint NEW template crept in ("To find … is to …" opened two of four), and
  the void vocabulary persists. Net: a genuinely distinct, smoother/essayistic register — a taste call
  vs the baseline's retained staccato punch. Kept as an alternate voice.
- **H6 — `herzog_deanchored.txt` (drop the name "Werner Herzog"): NO.** The void monotony **persisted
  unchanged** ("slow rot", "indifferent gravity", "vast, indifferent ocean"). Conclusion: the monotony
  comes from the **descriptive adjectives** (grave/fatalistic/indifference/abyss), NOT the name — H1b
  (broaden adjectives) failed AND H6 (keep adjectives, drop name) failed, so the void is *intrinsic to
  the doom-sensibility*. You can't broaden it without ceasing to be Herzog. So: **don't chase the
  monotony — it IS the character.** Removing the name only cost the "it's-Herzog" framing. Dropped.

**Decision unchanged:** ship `herzog_varyform.txt` @ 0.6 (varied openings, retained punch, good
cadence). Offer `herzog_flowing.txt` @ 0.6 as the smoother/essayistic alternate. Leave the void
monotony alone (it's the persona). H6 retired.
