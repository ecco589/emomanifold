# B1 Contextual Rewrite Protocol

> Methods subsection draft for RAL §IV.B1 — adversarial sarcasm stress test on iSarcasmEval SubTask C.

## Goal

Transform each original sarcasm–rephrase pair into a **contextual triplet** (sarcastic / sincere-positive / literal-negative) that preserves pragmatic intent while removing culture-specific knowledge barriers, so that valence readout is evaluated on **general situational understanding** rather than domain trivia.

## Input

- Source: iSarcasmEval SubTask C English test set (`task_C_En_test.csv`), 200 pairs.
- Each row provides `(text_0, text_1, sarcastic_id)`; we first identify which line is sarcastic and which is the non-sarcastic rephrase.

## Output schema (per item)

| Field | Description |
|-------|-------------|
| `id` | 1–200, aligned with CSV row order |
| `intent` | One-line summary of shared negative/positive pragmatic intent |
| `keywords` | 1–3 **emotion-bearing tokens** shared or contrastive across conditions (for token-level VAD) |
| `sarcastic` | Context + utterance: **positive surface, negative intent** |
| `positive` | Same situational frame: **positive surface, positive intent** (sincere) |
| `negative` | Same situational frame: **negative surface, negative intent** (literal rephrase) |

## Rewrite procedure (6 steps)

### Step 1 — Intent extraction (preserve semantics)

Read the original pair and write a single **intent label** (≤15 words) describing what the speaker truly feels, e.g. “frustrated that friend is late again”, “disappointed team lost”. This intent must be **identical** across all three rewritten sentences in the triplet.

**Do not** copy named entities or cultural references from the source; extract only the **situation + attitude**.

### Step 2 — De-culturation (lower knowledge barrier)

Replace culture-bound elements using this substitution table:

| Remove | Replace with |
|--------|----------------|
| Politicians, celebrities, parties | “the manager”, “our team leader”, “the company director” |
| Sports teams / players / leagues | “my team”, “the match”, “our group project” |
| Brands, shops, apps (Tesco, McDonald’s, TikTok) | “the shop”, “the café”, “the app”, “the delivery service” |
| Country-specific institutions (NHS, DWP, PMQs) | “the clinic”, “the benefits office”, “the staff meeting” |
| COVID-era jargon when not essential | “the outbreak rules”, “the health guidelines”, “staying home” |
| Place names (Accrington, Norfolk) | “my town”, “here”, “outside” |
| Slang / idioms requiring local knowledge | Plain English: “great job” (sarcastic), “of course”, “sure” |

**Rule:** A reader with **CEFR B1 English** and no UK/US news background should understand who did what and why the speaker is upset or pleased.

### Step 3 — Context injection (enable pragmatics)

Prepend **one situational sentence** (≤25 words) before the quoted utterance. The context must specify:

1. **Setting** — where/when (kitchen, bus stop, office, morning, before a meeting)
2. **Trigger** — observable event that makes sarcasm plausible (rain started, price shown, person arrived late)
3. **Optional speaker state** — tired, rushed, alone (only if implied by original)

Template:

> `[Setting + trigger]. [Speaker] said, "…"`

Both sarcastic and non-sarcastic versions **share the same context sentence**; only the quoted utterance changes.

### Step 4 — Triplet construction (three surface forms)

Given shared context + intent:

| Condition | Surface polarity | Intent | Construction rule |
|-----------|------------------|--------|-------------------|
| **Sarcastic** | Positive / enthusiastic words | Negative (same as intent) | Use praise, love, perfect, great, wonderful, etc. for a bad situation |
| **Positive** | Positive words | Positive | Same positive lexemes where possible, but situation trigger is ** genuinely good** |
| **Negative** | Negative words | Negative | Direct, literal expression of intent; no praise |

**Alignment constraints:**

- Sarcastic vs Positive: **maximize lexical overlap** on emotion keywords (e.g. both use “lovely weather”) so token-level comparison isolates **context**, not different words.
- Sarcastic vs Negative: **same intent**, opposite valence on surface.
- Length: each utterance 1–2 sentences; full item (context + quote) ≤ 60 words.

### Step 5 — Keyword anchoring (for token-level VAD)

Select 1–3 `keywords` per item:

1. **Primary:** highest-valence adjective/verb in the sarcastic/positive pair (`lovely`, `perfect`, `love`, `great`, `wonderful`, `fun`, `relaxing`, …).
2. **Secondary (optional):** negative counterpart appearing only in the negative condition (`awful`, `hate`, `terrible`, `frustrating`, …).
3. Prefer words that appear **verbatim** in both sarcastic and positive sentences.

These tokens are used for token-level V/A/D projection and Δ(S−P), Δ(S−N) analysis.

### Step 6 — Quality control checklist

Before accepting an item, verify:

- [ ] Intent identical across S / P / N
- [ ] No proper names, brands, or sport/politics trivia remain
- [ ] Context sentence present and shared across conditions
- [ ] Sarcastic surface is **more positive** than negative on naive word sentiment
- [ ] Positive condition describes a **plausible good outcome**, not sarcasm
- [ ] At least one keyword appears in both S and P
- [ ] Grammar and spelling normalized (minor fixes from source allowed)

## Example (item 3 in development set)

**Original pair:** “What a glorious weather today” (sarcastic) vs “the weather is gloomy…” (rephrase)

**Intent:** dislike current bad weather

| Condition | Text |
|-----------|------|
| Sarcastic | I stepped outside into cold rain and said, "What lovely weather we're having today!" |
| Positive | I stepped outside into warm sunshine and said, "What lovely weather we're having today!" |
| Negative | I stepped outside into cold rain and said, "This weather is awful — cold, wet, and miserable." |

**Keywords:** `lovely`, `awful`

## Analysis pairing (downstream)

- **Sentence-level:** last-token calibrated V on full text (S, P, N).
- **Token-level:** V/A/D at each `keyword` occurrence; report Δ(S−P) as context-sensitivity, Δ(S−N) as intent–surface decoupling.
- **Lexicon baseline:** NRC-VAD score of keyword (surface prior) vs projected V (model readout).

## Provenance

- Rewrites performed **manually** by authors following this protocol (not automated paraphrase).
- Original pair IDs retained for traceability; rewritten text may diverge substantially after Steps 2–3.
