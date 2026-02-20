# Preliminary Results Update — Pattern Mining on MIND-Derived Dataset

**Date:** 2026-02-20  
**Prepared by:** Sitotaw  
**Status:** Preliminary (descriptive; not causal)

## Executive Summary
We ran the mining pipeline on an **actual external dataset** (MIND-derived), with a working sample size of **49,998 records** and mined **7,600 frequent patterns**. The strongest recurring structures are concentrated in **archived, short-form, formal news content**, which is predominantly associated with **Low engagement** despite frequent co-occurrence with **Top_10 popularity**.

## Scope and Method
- Data source: MIND-derived content transformed into our MeTTa schema.
- Records analyzed: **49,998**.
- Frequent patterns mined: **7,600**.
- Pattern support shown as raw occurrence count.
- This phase is association mining only (no causal inference).

## Engagement Labeling Rule (Explicit)
Engagement is computed from click-through behavior in MIND impression logs:

- **CTR(article)** = `clicks / impressions`
	- `impressions`: how many times the article appeared in behavior logs
	- `clicks`: count of impression tokens with label `1` for that article

Thresholds used in this run:
- **Low**: `CTR < 0.05`
- **Medium**: `0.05 <= CTR <= 0.15`
- **High**: `CTR > 0.15`

Example:
- If an article appears 100 times and is clicked 7 times, `CTR = 0.07`, so engagement = **Medium**.

This is a deterministic proxy-labeling rule used for preliminary benchmarking.

## Key Findings

### 1) Low engagement patterns dominate the highest-support rules
Strongest support values in the output are mostly tied to **Low engagement**.

Representative high-support examples:
- `(content-type News, date-period Archived, popularity Top_10, reading-time Short, engagement Low)` → **support 3,124**
- `(content-type News, date-period Archived, popularity Top_10, tone Formal, engagement Low)` → **support 2,710**
- `(content-type News, date-period Archived, popularity Top_10, primary-goal Inform, engagement Low)` → **support 2,187**
- `(date-period Archived, popularity Top_10, primary-goal Inform, tone Formal, engagement Low)` → **support 1,888**

### 2) Medium engagement appears but with notably lower support
Medium-engagement rules are present but generally weaker than low-engagement counterparts.

Representative examples:
- `(content-type News, date-period Archived, popularity Top_10, reading-time Short, engagement Medium)` → **support 676**
- `(content-type News, date-period Archived, popularity Top_10, tone Formal, engagement Medium)` → **support 628**
- `(date-period Archived, popularity Top_10, primary-goal Inform, tone Formal, engagement Medium)` → **support 366**

### 3) High engagement exists but is sparse
High-engagement patterns are comparatively rare in this run.

Representative examples:
- `(content-type News, date-period Archived, popularity Top_10, reading-time Short, engagement High)` → **support 44**
- `(content-type News, date-period Archived, popularity Top_10, primary-goal Inform, engagement High)` → **support 24**
- `(content-type News, date-period Archived, popularity Top_10, primary-goal Entertain, engagement High)` → **support 20**

### 4) Category-level signal: Weather and Video
- **Weather**: Repetitive low-engagement structures (typically support ~73–78 for low, ~14–16 for medium).  
- **Video**: Frequent low-engagement clusters in short/entertainment/formal combinations, with smaller medium-engagement pockets.

## Preliminary Interpretation
- In this sample, **Top_10 popularity does not consistently map to high engagement**.
- The most common content shape is “archived + short + formal + informational/news,” and that shape is strongly associated with **low engagement**.
- There may be strong distribution effects (content mix, time-period skew, labeling proxies) that should be controlled in the next phase.

## Caveats
- Results are **preliminary** and based on support-frequency associations.
- Engagement labels are proxy-derived and should not be interpreted as causal outcomes.
- Pattern overlap/redundancy is expected (many near-duplicate conjunctions).

## Recommended Next Steps (1–2 week window)
1. **De-duplicate and rank rules** by support + lift/confidence to reduce redundancy.
2. **Holdout validation**: verify top patterns on a separate split for stability.
3. **Actionability shortlist**: extract top 10 segments for editorial/product experiments.
4. **Comparative baseline**: report against overall engagement base rates to quantify uplift.

## Conclusion
We now have credible preliminary evidence from an actual dataset run. The strongest mined structure is a robust low-engagement cluster around archived, short, formal news content. Next, we should move from descriptive pattern frequency to validated and actionable pattern scoring.
