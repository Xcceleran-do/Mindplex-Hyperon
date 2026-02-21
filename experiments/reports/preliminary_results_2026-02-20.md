# Preliminary Results Update — Pattern Mining on MIND-Derived Dataset

**Date:** 2026-02-21  
**Prepared by:** Sitotaw  
**Status:** Preliminary (descriptive; not causal)

## Executive Summary
We ran the mining pipeline on an **actual external dataset** (MIND-derived), with a working sample size of **1,001 records** and mined frequent patterns using **conjunction count = 5** and **min support = 20**. The strongest recurring structures are concentrated in **archived, short-form, formal news content**, which is predominantly associated with **Low engagement** despite frequent co-occurrence with **Top_10 popularity**.

## Scope and Method
- Data source: MIND-derived content transformed into our MeTTa schema.
- Records analyzed: **1,001**.
- Mining parameters: **conjunction count = 5**, **min support = 20**.
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

Confidence note (probability-style weighting):
Even within the same label, we will treat confidence as lower when CTR is close to a threshold and higher when CTR is farther above it. For example, an article at `CTR = 0.15` is still High, but it gets lower confidence than an article at `CTR = 0.30`, even though both are labeled High. This will be implemented once we got into pln based chainers.

## Key Findings

### 1) Low engagement patterns dominate the highest-support rules
Strongest support values in the output are mostly tied to **Low engagement**.

Representative high-support examples:
- `(authored-by MIND, content-type News, date-period Archived, popularity Top_10, engagement Low)` → **support 42**
- `(content-type News, date-period Archived, popularity Top_10, reading-time Short, engagement Low)` → **support 42**
- `(content-type News, date-period Archived, tone Formal, engagement Low)` → **support 34**
- `(audience-sentiment Mixed, authored-by MIND, content-type News, date-period Archived, engagement Low)` → **support 36**

### 2) Medium engagement appears but with notably lower support
Medium-engagement rules are present but generally weaker than low-engagement counterparts.

Representative examples:
- `(authored-by MIND, content-type News, date-period Archived, popularity Top_10, engagement Medium)` → **support 29**
- `(content-type News, date-period Archived, popularity Top_10, reading-time Short, engagement Medium)` → **support 29**
- `(content-type News, date-period Archived, tone Formal, engagement Medium)` → **support 28**

### 3) High engagement exists but is sparse
High-engagement patterns are comparatively rare in this run and do not dominate the top-support cluster.

### 4) Category-level signal
Category-anchored rules appear, but the highest supports are driven by cross-category structural combinations rather than a single category label.

## LLM Interpretation of Mined Patterns
The following narrative summarizes the model analysis of mined patterns (kept as descriptive guidance, not causal claims):

Based on the mined patterns, content targeting beginner audiences with mixed sentiment shows a strong and consistent trend: "Archived" date-period, "News" content-type, "Formal" tone, and "Short" reading-time are all individually associated with lower engagement, especially when combined with otherwise positive signals like "Top_10" popularity. In fact, high support counts (e.g., [106], [116], [126], [140] with support = 42) reveal that even popular or widely recognized content fails to drive engagement if it is short, formal, and labeled as archived.

However, a clear actionable pattern emerges: when any positive attribute is paired with "Top_10" popularity and "Short" reading-time, engagement shifts from Low to Medium, but only if the content avoids a formal tone. The presence of a formal tone consistently drags engagement down, even when all other factors are favorable (see [23], [70], [104], [136], [150]).

For example, "Top_10" popularity + "Short" reading-time + "Formal" tone almost guarantees low engagement ([40], [80], [116], [150]), but when tone shifts away from formal, engagement consistently climbs to Medium ([25], [43], [83], [121], [137], [151]). This suggests that beginner audiences may perceive formal tone as a barrier, even when the content is easy to read and timely.

Interestingly, archived content performs poorly regardless of topic or author, but if it is also formal and short, the effect is amplified ([106], [126], [140]). Yet, when archived but popular and short, and tone is not formal, engagement improves ([61], [131], [145]).

Key insights and recommendations:
- Avoid pairing "Formal" tone with short, archived, or even high-popularity content for beginners; it depresses engagement ([70], [110], [120]).
- Prioritize conversational or neutral tones even in news-style content; this is the difference between low and medium engagement ([72], [78], [123]).
- Popularity ("Top_10") and brevity ("Short") are not enough; if the tone is formal, engagement stays low ([80], [104]).
- Use archival status as a red flag: unless content is revised with friendly tone and clear value, it will likely underperform, especially if formally written ([106], [116]).

Final takeaway: for beginner audiences with mixed sentiment, tone matters more than timing or popularity. A short, popular, and timely article can still underperform if it feels too formal; a non-formal rewrite of the same content can lift engagement.

Note: bracketed references (e.g., [106], [116]) are included to verify the LLM response against hallucination by cross-checking specific mined-pattern positions.

## Backward Chainer Check (Consistency)
We validated one example with the backward chainer to confirm the mined rules can explain a concrete article label:

- Query: `(engagement A_N41020 "Medium")`
- Depth used: **5**
- Proofs found: **66**

Example proofs include rules that combine facts such as `audience-sentiment = Mixed`, `authored-by = MIND`, `content-type = News`, and `date-period = Archived` to derive the engagement label.

### LLM Explanation for A_N41020 (Why Medium Engagement)
The model explanation mirrors the backward-chainer output and highlights multiple proof paths. Summary:

- Direct fact: `(engagement A_N41020 "Medium")` is present in the fact base.
- Multiple rule paths: combinations of facts such as `audience-sentiment = Mixed`, `authored-by = MIND`, `content-type = News`, `date-period = Archived`, `popularity = Top_10`, `reading-time = Short`, and `tone = Formal` each trigger rules that conclude Medium engagement.
- The result is a large set of overlapping proofs (66), indicating robust rule support for the label rather than a single fragile rule.

## Facts Used (Sample)
Rules were generated from the exported fact base (`experiments/chainer/rules.metta`). Example facts for `A_N41020` include:

- `audience-expertise = Beginner`
- `content-type = News`
- `date-period = Archived`
- `engagement = Medium`
- `authored-by = MIND`

## Preliminary Interpretation
- In this sample, **Top_10 popularity does not consistently map to high engagement**.
- The most common content shape is “archived + short + formal + informational/news,” and that shape is strongly associated with **low engagement**.
- There may be strong distribution effects (content mix, time-period skew, labeling proxies) that should be controlled in the next phase.

## Caveats
- Results are **preliminary** and based on support-frequency associations.
- Engagement labels are proxy-derived and should not be interpreted as causal outcomes.
- Pattern overlap/redundancy is expected (many near-duplicate conjunctions).

## Conclusion
We now have credible preliminary evidence from an actual dataset run. The strongest mined structure is a robust low-engagement cluster around archived, short, formal news content. Next, we should move from descriptive pattern frequency to validated and actionable pattern scoring.
