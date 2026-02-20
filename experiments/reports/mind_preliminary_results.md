# Preliminary Results on MIND

## Executive Snapshot
- Articles processed: **10000**
- Source articles available: **51282**
- Articles with impressions: **10000**
- Total impressions: **5813624**
- Total clicks: **236344**
- Average CTR: **0.0389**
- Top-10 popularity cutoff (impressions): **30**

## Label Distribution
### Engagement
- Low: **7170**
- Medium: **2431**
- High: **399**

### Popularity
- Top_10: **5158**
- Other: **4842**

## Top Categories
- news: **3079**
- sports: **2177**
- finance: **692**
- lifestyle: **661**
- foodanddrink: **577**
- travel: **529**
- weather: **503**
- health: **449**

## Top Tone Buckets
- Formal: **8958**
- Instructional: **885**
- Casual: **157**

## Artifacts
- MeTTa facts: `experiments/atomspace_visualizer/public/data.metta`
- JSON stats: `experiments/reports/mind_preliminary_results.json`

## Caveats
- Labels are proxy labels from click logs (CTR/impression based), not editorial ground-truth.
- Heuristic fields (tone/sentiment/expertise) are rule-based and intended for preliminary benchmarking.