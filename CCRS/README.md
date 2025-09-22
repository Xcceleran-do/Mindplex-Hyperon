# Content Creator Recommender System (CCRS)

A modular Python system to help content creators analyze past content performance and receive data-driven recommendations for improving engagement.

## Features

- **Neo4j Integration:** Stores and queries content, topics, styles, and structures as a graph.
- **Performance Reporting:** Summarizes past engagement by topic, style, structure, and length.
- **Neural Network Model:** Predicts engagement for new content using TensorFlow.
- **Recommendations:** Suggests optimal content attributes for a given topic.
- **Explainability:** Uses SHAP to explain why a recommendation was made.
- **Logging:** All operations and errors are logged to `ccrs.log`.

## Project Structure

```
CCRS/
│
├── data/
│   └── sample_data.csv         # Sample input data (CSV)
├── models/
│   └── nn_model.h5             # Trained TensorFlow model (auto-generated)
├── db.py                       # Neo4j database operations
├── model.py                    # Neural network training and inference
├── recommender.py              # Performance reports and recommendations
├── main.py                     # System entry point
├── requirements.txt            # Python dependencies
├── .env                        # Neo4j connection credentials
├── ccrs.log                    # Log file (auto-generated)
└── README.md                   # This file
```

## Setup

1. **Clone the repository and navigate to `CCRS/`.**

2. **Install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Neo4j connection:**
   - Create a `.env` file in `CCRS/`:
     ```
     NEO4J_URI=bolt://localhost:7687
     NEO4J_USER=neo4j
     NEO4J_PASSWORD=your_password
     ```
   - Make sure Neo4j is running and accessible.

  Optional: configure engagement score weights in the same `.env` file. By default the system uses:

  ```
  ENG_WEIGHT_LIKES=1.0
  ENG_WEIGHT_SHARES=1.5
  ENG_WEIGHT_COMMENTS=2.0
  ENG_WEIGHT_VIEWS=0.05
  ```

  These are named, configurable weights applied when computing `engagement_score = likes*W_likes + shares*W_shares + comments*W_comments + views*W_views`. Avoid changing them unless you understand how they affect the target metric; they are configurable to let you match domain importance for interactions.

4. **Prepare your data:**
  - Place your CSV file (e.g., `sample_data.csv`) in the `data/` directory.
  - Ensure it has the following columns (a larger dataset improves model quality; ~50+ rows recommended):
     ```
     post_id,text,length,topic,style,structure,likes,shares,comments,views
     ```

## Usage

### Run the full workflow

```bash
python main.py
```

- Imports data, generates a performance report, trains the model, makes a recommendation, and explains it.

### Run individual modules

- **Import data and test Neo4j:**  
  `python db.py`
- **Generate performance report:**  
  `python recommender.py`
- **Train model and test prediction:**  
  `python model.py`

## Output

- **performance_report.csv:** Summary of past content performance.
- **models/nn_model.h5:** Trained neural network model.
- **ccrs.log:** Log file with all operations and errors.
- **Console output:** Recommendations and explanations.

## Example Recommendation Output

```
Past avg engagement for 'AI': 928.74.
Recommended: Length=500, Style=Casual, Structure=Narrative (Predicted: 1598.83 engagement).
Explanation (improved):
Baseline (avg): 928.74. Predicted: 1598.83. Delta: +670.09.
Top contributors:
  topic_AI: +299.1 (45% of top contributions)
  structure_Narrative: +295.5 (45% of top contributions)
  structure_List: -153.3 (23% of top contributions)
Recommendation: Recommendation increases predicted engagement mainly because the model favors Topic='AI' and Structure='Narrative' and disfavors Structure='List' — prefer Topic='AI', Structure='Narrative' over Structure='List'.
```

**Explanation format**

- **Baseline (avg):** average model prediction over the background samples (reference point).
- **Predicted:** model prediction for the recommended candidate.
- **Delta:** Predicted − Baseline (how much higher/lower the candidate is vs baseline).
- **Top contributors:** signed SHAP contributions with relative percentages to show importance among top contributors.
- **Recommendation sentence:** short human-friendly instruction derived from top contributors (what to prefer or avoid).

## Notes

- A larger dataset improves model quality; around 50+ posts is recommended but not required.
- All code is modular and well-documented.
- No UI is provided; all interaction is via the terminal and CSV files.

