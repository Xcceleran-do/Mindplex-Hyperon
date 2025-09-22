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
Explanation: Top factors: topic_AI (+299.1), structure_Narrative (+295.5), structure_List (-153.3)
```

## Notes

- A larger dataset improves model quality; around 50+ posts is recommended but not required.
- All code is modular and well-documented.
- No UI is provided; all interaction is via the terminal and CSV files.

