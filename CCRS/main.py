# main.py
# Entry point for CCRS system operations
# (FR-5.1, FR-5.2)

import logging
import os

import db
import model
import recommender

# Configure logging (FR-5.2)
logging.basicConfig(
    filename='ccrs.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def count_posts():
    """
    Count number of Content nodes in Neo4j (for retraining logic, FR-5.1)
    """
    data = db.fetch_data()
    return len(data)

def main():
    try:
        logging.info("CCRS system started.")
        db.setup_schema()
        db.import_csv(os.path.join('data', 'sample_data.csv'))
        recommender.performance_report()
        data = db.fetch_data()
        if len(data) < 50:
            print("Warning: fewer than 50 posts available — model may be less reliable.")
            logging.warning("Fewer than 50 posts: proceeding with training, but model quality may be poor.")
        model_obj, mean, std, feature_cols = model.train_model(data)
        # Recommendation for a sample topic
        msg, top, cand_df, top3 = recommender.recommend_for_topic('AI', model_obj, mean, std, feature_cols)
        print(msg)
        expl = recommender.explain_recommendation(top, model_obj, mean, std, feature_cols, data)
        print("Explanation:", expl)
        # Retrain if >10 new posts (FR-5.1)
        prev_count = len(data)
        new_count = count_posts()
        if new_count - prev_count > 10:
            logging.info("Detected >10 new posts. Retraining model.")
            data = db.fetch_data()
            model.train_model(data)
        logging.info("CCRS system completed successfully.")
    except Exception as e:
        logging.error(f"System error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

