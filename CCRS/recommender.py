# recommender.py
# Performance reports and recommendations for CCRS
# (FR-2.1, FR-4.1, FR-4.2)

import logging
import pandas as pd
import numpy as np
import shap
import db
import model

# Configure logging (FR-5.2)
logging.basicConfig(
    filename='ccrs.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def performance_report():
    """
    Generate a performance report of past content (FR-2.1).
    - Groups by topic.
    - Calculates average engagement, best style, best structure, and best length range.
    - Prints table and exports to CSV.
    """
    logging.info("Generating performance report.")
    data = db.fetch_data()
    if not data:
        print("No data available for report.")
        logging.warning("No data for performance report.")
        return
    df = pd.DataFrame(data)
    report = []
    for topic, group in df.groupby('topic'):
        avg_eng = group['engagement_score'].mean()
        length_min = group['length'].min()
        length_max = group['length'].max()
        best_style = group['style'].mode()[0] if not group['style'].mode().empty else ''
        best_structure = group['structure'].mode()[0] if not group['structure'].mode().empty else ''
        report.append({
            'Topic': topic,
            'Avg Engagement': round(avg_eng, 2),
            'Best Style': best_style,
            'Best Structure': best_structure,
            'Best Length Range': f"{length_min}-{length_max} words"
        })
    rep_df = pd.DataFrame(report)
    print(rep_df.to_string(index=False))
    rep_df.to_csv('performance_report.csv', index=False)
    logging.info("Performance report generated and saved to performance_report.csv.")

def recommend_for_topic(topic, model_obj, mean, std, feature_cols):
    """
    Recommend optimal content attributes for a topic (FR-4.1).
    Returns formatted string with recommendation, top config, candidate DataFrame, and top 3 configs.
    """
    logging.info(f"Generating recommendation for topic: {topic}")
    # Candidate configurations
    lengths = [500, 1000, 1500]
    styles = ['Casual', 'Formal']
    structures = ['List', 'Narrative']
    candidates = []
    for l in lengths:
        for s in styles:
            for st in structures:
                candidates.append({'length': l, 'topic': topic, 'style': s, 'structure': st})
    cand_df = pd.DataFrame(candidates)
    preds = model.predict_engagement(cand_df, mean, std, feature_cols)
    cand_df['predicted_engagement'] = preds
    top = cand_df.sort_values('predicted_engagement', ascending=False).iloc[0]
    # Past avg engagement
    past_data = db.fetch_data(topic_filter=topic)
    if past_data:
        avg_past = np.mean([d['engagement_score'] for d in past_data])
    else:
        avg_past = None
    msg = f"Past avg engagement for '{topic}': {avg_past:.2f}.\n" if avg_past else f"No past data for '{topic}'.\n"
    msg += f"Recommended: Length={top['length']}, Style={top['style']}, Structure={top['structure']} (Predicted: {top['predicted_engagement']:.2f} engagement)."
    logging.info(f"Recommendation for {topic}: {msg}")
    return msg, top, cand_df, cand_df[['length', 'topic', 'style', 'structure', 'predicted_engagement']].sort_values('predicted_engagement', ascending=False).head(3)

def explain_recommendation(top_config, model_obj, mean, std, feature_cols, train_data):
    """
    Explain recommendation using SHAP (FR-4.2).
    Returns explanation string.
    """
    logging.info("Generating SHAP explanation.")
    # Prepare background data (training set, normalized, one-hot)
    X_train = pd.DataFrame(train_data)[['length', 'topic', 'style', 'structure']]
    X_train = pd.get_dummies(X_train, columns=['topic', 'style', 'structure'])
    for col in feature_cols:
        if col not in X_train.columns:
            X_train[col] = 0
    X_train = X_train[feature_cols]
    X_train, _, _ = model._normalize(X_train, mean, std)
    X_train = X_train.values.astype(np.float32)  # Ensure correct dtype

    # Prepare input
    x = pd.DataFrame([top_config])[['length', 'topic', 'style', 'structure']]
    x = pd.get_dummies(x, columns=['topic', 'style', 'structure'])
    for col in feature_cols:
        if col not in x.columns:
            x[col] = 0
    x = x[feature_cols]
    x, _, _ = model._normalize(x, mean, std)
    x = x.values.astype(np.float32)  # Ensure correct dtype

    # SHAP KernelExplainer (uses model's predict function)
    loaded_model = model_obj if hasattr(model_obj, 'predict') else model.tf.keras.models.load_model(model.MODEL_PATH, compile=False)
    background_size = min(50, X_train.shape[0])
    explainer = shap.KernelExplainer(lambda v: loaded_model.predict(v), X_train[:background_size])
    shap_values = explainer.shap_values(x)
    # Normalize shap_values to a 1-D array of floats corresponding to feature_cols
    if isinstance(shap_values, list):
        sv = np.array(shap_values[0])
    else:
        sv = np.array(shap_values)
    sv = np.squeeze(sv)
    if sv.ndim > 1:
        sv = sv.ravel()
    # Ensure we have one value per feature
    sv = sv[:len(feature_cols)]
    sv = [float(v) for v in sv]
    shap_contrib = dict(zip(feature_cols, sv))

    # Compute baseline (expected value) as mean prediction over the background
    try:
        preds_bg = np.array(loaded_model.predict(X_train[:background_size]))
        baseline = float(np.mean(preds_bg.ravel()))
    except Exception:
        baseline = 0.0
    # Predicted value for the input x
    try:
        preds_x = np.array(loaded_model.predict(x))
        predicted = float(preds_x.ravel()[0])
    except Exception:
        predicted = 0.0
    delta = predicted - baseline
    # Select top features by absolute SHAP value
    top_feats = sorted(shap_contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    # Build human-friendly lines: feature (contribution, percent-of-delta)
    contrib_lines = []
    total_abs = float(sum(abs(float(v)) for _, v in top_feats))
    if total_abs == 0:
        total_abs = 1.0
    for k, v in top_feats:
        v_f = float(v)
        percent = (abs(v_f) / total_abs) * 100
        contrib_lines.append(f"{k}: {v_f:+.1f} ({percent:.0f}% of top contributions)")
    explanation = (
        f"Baseline (avg): {baseline:.2f}. Predicted: {predicted:.2f}. Delta: {delta:+.2f}.\n"
        f"Top contributors:\n  " + "\n  ".join(contrib_lines)
    )

    # Build a short recommendation sentence from top features.
    def pretty(name):
        if name.startswith('topic_'):
            return f"Topic='{name.split('_',1)[1]}'"
        if name.startswith('style_'):
            return f"Style='{name.split('_',1)[1]}'"
        if name.startswith('structure_'):
            return f"Structure='{name.split('_',1)[1]}'"
        return name

    pos = {'topic': [], 'style': [], 'structure': [], 'other': []}
    neg = {'topic': [], 'style': [], 'structure': [], 'other': []}
    for k, v in top_feats:
        if k.startswith('topic_'):
            (pos if v > 0 else neg)['topic'].append((k, v))
        elif k.startswith('style_'):
            (pos if v > 0 else neg)['style'].append((k, v))
        elif k.startswith('structure_'):
            (pos if v > 0 else neg)['structure'].append((k, v))
        else:
            (pos if v > 0 else neg)['other'].append((k, v))

    picks = []
    avoids = []
    for grp in ('topic', 'style', 'structure'):
        if pos[grp]:
            # take top positive contributor
            picks.append(pretty(sorted(pos[grp], key=lambda x: -x[1])[0][0]))
        if neg[grp]:
            avoids.append(pretty(sorted(neg[grp], key=lambda x: x[1])[0][0]))

    if picks and avoids:
        rec = f"Recommendation increases predicted engagement mainly because the model favors {', '.join(picks)} and disfavors {', '.join(avoids)} — prefer {', '.join(picks)} over {', '.join(avoids)}."
    elif picks:
        rec = f"Recommendation increases predicted engagement mainly because the model favors {', '.join(picks)}. Consider preferring {', '.join(picks)}."
    elif avoids:
        rec = f"Recommendation increases predicted engagement mainly because the model disfavors {', '.join(avoids)}. Consider avoiding {', '.join(avoids)}."
    else:
        # fallback: mention top overall features
        top_names = [pretty(k) for k, _ in top_feats[:3]]
        rec = f"Recommendation driven by top features: {', '.join(top_names)}."

    explanation = explanation + "\n" + "Recommendation: " + rec
    logging.info(f"SHAP explanation: {explanation}")
    return explanation

if __name__ == "__main__":
    # Example usage for manual testing
    from db import fetch_data
    data = fetch_data()
    if not data:
        print("No data available for recommendations.")
    else:
        if len(data) < 50:
            print("Warning: fewer than 50 posts available — recommendations may be less reliable.")
        model_obj, mean, std, feature_cols = model.train_model(data)
        msg, top, cand_df, top3 = recommend_for_topic('AI', model_obj, mean, std, feature_cols)
        print(msg)
        expl = explain_recommendation(top, model_obj, mean, std, feature_cols, data)
        print("Explanation:", expl)

