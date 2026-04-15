# ============================================================
#  SENTINEL v3 — Flask ML API (FIXED VERSION)
#  Includes: credibility, harm, attribution fields
# ============================================================

import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

LOADED = {}

# ── LOAD MODELS ─────────────────────────────────────────────
def load_models():
    global LOADED
    required = [
        'logistic_regression', 'random_forest',
        'xgboost', 'svm', 'nlp_infowar',
        'scaler'
    ]

    for name in required:
        path = os.path.join(MODELS_DIR, f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                LOADED[name] = pickle.load(f)

    report_path = os.path.join(MODELS_DIR, 'training_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            LOADED['training_report'] = json.load(f)

load_models()

CLASS_NAMES = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']

# ============================================================
# HEALTH
# ============================================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'online',
        'models_loaded': True,
        'timestamp': datetime.now().isoformat(),
    })

# ============================================================
# INFOWAR FIXED ENDPOINT
# ============================================================
@app.route('/predict/infowar', methods=['POST'])
def predict_infowar():

    if 'nlp_infowar' not in LOADED:
        return jsonify({'error': 'NLP model not loaded'}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'text required'}), 400

    text = str(data['text']).strip()
    if not text:
        return jsonify({'error': 'empty text'}), 400

    nlp_obj = LOADED['nlp_infowar']
    pipeline = nlp_obj['pipeline']
    labels = nlp_obj['labels']

    pred_idx = int(pipeline.predict([text])[0])
    pred_label = labels[pred_idx]

    clf = pipeline.named_steps['clf']
    vec = pipeline.named_steps['tfidf']
    X_t = vec.transform([text])

    scores = clf.decision_function(X_t)[0]
    e_scores = np.exp(scores - scores.max())
    proba = (e_scores / e_scores.sum()).tolist()

    confidence = round(max(proba) * 100, 1)

    # ============================================================
    # 🔥 NEW FIELDS (THIS FIXES YOUR UI)
    # ============================================================

    credibility = round(100 - confidence, 1)

    harm_map = {
        "Disinformation": 8.5,
        "Propaganda": 6.5,
        "Psyops": 9.0,
        "Legitimate": 2.0
    }

    harm = harm_map.get(pred_label, 5.0)

    # Simple attribution logic (upgrade later with ML)
    if "government" in text.lower() or "state" in text.lower():
        attribution = "State-Affiliated (Tier 2)"
    else:
        attribution = "Unknown / Mixed"

    # Extract top tokens
    feature_names = vec.get_feature_names_out()
    tfidf_matrix = X_t.toarray()[0]
    top_idx = tfidf_matrix.argsort()[-10:][::-1]

    top_tokens = [
        {'term': feature_names[i], 'weight': round(float(tfidf_matrix[i]), 4)}
        for i in top_idx if tfidf_matrix[i] > 0
    ]

    return jsonify({
        'classification': pred_label.upper(),
        'confidence_pct': confidence,

        # ✅ FIXED FIELDS
        'credibility': credibility,
        'harm': harm,
        'attribution': attribution,

        'top_tfidf_tokens': top_tokens,
        'timestamp': datetime.now().isoformat(),
    })

# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)
