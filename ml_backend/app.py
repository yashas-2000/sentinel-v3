"""
============================================================
 SENTINEL v3 — Flask ML API
 app.py

 Serves real ML predictions from trained scikit-learn models.
 The Node.js backend calls this API; it does NOT call Claude.

 Endpoints:
   GET  /health               — server status
   POST /predict/risk         — risk classification (RF/XGB/SVM/LR)
   POST /predict/infowar      — NLP classification (TF-IDF + SVM)
   POST /predict/buildup      — military buildup score
   GET  /models/info          — accuracy metrics for all models
   GET  /models/features      — feature importance from RF + XGBoost

 Start:  python app.py
 Port:   5001 (configurable via ML_PORT env variable)
============================================================
"""

import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)   # Allow calls from Node.js backend on localhost

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ── LOAD MODELS ON STARTUP ───────────────────────────────────
LOADED = {}

def load_models():
    """Load all trained .pkl models into memory at startup."""
    global LOADED
    required = [
        'logistic_regression', 'random_forest',
        'xgboost', 'svm', 'nlp_infowar',
        'scaler', 'feature_cols'
    ]
    missing = []
    for name in required:
        path = os.path.join(MODELS_DIR, f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                LOADED[name] = pickle.load(f)
            print(f"  ✓ Loaded {name}.pkl")
        else:
            missing.append(name)

    if missing:
        print(f"\n  ⚠ Missing model files: {missing}")
        print("  Run: python train_models.py  first!\n")
    else:
        print(f"\n  ✓ All {len(required)} models loaded successfully\n")

    # Load training report for metrics
    report_path = os.path.join(MODELS_DIR, 'training_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            LOADED['training_report'] = json.load(f)

load_models()

# ── FEATURE COLUMNS (ordered) ────────────────────────────────
FEATURE_COLS = [
    'mil_exp_usd_bn', 'mil_gdp_pct', 'arms_import_index', 'psi',
    'acled_events_6mo', 'fatalities_12mo', 'battle_ratio', 'vac_ratio',
    'gdp_usd_tn', 'budget_yoy_pct', 'mobilization', 'cyber_level', 'nuclear_posture'
]
CLASS_NAMES = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
NLP_LABELS  = ['Legitimate', 'Propaganda', 'Disinformation', 'Psyops']


def models_ready():
    """Check all risk models are loaded."""
    needed = ['logistic_regression', 'random_forest', 'xgboost', 'svm', 'scaler']
    return all(k in LOADED for k in needed)


def build_feature_vector(data):
    """
    Build a numpy feature vector from request JSON.
    Applies defaults for missing fields.
    Returns (vector, warnings_list)
    """
    warnings = []
    vec = []

    defaults = {
        'mil_exp_usd_bn':    20.0,
        'mil_gdp_pct':       3.0,
        'arms_import_index': 4.0,
        'psi':               -0.5,
        'acled_events_6mo':  50,
        'fatalities_12mo':   500,
        'battle_ratio':      0.4,
        'vac_ratio':         0.25,
        'gdp_usd_tn':        1.0,
        'budget_yoy_pct':    5.0,
        'mobilization':      2,
        'cyber_level':       2,
        'nuclear_posture':   1,
    }

    for col in FEATURE_COLS:
        val = data.get(col, defaults[col])
        try:
            val = float(val)
        except (ValueError, TypeError):
            warnings.append(f"Invalid value for '{col}', using default {defaults[col]}")
            val = float(defaults[col])
        vec.append(val)

    return np.array(vec).reshape(1, -1), warnings


# ══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'online',
        'system': 'SENTINEL v3 ML API',
        'models_loaded': models_ready(),
        'model_count': len([k for k in LOADED if k not in ('scaler','feature_cols','training_report')]),
        'timestamp': datetime.now().isoformat(),
    })


@app.route('/models/info', methods=['GET'])
def models_info():
    """Return accuracy metrics for all trained models."""
    if 'training_report' not in LOADED:
        return jsonify({'error': 'Training report not found. Run train_models.py first.'}), 404

    report = LOADED['training_report']
    summary = {}
    for mname, mdata in report.get('models', {}).items():
        summary[mname] = {
            'accuracy':    mdata.get('accuracy'),
            'macro_f1':    mdata.get('macro_f1'),
            'cv_f1_mean':  mdata.get('cv_f1_mean'),
            'cv_f1_std':   mdata.get('cv_f1_std'),
        }

    return jsonify({
        'models': summary,
        'dataset': report.get('dataset', {}),
        'generated_at': report.get('generated_at'),
    })


@app.route('/models/features', methods=['GET'])
def feature_importance():
    """Return feature importance from Random Forest and XGBoost."""
    result = {}

    if 'random_forest' in LOADED:
        rf_obj = LOADED['random_forest']['model']
        fi = dict(zip(FEATURE_COLS, rf_obj.feature_importances_.tolist()))
        result['random_forest'] = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    if 'xgboost' in LOADED:
        xgb_obj = LOADED['xgboost']['model']
        fi = dict(zip(FEATURE_COLS, xgb_obj.feature_importances_.tolist()))
        result['xgboost'] = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    return jsonify({'feature_importance': result})


@app.route('/predict/risk', methods=['POST'])
def predict_risk():
    """
    Run risk classification using the selected model.
    
    Request body (JSON):
    {
        "model": "xgboost",        // xgboost | random_forest | svm | logistic_regression
        "mil_exp_usd_bn": 68.5,
        "mil_gdp_pct": 14.67,
        "arms_import_index": 7.8,
        "psi": -1.4,
        "acled_events_6mo": 312,
        "fatalities_12mo": 12400,
        "battle_ratio": 0.50,
        "vac_ratio": 0.27,
        "gdp_usd_tn": 3.5,
        "budget_yoy_pct": 24.3,
        "mobilization": 3,
        "cyber_level": 3,
        "nuclear_posture": 2
    }
    """
    if not models_ready():
        return jsonify({'error': 'Models not loaded. Run train_models.py first.'}), 503

    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    model_name = data.get('model', 'xgboost').lower()
    if model_name not in LOADED:
        return jsonify({'error': f"Model '{model_name}' not available. Choose from: xgboost, random_forest, svm, logistic_regression"}), 400

    X, warns = build_feature_vector(data)

    model_obj  = LOADED[model_name]
    model      = model_obj['model']
    scaler     = LOADED['scaler']
    needs_sc   = model_obj.get('needs_scaling', False)

    X_input = scaler.transform(X) if needs_sc else X

    # Prediction
    pred_class   = int(model.predict(X_input)[0])
    class_label  = CLASS_NAMES[pred_class]

    # Probability (if available)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_input)[0].tolist()
        confidence = round(max(proba) * 100, 1)
        proba_dict = {CLASS_NAMES[i]: round(p * 100, 1) for i, p in enumerate(proba)}
    else:
        # SVM with LinearSVC doesn't have predict_proba — use decision function
        proba_dict  = {c: 0.0 for c in CLASS_NAMES}
        proba_dict[class_label] = 100.0
        confidence  = 100.0

    # Compute a continuous risk score (weighted sum)
    weights = np.array([0.12, 0.10, 0.18, 0.15, 0.12, 0.08, 0.06, 0.05, 0.03, 0.08, 0.05, 0.04, 0.04])
    raw = X[0]
    norm_vec = np.array([
        raw[0] / 200,           # mil_exp
        raw[1] / 40,            # mil_gdp_pct
        raw[2] / 10,            # arms_import
        (-raw[3] + 2.5) / 5,   # psi (inverted)
        raw[4] / 600,           # acled
        raw[5] / 40000,         # fatalities
        raw[6],                 # battle_ratio
        raw[7],                 # vac_ratio
        0.5,                    # gdp (neutral)
        min(raw[9], 60) / 60,  # budget_yoy
        raw[10] / 4,            # mobilization
        raw[11] / 4,            # cyber
        raw[12] / 4,            # nuclear
    ])
    risk_score = float(np.clip(np.dot(weights, np.clip(norm_vec, 0, 1)) * 100, 0, 100))

    # Feature importance for this model
    fi_out = {}
    if hasattr(model_obj['model'], 'feature_importances_'):
        fi = dict(zip(FEATURE_COLS, model_obj['model'].feature_importances_.tolist()))
        fi_out = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    # Get training accuracy from report
    train_acc = None
    if 'training_report' in LOADED:
        train_acc = LOADED['training_report']['models'].get(model_name, {}).get('accuracy')

    return jsonify({
        'model_used':       model_name,
        'risk_class':       pred_class,
        'risk_label':       class_label,
        'risk_score':       round(risk_score, 1),
        'confidence_pct':   confidence,
        'class_probabilities': proba_dict,
        'feature_importance':  fi_out,
        'training_accuracy':   train_acc,
        'warnings':            warns,
        'timestamp':           datetime.now().isoformat(),
    })


@app.route('/predict/infowar', methods=['POST'])
def predict_infowar():
    """
    Classify information warfare content using TF-IDF + SVM.

    Request body:
    {
        "text": "Breaking: government hiding civilian casualties..."
    }
    """
    if 'nlp_infowar' not in LOADED:
        return jsonify({'error': 'NLP model not loaded. Run train_models.py first.'}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'JSON body with "text" field required'}), 400

    text = str(data['text']).strip()
    if not text:
        return jsonify({'error': 'text field cannot be empty'}), 400

    nlp_obj  = LOADED['nlp_infowar']
    pipeline = nlp_obj['pipeline']
    labels   = nlp_obj['labels']

    pred_idx   = int(pipeline.predict([text])[0])
    pred_label = labels[pred_idx]

    # Decision function scores (not probabilities but proportional)
    clf = pipeline.named_steps['clf']
    vec = pipeline.named_steps['tfidf']
    X_t = vec.transform([text])

    scores = clf.decision_function(X_t)[0]
    # Softmax-normalise scores to get pseudo-probabilities
    e_scores  = np.exp(scores - scores.max())
    proba     = (e_scores / e_scores.sum()).tolist()
    confidence = round(max(proba) * 100, 1)

    proba_dict = {labels[i]: round(p * 100, 1) for i, p in enumerate(proba)}

    # Extract top TF-IDF terms for this specific input text
    feature_names = vec.get_feature_names_out()
    tfidf_matrix  = X_t.toarray()[0]
    top_idx       = tfidf_matrix.argsort()[-15:][::-1]
    top_tokens    = [
        {'term': feature_names[i], 'weight': round(float(tfidf_matrix[i]), 4)}
        for i in top_idx if tfidf_matrix[i] > 0
    ]

    # Get training accuracy
    train_acc = None
    if 'training_report' in LOADED:
        train_acc = LOADED['training_report']['models'].get('nlp_infowar', {}).get('accuracy')

    return jsonify({
        'classification':      pred_label,
        'class_index':         pred_idx,
        'confidence_pct':      confidence,
        'class_probabilities': proba_dict,
        'top_tfidf_tokens':    top_tokens,
        'training_accuracy':   train_acc,
        'timestamp':           datetime.now().isoformat(),
    })


@app.route('/predict/buildup', methods=['POST'])
def predict_buildup():
    """
    Military buildup detection using XGBoost on multi-domain indicators.
    Reuses the risk model with specific feature emphasis.
    """
    if 'xgboost' not in LOADED:
        return jsonify({'error': 'XGBoost model not loaded. Run train_models.py first.'}), 503

    data = request.get_json() or {}

    # Map buildup-specific inputs to feature vector
    # Ordinal signals from UI → numeric feature values
    mob_map  = {'Confirmed Mobilization':4, 'Strong Signals':3, 'Moderate Signals':2, 'No Signal':1}
    sat_map  = {'High Activity Detected':3, 'Moderate Activity':2, 'Normal Activity':1}
    nuke_map = {'DEFCON-equivalent Elevated':4, 'Heightened Rhetoric / Tests':3, 'Minor Posture Signals':2, 'No Change — Routine':1}
    cyb_map  = {'Active Major Campaigns':4, 'Elevated Reconnaissance':3, 'Routine Probing':2, 'No Significant Activity':1}

    mobilization   = mob_map.get(data.get('mobilization','Moderate Signals'), 2)
    cyber_level    = cyb_map.get(data.get('cyber_activity','Routine Probing'), 2)
    nuclear        = nuke_map.get(data.get('nuclear_posture','No Change — Routine'), 1)
    budget_yoy     = float(data.get('budget_yoy_pct', 10.0))
    sipri_tiv      = float(data.get('sipri_tiv', 500.0))
    arms_idx       = round(min(sipri_tiv / 200, 10), 2)

    feature_data = {
        'mil_exp_usd_bn':    float(data.get('mil_exp_usd_bn', 30.0)),
        'mil_gdp_pct':       float(data.get('mil_gdp_pct', 4.0)),
        'arms_import_index': arms_idx,
        'psi':               float(data.get('psi', -1.0)),
        'acled_events_6mo':  int(data.get('acled_events_6mo', 100)),
        'fatalities_12mo':   int(data.get('fatalities_12mo', 1000)),
        'battle_ratio':      0.5,
        'vac_ratio':         0.25,
        'gdp_usd_tn':        float(data.get('gdp_usd_tn', 1.5)),
        'budget_yoy_pct':    budget_yoy,
        'mobilization':      mobilization,
        'cyber_level':       cyber_level,
        'nuclear_posture':   nuclear,
    }

    X, _ = build_feature_vector(feature_data)
    xgb_obj = LOADED['xgboost']
    model   = xgb_obj['model']

    pred_class  = int(model.predict(X)[0])
    proba       = model.predict_proba(X)[0]
    confidence  = round(float(max(proba)) * 100, 1)

    # Buildup-specific scoring: weight mobilization, cyber, nuclear more heavily
    buildup_score = round(float(np.clip(
        pred_class * 22 +
        mobilization * 6 +
        cyber_level * 5 +
        nuclear * 7 +
        min(budget_yoy, 60) / 3 +
        arms_idx * 3,
        0, 100
    )), 1)

    # Domain classification
    domain_scores = {
        'Land':    mobilization * 20,
        'Naval':   int(data.get('naval_signal', 0)) * 20,
        'Air':     int(data.get('air_signal', 0)) * 20,
        'Cyber':   cyber_level * 20,
        'Nuclear': nuclear * 18,
    }
    primary_domain = max(domain_scores, key=domain_scores.get)

    # Warning level
    level_map = {0:'GREEN', 1:'AMBER', 2:'ORANGE', 3:'RED'}
    warning   = level_map[pred_class]

    return jsonify({
        'buildup_score':   buildup_score,
        'risk_class':      CLASS_NAMES[pred_class],
        'warning_level':   warning,
        'primary_domain':  primary_domain,
        'cyber_score':     round(cyber_level * 22.5, 1),
        'nuclear_score':   round(nuclear * 20.0, 1),
        'confidence_pct':  confidence,
        'domain_scores':   domain_scores,
        'feature_inputs':  feature_data,
        'timestamp':       datetime.now().isoformat(),
    })


if __name__ == '__main__':
    port = int(os.environ.get('ML_PORT', 5001))
    print("\n╔══════════════════════════════════════════════╗")
    print("║     SENTINEL v3  —  ML API Online             ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  Flask API  →  http://localhost:{port}            ║")
    print(f"║  Health     →  http://localhost:{port}/health     ║")
    print(f"║  Models     →  http://localhost:{port}/models/info║")
    print("╚══════════════════════════════════════════════╝\n")
    app.run(host='0.0.0.0', port=port, debug=False)
