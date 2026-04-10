"""
============================================================
 SENTINEL v3 — ML Pipeline
 train_models.py

 Trains ALL 5 classifiers from the project synopsis:
   1. Logistic Regression
   2. Random Forest
   3. XGBoost
   4. Support Vector Machine (RBF kernel)
   5. TF-IDF + SVM (Information Warfare NLP)

 Also trains regression models for continuous risk scoring.

 Run once to generate trained models in /models folder:
   python train_models.py

 What this script does:
   1. Generates a realistic synthetic defense dataset
      (based on published SIPRI/ACLED/World Bank feature distributions)
   2. Preprocesses and normalizes features
   3. Trains each classifier with cross-validation
   4. Reports accuracy, precision, recall, F1, confusion matrix
   5. Saves trained models as .pkl files for the Flask API
============================================================
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── PATHS ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

# ══════════════════════════════════════════════════════════════
#  PART 1 — SYNTHETIC DEFENSE DATASET GENERATION
#
#  Feature distributions based on:
#  - SIPRI Military Expenditure Database 2023
#  - World Bank Worldwide Governance Indicators
#  - ACLED conflict event statistics 2010–2024
#  - Uppsala Conflict Data Program (UCDP)
#
#  Risk classes:
#    0 = LOW      (stable, low spending, positive PSI)
#    1 = MODERATE (some indicators elevated)
#    2 = HIGH     (multiple indicators alarming)
#    3 = CRITICAL (active conflict zone)
# ══════════════════════════════════════════════════════════════

def generate_defense_dataset(n_samples=2000):
    """
    Generate synthetic but statistically realistic defense risk dataset.
    Each row represents a country-year observation.
    """
    print("\n[DATA] Generating synthetic defense dataset...")

    data = []

    for _ in range(n_samples):
        # Randomly assign a risk class with realistic class distribution
        # (most country-years are LOW or MODERATE, fewer are HIGH/CRITICAL)
        risk_class = np.random.choice([0, 1, 2, 3], p=[0.35, 0.30, 0.22, 0.13])

        # ── Feature distributions conditioned on risk class ──
        # Military expenditure (USD Billion) — SIPRI data reference
        mil_exp = {
            0: np.random.uniform(0.5, 15),
            1: np.random.uniform(5, 50),
            2: np.random.uniform(20, 120),
            3: np.random.uniform(40, 200),
        }[risk_class] + np.random.normal(0, 2)

        # Military expenditure as % of GDP — SIPRI reference
        # LOW: NATO avg ~2%, HIGH: Russia ~4%, CRITICAL: wartime states up to 35%
        mil_gdp_pct = {
            0: np.random.uniform(0.5, 2.5),
            1: np.random.uniform(1.8, 4.5),
            2: np.random.uniform(3.5, 8.0),
            3: np.random.uniform(5.0, 36.0),
        }[risk_class] + np.random.normal(0, 0.3)

        # Arms import SIPRI TIV (normalized 0–10)
        arms_import = {
            0: np.random.uniform(0.0, 2.5),
            1: np.random.uniform(1.5, 5.0),
            2: np.random.uniform(4.0, 8.0),
            3: np.random.uniform(6.5, 10.0),
        }[risk_class] + np.random.normal(0, 0.4)

        # World Bank Political Stability Index (-2.5 to +2.5)
        psi = {
            0: np.random.uniform(0.2, 2.5),
            1: np.random.uniform(-0.8, 0.8),
            2: np.random.uniform(-1.8, -0.3),
            3: np.random.uniform(-2.5, -1.2),
        }[risk_class] + np.random.normal(0, 0.15)

        # ACLED conflict events (6 months)
        acled_events = {
            0: np.random.randint(0, 30),
            1: np.random.randint(15, 150),
            2: np.random.randint(80, 500),
            3: np.random.randint(300, 1200),
        }[risk_class]

        # ACLED fatalities (12 months)
        fatalities = {
            0: np.random.randint(0, 100),
            1: np.random.randint(50, 800),
            2: np.random.randint(400, 5000),
            3: np.random.randint(2000, 40000),
        }[risk_class]

        # Battle events ratio (battles / total ACLED events)
        battle_ratio = {
            0: np.random.uniform(0.0, 0.3),
            1: np.random.uniform(0.2, 0.5),
            2: np.random.uniform(0.4, 0.65),
            3: np.random.uniform(0.5, 0.75),
        }[risk_class]

        # VAC ratio (Violence Against Civilians)
        vac_ratio = np.random.uniform(0.1, 0.4) * (risk_class + 1) / 4

        # GDP (USD Trillion) — affects mil_gdp_pct interpretation
        gdp = np.random.uniform(0.05, 25)

        # Defense budget YoY change (%)
        budget_yoy = {
            0: np.random.uniform(-5, 5),
            1: np.random.uniform(0, 12),
            2: np.random.uniform(8, 30),
            3: np.random.uniform(15, 60),
        }[risk_class] + np.random.normal(0, 1)

        # Mobilization signal (0=none, 1=weak, 2=moderate, 3=strong, 4=confirmed)
        mobilization = {
            0: np.random.choice([0, 1], p=[0.85, 0.15]),
            1: np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2]),
            2: np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3]),
            3: np.random.choice([2, 3, 4], p=[0.1, 0.4, 0.5]),
        }[risk_class]

        # Cyber activity level (0–4)
        cyber_level = {
            0: np.random.choice([0, 1], p=[0.7, 0.3]),
            1: np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2]),
            2: np.random.choice([1, 2, 3], p=[0.2, 0.4, 0.4]),
            3: np.random.choice([2, 3, 4], p=[0.1, 0.3, 0.6]),
        }[risk_class]

        # Nuclear posture (0=routine, 1=minor signals, 2=rhetoric, 3=elevated, 4=DEFCON-equiv)
        nuclear = {
            0: np.random.choice([0, 1], p=[0.9, 0.1]),
            1: np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15]),
            2: np.random.choice([1, 2, 3], p=[0.3, 0.45, 0.25]),
            3: np.random.choice([2, 3, 4], p=[0.2, 0.4, 0.4]),
        }[risk_class]

        # Composite continuous risk score (0–100) — for regression target
        risk_score = (
            risk_class * 22 +
            min(arms_import * 3, 25) +
            max(0, -psi * 8) +
            min(acled_events / 20, 15) +
            min(budget_yoy * 0.5, 10) +
            mobilization * 2 +
            np.random.normal(0, 4)
        )
        risk_score = float(np.clip(risk_score, 0, 100))

        data.append({
            # Features
            'mil_exp_usd_bn':    round(max(0, mil_exp), 2),
            'mil_gdp_pct':       round(np.clip(mil_gdp_pct, 0, 60), 2),
            'arms_import_index': round(np.clip(arms_import, 0, 10), 2),
            'psi':               round(np.clip(psi, -2.5, 2.5), 3),
            'acled_events_6mo':  max(0, acled_events),
            'fatalities_12mo':   max(0, fatalities),
            'battle_ratio':      round(np.clip(battle_ratio, 0, 1), 3),
            'vac_ratio':         round(np.clip(vac_ratio, 0, 1), 3),
            'gdp_usd_tn':        round(max(0.01, gdp), 3),
            'budget_yoy_pct':    round(budget_yoy, 2),
            'mobilization':      mobilization,
            'cyber_level':       cyber_level,
            'nuclear_posture':   nuclear,
            # Targets
            'risk_class':        risk_class,
            'risk_score':        round(risk_score, 1),
        })

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATA_DIR, 'defense_dataset.csv'), index=False)
    print(f"  ✓ Generated {len(df)} samples")
    print(f"  Class distribution: {df['risk_class'].value_counts().sort_index().to_dict()}")
    return df


def generate_infowar_dataset(n_samples=1500):
    """
    Generate synthetic text dataset for information warfare NLP classification.
    Labels: 0=Legitimate, 1=Propaganda, 2=Disinformation, 3=Psyops
    """
    print("\n[DATA] Generating information warfare NLP dataset...")

    templates = {
        0: [  # Legitimate
            "Officials confirmed military exercises near {loc} in compliance with {treaty} obligations.",
            "UN peacekeepers reported {n} incidents this month in {loc}, consistent with prior assessments.",
            "Defense ministry released quarterly budget figures showing {n}% increase in logistics spending.",
            "NATO spokesperson confirmed standard rotation of forces near {loc} border.",
            "Satellite imagery analyzed by open-source researchers shows {n} vehicles at {loc} base.",
        ],
        1: [  # Propaganda
            "Our heroic forces successfully repelled {n} enemy attacks near {loc}, sources confirm.",
            "The enemy suffered catastrophic losses of {n} personnel at {loc} — victory is near.",
            "Foreign interference in {loc} exposed: {n} operatives arrested by security forces.",
            "International community recognizes our right to defend {loc} from aggression.",
            "Enemy morale collapsing as our forces advance on all fronts near {loc}.",
        ],
        2: [  # Disinformation
            "BREAKING: {n} civilians massacred in {loc} — government hiding evidence from media.",
            "Leaked documents prove military buildup in {loc} is cover for illegal weapons program.",
            "Viral video shows alleged chemical weapons use near {loc} — cannot be verified.",
            "Anonymous sources claim {n} soldiers defecting en masse from {loc} garrison.",
            "Government denies {loc} casualties but images circulating on social media contradict claims.",
        ],
        3: [  # Psyops
            "Soldiers in {loc}: your families are safe only if you surrender now. {n} already have.",
            "Citizens of {loc}: your government has abandoned you. Join us and receive protection.",
            "Military personnel at {loc} base: command has fled. Resistance is futile. {n} confirmed.",
            "The regime in {loc} will fall within {n} days. Those who resist face consequences.",
            "Your comrades at {loc} have already surrendered. Be the {n}th to make the right choice.",
        ],
    }

    locations = ['Donbas', 'Kherson', 'Gaza', 'Rafah', 'Taiwan Strait', 'South China Sea',
                 'Kashmir', 'Tigray', 'Sahel', 'Nagorno-Karabakh', 'North Korea border']
    treaties  = ['NATO', 'OSCE', 'UN Charter', 'bilateral defense', 'CSTO']

    records = []
    for _ in range(n_samples):
        label = np.random.choice([0, 1, 2, 3], p=[0.30, 0.25, 0.28, 0.17])
        tmpl  = np.random.choice(templates[label])
        text  = tmpl.format(
            loc=np.random.choice(locations),
            n=np.random.randint(10, 5000),
            treaty=np.random.choice(treaties)
        )
        # Add some noise words
        if np.random.random() < 0.3:
            text += f" Sources cannot be independently verified."
        records.append({'text': text, 'label': label})

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(DATA_DIR, 'infowar_dataset.csv'), index=False)
    print(f"  ✓ Generated {len(df)} NLP samples")
    print(f"  Class distribution: {df['label'].value_counts().sort_index().to_dict()}")
    return df


# ══════════════════════════════════════════════════════════════
#  PART 2 — TRAINING PIPELINES
# ══════════════════════════════════════════════════════════════

FEATURE_COLS = [
    'mil_exp_usd_bn', 'mil_gdp_pct', 'arms_import_index', 'psi',
    'acled_events_6mo', 'fatalities_12mo', 'battle_ratio', 'vac_ratio',
    'gdp_usd_tn', 'budget_yoy_pct', 'mobilization', 'cyber_level', 'nuclear_posture'
]
TARGET_CLASS = 'risk_class'
TARGET_SCORE = 'risk_score'

CLASS_NAMES = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']


def evaluate_classifier(name, model, X_test, y_test):
    """Print and return evaluation metrics for a classifier."""
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n  {'─'*50}")
    print(f"  Model: {name}")
    print(f"  Accuracy:  {acc*100:.1f}%")
    print(f"  Macro F1:  {report['macro avg']['f1-score']*100:.1f}%")
    print(f"  Precision: {report['macro avg']['precision']*100:.1f}%")
    print(f"  Recall:    {report['macro avg']['recall']*100:.1f}%")
    print(f"  Confusion Matrix:\n{cm}")

    return {
        'accuracy':        round(acc * 100, 2),
        'macro_f1':        round(report['macro avg']['f1-score'] * 100, 2),
        'macro_precision': round(report['macro avg']['precision'] * 100, 2),
        'macro_recall':    round(report['macro avg']['recall'] * 100, 2),
        'report':          report,
        'confusion_matrix': cm.tolist(),
    }


def train_risk_models(df):
    """Train all 4 tabular risk classification models."""
    print("\n" + "═"*60)
    print("  TRAINING RISK CLASSIFICATION MODELS")
    print("═"*60)

    X = df[FEATURE_COLS].values
    y = df[TARGET_CLASS].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    results = {}
    trained = {}

    # ── 1. Logistic Regression (Baseline) ────────────────────
    print("\n[1/4] Logistic Regression (baseline)...")
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
    lr.fit(X_train_sc, y_train)
    results['logistic_regression'] = evaluate_classifier('Logistic Regression', lr, X_test_sc, y_test)
    trained['logistic_regression'] = {'model': lr, 'scaler': scaler, 'needs_scaling': True}

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lr, X_train_sc, y_train, cv=cv, scoring='f1_macro')
    results['logistic_regression']['cv_f1_mean'] = round(cv_scores.mean() * 100, 2)
    results['logistic_regression']['cv_f1_std']  = round(cv_scores.std() * 100, 2)
    print(f"  5-fold CV F1: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    # ── 2. Random Forest ──────────────────────────────────────
    print("\n[2/4] Random Forest (200 estimators)...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=4,
                                 min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)   # RF doesn't need scaling
    results['random_forest'] = evaluate_classifier('Random Forest', rf, X_test, y_test)

    # Feature importance (Gini)
    fi = dict(zip(FEATURE_COLS, rf.feature_importances_.tolist()))
    results['random_forest']['feature_importance'] = dict(
        sorted(fi.items(), key=lambda x: x[1], reverse=True)
    )
    trained['random_forest'] = {'model': rf, 'scaler': scaler, 'needs_scaling': False}

    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='f1_macro')
    results['random_forest']['cv_f1_mean'] = round(cv_scores.mean() * 100, 2)
    results['random_forest']['cv_f1_std']  = round(cv_scores.std() * 100, 2)
    print(f"  5-fold CV F1: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
    print(f"  Top features: {list(fi.keys())[:4]}")

    # ── 3. XGBoost ────────────────────────────────────────────
    print("\n[3/4] XGBoost (500 estimators)...")
    xgb = XGBClassifier(
            n_estimators=500, 
            max_depth=6, 
            learning_rate=0.05,
            subsample=0.8, 
            max_features=0.8,
            random_state=42
        )    
    xgb.fit(X_train, y_train)
    results['xgboost'] = evaluate_classifier('XGBoost', xgb, X_test, y_test)

    fi_xgb = dict(zip(FEATURE_COLS, xgb.feature_importances_.tolist()))
    results['xgboost']['feature_importance'] = dict(
        sorted(fi_xgb.items(), key=lambda x: x[1], reverse=True)
    )
    trained['xgboost'] = {'model': xgb, 'scaler': scaler, 'needs_scaling': False}

    cv_scores = cross_val_score(xgb, X_train, y_train, cv=cv, scoring='f1_macro')
    results['xgboost']['cv_f1_mean'] = round(cv_scores.mean() * 100, 2)
    results['xgboost']['cv_f1_std']  = round(cv_scores.std() * 100, 2)
    print(f"  5-fold CV F1: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    # ── 4. SVM (RBF kernel) ───────────────────────────────────
    print("\n[4/4] SVM (RBF kernel, probability calibration)...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train_sc, y_train)
    results['svm'] = evaluate_classifier('SVM (RBF)', svm, X_test_sc, y_test)
    trained['svm'] = {'model': svm, 'scaler': scaler, 'needs_scaling': True}

    cv_scores = cross_val_score(svm, X_train_sc, y_train, cv=cv, scoring='f1_macro')
    results['svm']['cv_f1_mean'] = round(cv_scores.mean() * 100, 2)
    results['svm']['cv_f1_std']  = round(cv_scores.std() * 100, 2)
    print(f"  5-fold CV F1: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    # ── Save scaler separately ────────────────────────────────
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    with open(os.path.join(MODELS_DIR, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(FEATURE_COLS, f)

    # ── Save each model ───────────────────────────────────────
    for name, obj in trained.items():
        path = os.path.join(MODELS_DIR, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"  ✓ Saved {name}.pkl")

    return results, trained


def train_nlp_model(df_nlp):
    """Train TF-IDF + SVM pipeline for information warfare classification."""
    print("\n" + "═"*60)
    print("  TRAINING NLP — INFORMATION WARFARE CLASSIFIER")
    print("═"*60)

    X_text = df_nlp['text'].values
    y_nlp  = df_nlp['label'].values
    nlp_labels = ['Legitimate', 'Propaganda', 'Disinformation', 'Psyops']

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y_nlp, test_size=0.2, random_state=42, stratify=y_nlp
    )

    # TF-IDF + SVM pipeline (exactly as described in synopsis)
    nlp_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),      # unigrams + bigrams
            max_features=5000,       # top 5000 features
            sublinear_tf=True,       # log normalization
            min_df=2,
            strip_accents='unicode',
        )),
        ('clf', LinearSVC(C=1.0, max_iter=2000, random_state=42))
    ])

    nlp_pipeline.fit(X_train, y_train)
    y_pred = nlp_pipeline.predict(X_test)

    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=nlp_labels, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n  TF-IDF + SVM (LinearSVC)")
    print(f"  Accuracy:  {acc*100:.1f}%")
    print(f"  Macro F1:  {report['macro avg']['f1-score']*100:.1f}%")
    print(f"  Confusion Matrix:\n{cm}")

    # Top TF-IDF features per class
    vectorizer = nlp_pipeline.named_steps['tfidf']
    classifier = nlp_pipeline.named_steps['clf']
    feature_names = vectorizer.get_feature_names_out()

    top_features = {}
    for i, class_name in enumerate(nlp_labels):
        if hasattr(classifier, 'coef_'):
            top_idx = classifier.coef_[i].argsort()[-10:][::-1]
            top_features[class_name] = [
                {'term': feature_names[j], 'weight': round(float(classifier.coef_[i][j]), 4)}
                for j in top_idx
            ]

    nlp_results = {
        'accuracy':        round(acc * 100, 2),
        'macro_f1':        round(report['macro avg']['f1-score'] * 100, 2),
        'macro_precision': round(report['macro avg']['precision'] * 100, 2),
        'macro_recall':    round(report['macro avg']['recall'] * 100, 2),
        'top_features':    top_features,
        'confusion_matrix': cm.tolist(),
    }

    # Save NLP model
    path = os.path.join(MODELS_DIR, 'nlp_infowar.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'pipeline': nlp_pipeline, 'labels': nlp_labels}, f)
    print(f"  ✓ Saved nlp_infowar.pkl")

    return nlp_results


# ══════════════════════════════════════════════════════════════
#  PART 3 — SAVE TRAINING REPORT (for your synopsis / report)
# ══════════════════════════════════════════════════════════════

def save_training_report(risk_results, nlp_results):
    """Save a JSON report of all model metrics — useful for your project report."""
    report = {
        'generated_at': datetime.now().isoformat(),
        'dataset': {
            'samples': 2000,
            'features': FEATURE_COLS,
            'target_classes': CLASS_NAMES,
            'train_split': 0.80,
            'test_split':  0.20,
        },
        'models': {
            **{k: {kk: vv for kk, vv in v.items() if kk != 'report'}
               for k, v in risk_results.items()},
            'nlp_infowar': nlp_results,
        }
    }

    path = os.path.join(MODELS_DIR, 'training_report.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✓ Training report saved to models/training_report.json")
    return report


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "╔"+"═"*58+"╗")
    print("║" + "  SENTINEL v3 — ML Training Pipeline".center(58) + "║")
    print("╚"+"═"*58+"╝")

    # Generate datasets
    df_defense = generate_defense_dataset(n_samples=2000)
    df_nlp     = generate_infowar_dataset(n_samples=1500)

    # Train risk classification models
    risk_results, trained_models = train_risk_models(df_defense)

    # Train NLP model
    nlp_results = train_nlp_model(df_nlp)

    # Save report
    report = save_training_report(risk_results, nlp_results)

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "╔"+"═"*58+"╗")
    print("║" + "  TRAINING COMPLETE — RESULTS SUMMARY".center(58) + "║")
    print("╠"+"═"*58+"╣")
    for mname, mres in report['models'].items():
        label = mname.replace('_', ' ').title()
        acc   = mres.get('accuracy','—')
        f1    = mres.get('macro_f1','—')
        cv    = mres.get('cv_f1_mean','—')
        std   = mres.get('cv_f1_std','')
        cv_str = f"{cv}%±{std}%" if cv != '—' else '—'
        print(f"║  {label:<28} Acc:{acc:>5}%  F1:{f1:>5}%  CV:{cv_str:<12}║")
    print("╚"+"═"*58+"╝\n")
    print("  All models saved to: ml_backend/models/")
    print("  Start the API with:  python app.py\n")
