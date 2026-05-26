import os
import random
import string
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM
from datetime import datetime

app = Flask(__name__)

# ─────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────
scaler = None
iso_forest = None
nn_model = None
oc_svm = None
dataset_stats = {}
recent_predictions = []   # newest first, max 20


# ─────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────
def generate_dataset(n_total=8000, fraud_rate=0.02, seed=42):
    rng = np.random.RandomState(seed)
    n_fraud = int(n_total * fraud_rate)      # 160
    n_legit = n_total - n_fraud              # 7840

    # ── legitimate transactions ──────────────
    legit_amount   = rng.lognormal(mean=4.5, sigma=1.2, size=n_legit)
    legit_hour     = rng.randint(0, 24, size=n_legit)
    legit_dow      = rng.randint(0, 7,  size=n_legit)
    legit_v        = rng.randn(n_legit, 10)
    legit_label    = np.zeros(n_legit, dtype=int)

    # ── fraudulent transactions ──────────────
    fraud_amount   = rng.lognormal(mean=6.5, sigma=1.5, size=n_fraud)   # higher
    # peak hours 2-5 am
    fraud_hour     = rng.choice([2, 3, 4, 5], size=n_fraud)
    fraud_dow      = rng.randint(0, 7, size=n_fraud)
    fraud_v        = rng.randn(n_fraud, 10) + rng.choice([-2, 2], size=(n_fraud, 10))
    fraud_label    = np.ones(n_fraud, dtype=int)

    # ── combine ──────────────────────────────
    amounts   = np.concatenate([legit_amount, fraud_amount])
    hours     = np.concatenate([legit_hour,   fraud_hour])
    dows      = np.concatenate([legit_dow,    fraud_dow])
    v_matrix  = np.vstack([legit_v, fraud_v])
    labels    = np.concatenate([legit_label,  fraud_label])

    v_cols = {f"v{i+1}": v_matrix[:, i] for i in range(10)}
    df = pd.DataFrame({
        "amount":      amounts,
        "hour":        hours,
        "day_of_week": dows,
        **v_cols,
        "is_fraud":    labels,
    })
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train_models():
    global scaler, iso_forest, nn_model, oc_svm, dataset_stats

    print("[startup] Generating synthetic dataset …")
    df = generate_dataset()

    feature_cols = ["amount", "hour", "day_of_week"] + [f"v{i}" for i in range(1, 11)]
    X = df[feature_cols].values
    y = df["is_fraud"].values

    # ── preprocess: scale amount + hour only ──
    scale_idx = [0, 1]          # amount, hour
    scaler = StandardScaler()
    X_scaled = X.copy().astype(float)
    X_scaled[:, scale_idx] = scaler.fit_transform(X[:, scale_idx])

    # ── dataset statistics ────────────────────
    fraud_mask = y == 1
    peak_hour_counts = df[fraud_mask]["hour"].value_counts()
    dataset_stats = {
        "total_transactions": int(len(df)),
        "fraud_count":        int(fraud_mask.sum()),
        "fraud_rate":         round(float(fraud_mask.mean()) * 100, 2),
        "avg_amount":         round(float(df["amount"].mean()), 2),
        "avg_fraud_amount":   round(float(df.loc[fraud_mask, "amount"].mean()), 2),
        "peak_fraud_hour":    int(peak_hour_counts.idxmax()),
    }
    print(f"[startup] Dataset stats: {dataset_stats}")

    # ── Isolation Forest ─────────────────────
    print("[startup] Training Isolation Forest …")
    iso_forest = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
    iso_forest.fit(X_scaled)

    # ── Neural Network (MLP) ─────────────────
    print("[startup] Training Neural Network …")
    nn_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    nn_model.fit(X_scaled, y)

    # ── One-Class SVM (subsample for speed) ──
    print("[startup] Training One-Class SVM …")
    legit_idx = np.where(y == 0)[0]
    sample_idx = np.random.RandomState(42).choice(legit_idx, size=min(1000, len(legit_idx)), replace=False)
    oc_svm = OneClassSVM(kernel="rbf", nu=0.02)
    oc_svm.fit(X_scaled[sample_idx])

    print("[startup] All models ready.")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def preprocess_input(data: dict) -> np.ndarray:
    """Convert incoming JSON dict to scaled feature vector."""
    raw = np.array([[
        float(data.get("amount", 0)),
        float(data.get("hour",   0)),
        float(data.get("day_of_week", 0)),
        *[float(data.get(f"v{i}", 0)) for i in range(1, 11)],
    ]])
    scaled = raw.copy()
    scaled[:, [0, 1]] = scaler.transform(raw[:, [0, 1]])
    return scaled


def risk_level(fraud_count: int) -> str:
    if fraud_count == 0:
        return "Low"
    elif fraud_count == 1:
        return "Medium"
    elif fraud_count == 2:
        return "High"
    else:
        return "Critical"


def transaction_id() -> str:
    return "TXN-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def api_stats():
    return jsonify(dataset_stats)


@app.route("/api/detect", methods=["POST"])
def api_detect():
    data = request.get_json(force=True)
    X = preprocess_input(data)

    # ── Isolation Forest ──────────────────────
    if_score  = float(iso_forest.decision_function(X)[0])
    if_pred   = iso_forest.predict(X)[0]          # 1 = normal, -1 = anomaly
    if_fraud  = bool(if_pred == -1)
    # normalise score to [0,1] confidence of being fraud
    if_conf   = round(max(0.0, min(1.0, (-if_score + 0.2) / 0.4)), 3)

    # ── Neural Network ────────────────────────
    nn_proba  = nn_model.predict_proba(X)[0]      # [p_legit, p_fraud]
    nn_prob   = round(float(nn_proba[1]), 3)
    nn_fraud  = bool(nn_model.predict(X)[0] == 1)
    nn_conf   = nn_prob if nn_fraud else round(1.0 - nn_prob, 3)

    # ── One-Class SVM ─────────────────────────
    svm_score = float(oc_svm.decision_function(X)[0])
    svm_pred  = oc_svm.predict(X)[0]             # 1 = normal, -1 = anomaly
    svm_fraud = bool(svm_pred == -1)

    # ── Consensus ────────────────────────────
    fraud_votes = sum([if_fraud, nn_fraud, svm_fraud])
    consensus_fraud = fraud_votes >= 2

    tid = transaction_id()
    result = {
        "transaction_id": tid,
        "isolation_forest": {
            "is_fraud":   if_fraud,
            "confidence": if_conf,
            "score":      round(if_score, 4),
        },
        "neural_network": {
            "is_fraud":    nn_fraud,
            "confidence":  nn_conf,
            "probability": nn_prob,
        },
        "one_class_svm": {
            "is_fraud": svm_fraud,
            "score":    round(svm_score, 4),
        },
        "consensus": {
            "is_fraud":    consensus_fraud,
            "fraud_models": fraud_votes,
            "risk_level":  risk_level(fraud_votes),
        },
    }

    # ── store for /api/recent ─────────────────
    entry = {
        **result,
        "input": {
            "amount":      float(data.get("amount", 0)),
            "hour":        int(data.get("hour", 0)),
            "day_of_week": int(data.get("day_of_week", 0)),
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    recent_predictions.insert(0, entry)
    if len(recent_predictions) > 20:
        recent_predictions.pop()

    return jsonify(result)


@app.route("/api/recent")
def api_recent():
    return jsonify(recent_predictions[:20])


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_models()
    app.run(debug=False, port=5000)
