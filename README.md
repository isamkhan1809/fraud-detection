<div align="center">

```
███████╗██████╗  █████╗ ██╗   ██╗██████╗     ██████╗ ███████╗████████╗███████╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██╔══██╗██╔══██╗██║   ██║██╔══██╗    ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║
█████╗  ██████╔╝███████║██║   ██║██║  ██║    ██║  ██║█████╗     ██║   █████╗  ██║        ██║   ██║██║   ██║██╔██╗ ██║
██╔══╝  ██╔══██╗██╔══██║██║   ██║██║  ██║    ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   ██║██║   ██║██║╚██╗██║
██║     ██║  ██║██║  ██║╚██████╔╝██████╔╝    ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝    ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
```

### *Find the Signal in the Noise. Stop Fraud Before It Lands.*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

> **A multi-model fraud detection system combining supervised deep learning and four unsupervised anomaly detectors — built for the hardest problem in fintech: finding 0.2% needles in a 99.8% haystack.**

</div>

---

## ◈ The Challenge

Credit card fraud costs billions annually. The core difficulty is **extreme class imbalance** — fraud is rare, yet catastrophically expensive when missed. This project tackles the problem from two angles simultaneously: supervised learning when labels exist, and unsupervised anomaly detection when they don't.

```
┌────────────────────────────────────────────────────────────────────┐
│                  DETECTION ARCHITECTURE                            │
│                                                                    │
│  10,000 Transactions  →  Preprocessing  →  Train/Test Split        │
│  (~2% fraud rate)          StandardScaler     80/20 stratified    │
│                                  │                                 │
│              ┌───────────────────┴───────────────────┐            │
│              │ SUPERVISED                UNSUPERVISED │            │
│              │                                        │            │
│              │ Neural Network ──→ 93% ROC-AUC         │            │
│              │                                        │            │
│              │ Isolation Forest ──→ anomaly score     │            │
│              │ Local Outlier Factor ──→ density score │            │
│              │ One-Class SVM ──→ kernel boundary      │            │
│              │ Autoencoder ──→ reconstruction error   │            │
│              └────────────────────────────────────────┘            │
│                                  │                                 │
│                    Unified Evaluation + Comparison                 │
└────────────────────────────────────────────────────────────────────┘
```

---

## ◈ Model Comparison

| Method | Precision | Recall | F1 | Best For |
|---|---|---|---|---|
| **Neural Network** | ~0.90 | ~0.85 | ~0.87 | Production, labeled data |
| Autoencoder | ~0.20 | ~0.80 | ~0.32 | Novel fraud patterns |
| Isolation Forest | ~0.15 | ~0.75 | ~0.25 | No labels needed |
| Local Outlier Factor | ~0.10 | ~0.70 | ~0.17 | Density-based anomalies |
| One-Class SVM | ~0.12 | ~0.65 | ~0.20 | Kernel-space boundary |

---

## ◈ The Models Explained

### Neural Network (Supervised)

```
Input (30 features)
  → Dense(128, ReLU) → Dropout(0.3)
  → Dense(64, ReLU)  → Dropout(0.3)
  → Dense(1, Sigmoid)
```

Trained with **class weights** to handle imbalance. Best precision in production.

### Isolation Forest

Anomalies are isolated **faster** in random decision trees. Fraudulent transactions require fewer splits — they're outliers by nature.

### Autoencoder

Trains to reconstruct **normal** transactions. At inference time, fraud produces **high reconstruction error** — threshold at mean + 3σ.

### Use Supervised When

- Large labeled historical dataset exists
- False positives are costly (blocking real customers)
- Fraud patterns are stable

### Use Unsupervised When

- Labels are scarce or expensive
- You need to catch **novel** fraud types not in training data
- Deploying a second-pass filter on top of supervised models

---

## ◈ Dataset Schema

| Column | Description |
|---|---|
| `Time` | Seconds since first transaction |
| `V1–V28` | PCA-transformed anonymized features |
| `Amount` | Transaction value (USD) |
| `Class` | Target: 0=legitimate, 1=fraud |

To use the real Kaggle dataset: download `creditcard.csv` from [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) → place in `data/`.

---

## ◈ Quick Start

```bash
# 1. Clone
git clone https://github.com/isamkhan1809/fraud-detection.git
cd fraud-detection

# 2. Virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install
pip install -r requirements.txt

# 4. Launch notebook
jupyter notebook fraud_detection.ipynb
```

Run all cells — synthetic data is generated automatically if no CSV is present.

---

## ◈ Project Structure

```
fraud-detection/
├── fraud_detection.ipynb   ← All models, training, evaluation
├── data/                   ← Place creditcard.csv here
├── requirements.txt
└── README.md
```

---

## ◈ Key Concepts Demonstrated

- Imbalanced classification with class weights and stratified splits
- Isolation-based, density-based, kernel-based, and reconstruction-based anomaly detection
- Neural network regularisation with Dropout
- Autoencoder threshold calibration from reconstruction error distribution
- Unified precision/recall/F1 comparison across all five models

---

<div align="center">

**Five Models. One Goal. Zero Fraud.**

*MIT License*

</div>
