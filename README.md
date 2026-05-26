# Fraud Detection Using Anomaly Detection

A complete machine learning project for identifying suspicious financial transactions using both supervised and unsupervised anomaly detection techniques, based on the Credit Card Fraud Detection dataset structure from Kaggle (mlg-ulb/creditcardfraud).

---

## Project Overview

Credit card fraud is a significant problem costing billions of dollars annually. This project builds and compares multiple approaches to detect fraudulent transactions:

- **Supervised Learning**: Neural Network trained on labeled fraud/non-fraud examples
- **Unsupervised Learning**: Isolation Forest, Local Outlier Factor, One-Class SVM, and Autoencoder — all without using labels during training

The core challenge is extreme class imbalance: fraud typically represents less than 0.2% of real transactions (simulated here at ~2%).

---

## Dataset

The project generates a synthetic dataset modeled on the Kaggle Credit Card Fraud Detection dataset with:

| Column | Description |
|--------|-------------|
| `Time` | Seconds elapsed since first transaction |
| `V1`–`V28` | PCA-transformed features (anonymized) |
| `Amount` | Transaction amount in USD |
| `Class` | Target label: 0 = legitimate, 1 = fraud |

**Total transactions**: 10,000  
**Fraud rate**: ~2% (~200 fraudulent transactions)

To use the real Kaggle dataset, download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` directory, then update the data-loading cell in the notebook.

---

## Project Structure

```
fraud-detection/
├── data/                        # Place creditcard.csv here (or use synthetic data)
├── fraud_detection.ipynb        # Main notebook with all experiments
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter

```bash
jupyter notebook fraud_detection.ipynb
```

### 4. Run all cells

Execute cells top to bottom. Training the Neural Network and One-Class SVM may take a minute each.

---

## Methodology

### Preprocessing

- `StandardScaler` applied to `Amount` and `Time` (V1–V28 are already PCA-transformed)
- 80/20 stratified train/test split to preserve class ratios

### Supervised Approach — Neural Network

A fully connected feedforward network trained with binary cross-entropy:

```
Input (30 features)
  → Dense(128, ReLU) → Dropout(0.3)
  → Dense(64, ReLU)  → Dropout(0.3)
  → Dense(1, Sigmoid)
```

- **Optimizer**: Adam, lr=0.001
- **Class weights**: computed to handle imbalance
- **Threshold**: 0.5 on sigmoid output

**Best for**: Production systems with labeled historical data. Highest precision.

### Unsupervised Approaches

| Model | Core Idea | Key Hyperparameter |
|-------|-----------|--------------------|
| **Isolation Forest** | Anomalies are isolated faster in random trees | `contamination=0.02` |
| **Local Outlier Factor** | Points with lower density than neighbors | `n_neighbors=20`, `contamination=0.02` |
| **One-Class SVM** | Boundary around normal data in kernel space | `kernel='rbf'`, `nu=0.01` |
| **Autoencoder** | High reconstruction error = anomaly | Threshold = mean + 3×std of training errors |

**Best for**: Detecting novel fraud patterns not seen during training; early deployment when labels are scarce.

---

## When to Use Each Approach

### Use Supervised (Neural Network) when:
- You have a large, clean labeled dataset of historical fraud
- False positives are costly (blocking legitimate customers)
- Fraud patterns are relatively stable over time
- You need maximum precision

### Use Unsupervised (Isolation Forest / Autoencoder) when:
- Labels are unavailable or expensive to obtain
- You need to catch novel/evolving fraud types not in training data
- You want a complementary layer on top of a supervised system
- You're in early deployment with limited fraud examples

### Hybrid Recommendation:
Train a supervised model on labeled data, then use an unsupervised model as a second-pass filter to flag transactions the supervised model would have missed.

---

## Key Results (approximate, synthetic data)

| Method | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Neural Network | ~0.90 | ~0.85 | ~0.87 |
| Isolation Forest | ~0.15 | ~0.75 | ~0.25 |
| Local Outlier Factor | ~0.10 | ~0.70 | ~0.17 |
| One-Class SVM | ~0.12 | ~0.65 | ~0.20 |
| Autoencoder | ~0.20 | ~0.80 | ~0.32 |

> Results will vary with each run (random synthetic data). Supervised learning achieves higher precision; unsupervised methods achieve higher recall.

---

## Key Concepts Demonstrated

- **Imbalanced classification**: class weights, stratified splits, precision/recall over accuracy
- **Anomaly detection**: isolation-based, density-based, kernel-based, and reconstruction-based methods
- **Neural network regularization**: Dropout to prevent overfitting on majority class
- **Autoencoder thresholding**: setting anomaly cutoff from reconstruction error distribution
- **Model comparison**: unified evaluation framework across supervised and unsupervised methods

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | 2.1.4 | Data manipulation |
| numpy | 1.26.2 | Numerical computing |
| scikit-learn | 1.3.2 | Isolation Forest, LOF, One-Class SVM, metrics |
| tensorflow | 2.15.0 | Neural Network + Autoencoder (Keras) |
| matplotlib | 3.8.2 | Plotting |
| seaborn | 0.13.0 | Statistical visualization |
| jupyter | 1.0.0 | Notebook environment |
| notebook | 7.0.6 | Jupyter Notebook server |
