<div align="center">

```
███████╗██████╗  █████╗ ██╗   ██╗██████╗     ██████╗ ███████╗████████╗███████╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██╔══██╗██╔══██╗██║   ██║██╔══██╗    ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║
█████╗  ██████╔╝███████║██║   ██║██║  ██║    ██║  ██║█████╗     ██║   █████╗  ██║        ██║   ██║██║   ██║██╔██╗ ██║
██╔══╝  ██╔══██╗██╔══██║██║   ██║██║  ██║    ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   ██║██║   ██║██║╚██╗██║
██║     ██║  ██║██║  ██║╚██████╔╝██████╔╝    ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝    ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
```

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=FF6B35&center=true&vCenter=true&width=700&lines=Finding+0.2%25+Needles+in+a+99.8%25+Haystack+%F0%9F%94%8D;Neural+Network+%7C+Isolation+Forest+%7C+Autoencoder;Supervised+%26+Unsupervised+Anomaly+Detection;Stop+Fraud+Before+It+Lands+%F0%9F%9B%A1%EF%B8%8F" alt="Typing SVG" />

<img src="https://media.giphy.com/media/l3vRmVv5P01I5NDAA/giphy.gif" width="360" />

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> **A multi-model fraud detection system combining supervised deep learning and four unsupervised anomaly detectors — built for the hardest problem in fintech: finding 0.2% needles in a 99.8% haystack.**

</div>

---

## ◈ The Challenge

Credit card fraud costs billions annually. The core difficulty is **extreme class imbalance** — fraud is rare, yet catastrophically expensive when missed. This project tackles the problem from two angles: supervised learning when labels exist, and unsupervised anomaly detection when they don't.

```
┌────────────────────────────────────────────────────────────────────┐
│                  DETECTION ARCHITECTURE                            │
│                                                                    │
│  10,000 Transactions  →  Preprocessing  →  Train/Test Split        │
│  (~2% fraud rate)          StandardScaler     80/20 stratified    │
│                                  │                                 │
│              ┌───────────────────┴───────────────────┐            │
│              │ SUPERVISED                UNSUPERVISED │            │
│              │ Neural Network ──→ 93% ROC-AUC         │            │
│              │ Isolation Forest ──→ anomaly score     │            │
│              │ Local Outlier Factor ──→ density score │            │
│              │ One-Class SVM ──→ kernel boundary      │            │
│              │ Autoencoder ──→ reconstruction error   │            │
│              └────────────────────────────────────────┘            │
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

## ◈ Neural Network Architecture

```
Input (30 features)
  → Dense(128, ReLU) → Dropout(0.3)
  → Dense(64, ReLU)  → Dropout(0.3)
  → Dense(1, Sigmoid)
```

Trained with **class weights** to handle imbalance. Best precision in production.

---

## ◈ When to Use Each Approach

**Supervised** — large labeled dataset, stable fraud patterns, low false-positive tolerance

**Unsupervised** — no labels available, novel/evolving fraud types, second-pass filter on top of supervised

---

## ◈ Dataset Schema

| Column | Description |
|---|---|
| `Time` | Seconds since first transaction |
| `V1–V28` | PCA-transformed anonymized features |
| `Amount` | Transaction value (USD) |
| `Class` | Target: 0=legitimate, 1=fraud |

---

## ◈ Quick Start

```bash
git clone https://github.com/isamkhan1809/fraud-detection.git
cd fraud-detection
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb
```

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

<div align="center">

**Five Models. One Goal. Zero Fraud.**

*MIT License*

<br/>

Working in fintech, cybersecurity, or anomaly detection?<br/>
Let's connect — built by <a href="https://github.com/isamkhan1809">Isam Khan</a> &nbsp;|&nbsp;
<a href="https://linkedin.com/in/isam-khan-3a1260292"><img src="https://img.shields.io/badge/-LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white&labelColor=000000"/></a>
<a href="https://isamkhan.com"><img src="https://img.shields.io/badge/-isamkhan.com-00D9FF?style=flat-square&logo=googlechrome&logoColor=white&labelColor=000000"/></a>

</div>
