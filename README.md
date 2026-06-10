<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,17,20&height=200&section=header&text=Fraud%20Detection&fontSize=72&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Finding%200.2%25%20Needles%20in%20a%2099.8%25%20Haystack&descAlignY=60&descSize=20" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-FF6B35?style=for-the-badge&logo=python&logoColor=white&labelColor=0D0D0D)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white&labelColor=0D0D0D)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-5%20Models-FF6B35?style=for-the-badge&logo=scikit-learn&logoColor=white&labelColor=0D0D0D)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-FF6B35?style=for-the-badge&labelColor=0D0D0D)](LICENSE)

<br/>

<a href="https://github.com/isamkhan1809/fraud-detection">
  <img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=22&pause=1000&color=FF6B35&center=true&vCenter=true&width=700&lines=Neural+Network+%7C+Isolation+Forest+%7C+Autoencoder;Supervised+%26+Unsupervised+Anomaly+Detection;93%25+ROC-AUC+%E2%80%94+Built+for+Class+Imbalance;Stop+Fraud+Before+It+Lands." alt="Typing SVG" />
</a>

</div>

---

<br/>

<div align="center">

```
  ╔══════════════════════════════════════════════════════════════╗
  ║                                                              ║
  ║   In a sea of 10,000 transactions, 200 are fraudulent.      ║
  ║   They look almost identical to the legitimate ones.        ║
  ║                                                              ║
  ║       Five models. One mission. Find them all.              ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
```

</div>

<br/>

## `>_ The Story`

> *Credit card fraud costs billions annually. The hardest part isn't the algorithm — it's the imbalance. Fraud is rare. So rare that a model predicting "legitimate" every single time would still be 98% accurate.*
>
> *This project tackles that challenge head-on, combining a supervised neural network with four unsupervised anomaly detectors — so fraud is caught whether you have labels or not.*
>
> *One architecture for production. Four for the unknown.*

<br/>

## `>_ Five Models`

<table>
<tr>
<td width="50%">

**Supervised — when labels exist:**
```
Neural Network
  Dense(128) → Dropout(0.3)
  Dense(64)  → Dropout(0.3)
  Dense(1, Sigmoid)
  Class weights for imbalance
```

</td>
<td width="50%">

**Unsupervised — when they don't:**
```
Isolation Forest   → anomaly score
Local Outlier Factor → density gap
One-Class SVM      → kernel boundary
Autoencoder        → reconstruction error
                     threshold: μ + 3σ
```

</td>
</tr>
</table>

<br/>

## `>_ The Pipeline`

```
┌─────────────────────────────────────────────────────────────┐
│                  FRAUD DETECTION PIPELINE                   │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ 10,000 trans │───▶│ StandardScaler│───▶│  80/20 Split │  │
│  │  ~2% fraud   │    │ Amount + Time │    │  Stratified  │  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘  │
│                                                 │          │
│                 ┌───────────────────────────────┤          │
│                 ▼                               ▼          │
│         SUPERVISED                       UNSUPERVISED      │
│         Neural Network                   Isolation Forest  │
│         (class weights)                  LOF               │
│                                          One-Class SVM     │
│                                          Autoencoder       │
│                 └───────────────────────────────┤          │
│                                                 ▼          │
│                             Unified Evaluation + Compare   │
│                             Precision · Recall · F1 · AUC  │
└─────────────────────────────────────────────────────────────┘
```

<br/>

## `>_ Results`

<div align="center">

| Method | Precision | Recall | F1 | Use When |
|---|---|---|---|---|
| **Neural Network** | ~0.90 | ~0.85 | ~0.87 | Labels available, stable patterns |
| Autoencoder | ~0.20 | ~0.80 | ~0.32 | Novel fraud types |
| Isolation Forest | ~0.15 | ~0.75 | ~0.25 | No labels at all |
| Local Outlier Factor | ~0.10 | ~0.70 | ~0.17 | Density-based anomalies |
| One-Class SVM | ~0.12 | ~0.65 | ~0.20 | Kernel-space boundary |

</div>

<br/>

## `>_ Get Running`

```bash
# Clone
git clone https://github.com/isamkhan1809/fraud-detection.git
cd fraud-detection

# Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Launch notebook (synthetic data auto-generated)
jupyter notebook fraud_detection.ipynb
```

To use the real Kaggle dataset: download `creditcard.csv` from [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) → place in `data/`.

<br/>

## `>_ Tech Stack`

<div align="center">

| Layer | Technology |
|---|---|
| **Deep Learning** | TensorFlow / Keras |
| **Anomaly Detection** | scikit-learn |
| **Data** | pandas, numpy |
| **Visualisation** | matplotlib, seaborn |
| **Notebook** | Jupyter |

</div>

<br/>

## `>_ Project Structure`

```
fraud-detection/
├── fraud_detection.ipynb   ← All 5 models, training, evaluation
├── data/                   ← Place creditcard.csv here
└── requirements.txt
```

<br/>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,17,20&height=120&section=footer&animation=twinkling" width="100%"/>

<br/>

*Five models. One goal. Zero fraud.*
*Supervised and unsupervised — because fraud evolves.*

<br/>

[![GitHub](https://img.shields.io/badge/github-isamkhan1809-FF6B35?style=for-the-badge&logo=github&logoColor=white&labelColor=0D0D0D)](https://github.com/isamkhan1809)

</div>
