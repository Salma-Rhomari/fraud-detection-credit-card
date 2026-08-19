# 💳 Credit Card Fraud Detection (Decision Tree)

A supervised machine learning pipeline using a **Decision Tree Classifier** to detect fraudulent credit card transactions in a highly imbalanced dataset.

Academic project — Faculty of Sciences and Technology, Tangier (Module: Data Mining, Major SSD, 2025/2026).

## Overview

The dataset contains **284,807** real European credit card transactions from September 2013, of which only **492 (≈0.17%)** are fraudulent. This extreme class imbalance is the central challenge of the project: a naive model predicting "normal" for every transaction would already score ~99.8% accuracy while catching zero fraud. The project focuses on handling this imbalance correctly and evaluating with metrics that actually reflect fraud-detection performance (precision, recall, F1-score, confusion matrix) rather than raw accuracy.

28 of the 31 features (`V1`–`V28`) are PCA-anonymized for confidentiality. Only `Time`, `Amount`, and the target `Class` are in their original form.

## Methodology

1. **Data Loading & Quality Check** — verify no missing values, inspect the class distribution.
2. **Feature/Target Split** — `X` = all columns except `Class`, `y` = `Class`.
3. **Stratified Train/Test Split** — 80/20 split with `stratify=y`, so the ~0.17% fraud ratio is preserved in both sets (a purely random split risks placing too few fraud cases in the test set).
4. **Model** — `DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=42)`.
   - `criterion='entropy'`: the tree splits nodes to maximize Information Gain.
   - `class_weight='balanced'`: penalizes misclassifying fraud (the minority class) far more heavily than normal transactions, forcing the tree to prioritize catching fraud instead of ignoring it.
   - Note: unlike KNN, Decision Trees are not sensitive to feature scale, so no `StandardScaler` is needed here.
5. **Evaluation** — confusion matrix, classification report (precision/recall/F1 per class), and a log-scale bar chart of the four confusion-matrix quadrants (TN/FP/FN/TP).
6. **Visualization** — the trained tree is plotted (depth capped at 3 for readability) to show exactly which variable thresholds (e.g. `V14`, `V12`, `V17`) drive a fraud classification — a key advantage of Decision Trees over black-box models.

## Results

On the held-out test set:

| Metric | Class 1 (Fraud) |
|---|---|
| Precision | ~0.79 |
| Recall | ~0.68–0.90 (varies by run/config) |
| F1-score | ~0.73 |

The model correctly intercepts the large majority of real fraud cases while generating relatively few false alarms on legitimate transactions. The root-node split (`V14 <= -2.736` in the full run) alone separates a highly fraud-dense branch from an overwhelmingly normal one — a strong first signal, refined further down the tree by `V12` and `V4`.

**Key takeaway:** `class_weight='balanced'` is what makes the difference between a model that "cheats" by predicting the majority class and one that actually learns to flag fraud. The trade-off is a higher false-positive rate, which is an acceptable cost in a banking risk-management context (a missed fraud is far more costly than a legitimate transaction flagged for review).

## 📁 Project Structure

```text
fraud-detection-credit-card/
│
├── README.md
├── requirements.txt
├── .gitignore
├── main.py                    # Runs the full pipeline end-to-end
│
├── data/
│   └── creditcard.csv         # NOT included — see Setup below
│
├── src/
│   ├── __init__.py
│   ├── data_loading.py        # Step 1: load, missing values, class distribution
│   ├── preprocessing.py       # Step 2: feature/target split, stratified split
│   ├── model.py                # Step 3: Decision Tree training
│   ├── evaluation.py           # Step 4: confusion matrix, classification report
│   └── visualize_tree.py       # Step 5: decision tree plot
│
└── outputs/
    ├── confusion_matrix_bars.png
    └── decision_tree.png
```

## Setup & Usage

The dataset (~150 MB) is not included in this repository (too large for GitHub). Download it from Kaggle:

**[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**

```bash
git clone https://github.com/Salma-Rhomari/fraud-detection-credit-card.git
cd fraud-detection-credit-card
pip install -r requirements.txt

# Download creditcard.csv from the Kaggle link above and place it in data/

python main.py
```

This prints the classification report and confusion matrix, and saves both the confusion-matrix bar chart and the decision tree plot to `outputs/`.

## Technologies

- Python
- Pandas
- Scikit-learn (DecisionTreeClassifier, train_test_split, classification_report, confusion_matrix, plot_tree)
- Matplotlib
