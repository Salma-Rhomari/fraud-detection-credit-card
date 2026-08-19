"""
Step 4: Model Evaluation
------------------------------
Accuracy alone is a trap on imbalanced data: a naive model predicting
"normal" for every transaction would still score ~99.8% accuracy while
missing 100% of fraud. Instead, this module reports the confusion
matrix, classification report (precision/recall/F1 per class), and a
readable bar chart of the four confusion-matrix quadrants.

Confusion matrix quadrants:
    TN (True Negatives):  normal transactions correctly left alone
    FP (False Positives): honest customers flagged by mistake
    FN (False Negatives): real fraud that slipped through
    TP (True Positives):  fraud correctly intercepted
"""

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    print("\n--- CONFUSION MATRIX ---")
    print(confusion_matrix(y_test, predictions))

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, predictions, target_names=["Normal", "Fraud"]))

    return predictions


def plot_confusion_matrix(y_test, predictions, output_path="outputs/confusion_matrix_bars.png"):
    """Bar chart of TN / FP / FN / TP on a log scale (classes are very imbalanced)."""
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    bar_names = [
        "True Negatives\n(Good customers)",
        "False Positives\n(False alarms)",
        "False Negatives\n(Missed fraud)",
        "True Positives\n(Fraud blocked)",
    ]
    values = [tn, fp, fn, tp]
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#27ae60"]

    # log scale can't represent 0, so use a small floor just for plotting height
    plot_values = [v if v > 0 else 0.5 for v in values]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(bar_names, plot_values, color=colors)
    for bar, real_value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f"{int(real_value)}",
                  va="bottom", ha="center", fontweight="bold", fontsize=12)

    plt.title("Fraud Detection Results (Decision Tree)", fontsize=15, fontweight="bold")
    plt.ylabel("Number of transactions", fontsize=12)
    plt.yscale("log")
    plt.ylim(top=max(values) * 10)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix chart saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    from data_loading import load_data
    from preprocessing import split_features_target, train_test_split_stratified
    from model import train_decision_tree

    data = load_data()
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    tree = train_decision_tree(X_train, y_train)

    predictions = evaluate_model(tree, X_test, y_test)
    plot_confusion_matrix(y_test, predictions)
