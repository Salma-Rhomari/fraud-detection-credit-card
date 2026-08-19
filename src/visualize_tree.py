"""
Step 5: Decision Tree Visualization
-----------------------------------------
The great advantage of Decision Trees over black-box models is
transparency. Plotting the tree (limited to max_depth=3 for
readability) lets us trace exactly which thresholds on which
variables (e.g. V14, V12, V17) led to a transaction being flagged
as fraud -- useful to explain a decision to a bank or a client.
"""

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


def plot_decision_tree(model, X, output_path="outputs/decision_tree.png", max_depth=3):
    plt.figure(figsize=(15, 10))
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=["Normal", "Fraud"],
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=10,
    )
    plt.title("Decision Tree")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Decision tree plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    from data_loading import load_data
    from preprocessing import split_features_target, train_test_split_stratified
    from model import train_decision_tree

    data = load_data()
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    tree = train_decision_tree(X_train, y_train)

    plot_decision_tree(tree, X)
