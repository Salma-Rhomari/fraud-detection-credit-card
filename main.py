"""
Entry point for the full Credit Card Fraud Detection (Decision Tree) pipeline.

Run with:  python main.py

Requires data/creditcard.csv -- see README.md for download instructions
(the dataset is too large to be stored in this repository).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loading import load_data, class_distribution
from preprocessing import split_features_target, train_test_split_stratified
from model import train_decision_tree
from evaluation import evaluate_model, plot_confusion_matrix
from visualize_tree import plot_decision_tree


def main():
    # Step 1: Load & inspect data
    data = load_data("data/creditcard.csv")
    class_distribution(data)

    # Step 2: Feature/target split, stratified train/test split
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    # Step 3: Train the Decision Tree
    tree = train_decision_tree(X_train, y_train)

    # Step 4: Evaluate
    predictions = evaluate_model(tree, X_test, y_test)
    plot_confusion_matrix(y_test, predictions)

    # Step 5: Visualize the tree
    plot_decision_tree(tree, X)


if __name__ == "__main__":
    main()
