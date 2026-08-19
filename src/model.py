"""
Step 3: Decision Tree Training
------------------------------------
The classifier is configured with two key hyperparameters to handle
the extreme class imbalance:

- criterion='entropy':      the tree splits nodes to maximize
                             Information Gain (purity), using Shannon
                             entropy as the impurity measure.
- class_weight='balanced':  penalizes misclassifying the minority
                             class (fraud) much more heavily than the
                             majority class, preventing the tree from
                             lazily predicting "normal" for everything.
"""

from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(X_train, y_train, criterion="entropy", class_weight="balanced", random_state=42):
    tree = DecisionTreeClassifier(
        criterion=criterion,
        class_weight=class_weight,
        random_state=random_state,
    )
    tree.fit(X_train, y_train)
    print("\nThe tree has finished training!")
    return tree


if __name__ == "__main__":
    from data_loading import load_data
    from preprocessing import split_features_target, train_test_split_stratified

    data = load_data()
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    tree = train_decision_tree(X_train, y_train)
