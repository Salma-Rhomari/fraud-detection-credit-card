"""
Step 2: Feature/Target Split and Stratified Train/Test Split
------------------------------------------------------------------
Because fraud represents only ~0.17% of transactions, a purely random
split risks placing too few (or zero) fraud cases in the test set.
Scikit-learn's train_test_split with stratify=y guarantees the exact
fraud ratio is preserved in both the training and test sets.

Note: Decision Trees are not sensitive to feature scale, so unlike
KNN or Logistic Regression, no StandardScaler is needed here.
"""

from sklearn.model_selection import train_test_split


def split_features_target(data):
    """Separate features (X) from the target label (y)."""
    X = data.drop("Class", axis=1)
    y = data["Class"]
    return X, y


def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    """Chronology doesn't matter here, but class balance does."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Number of transactions for training: {len(X_train)}")
    print(f"Number of transactions for testing: {len(X_test)}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from data_loading import load_data

    data = load_data()
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
