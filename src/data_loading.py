"""
Step 1: Data Loading and Quality Check
------------------------------------------
Loads the raw credit card transactions dataset and performs two
crucial checks: verifying there are no missing values, and measuring
the class imbalance between normal (0) and fraudulent (1) transactions.
"""

import pandas as pd


def load_data(path="data/creditcard.csv"):
    """Load the credit card transactions dataset."""
    data = pd.read_csv(path)
    print("Are there any missing values in the dataset?")
    print(data.isnull().any().any())
    print("Data preview:")
    print(data.head())
    print("\nCount of normal and fraudulent transactions:")
    print(data["Class"].value_counts())
    return data


def class_distribution(data):
    """Print the percentage distribution of the target class."""
    dist = data["Class"].value_counts(normalize=True) * 100
    print(dist)
    return dist


if __name__ == "__main__":
    data = load_data()
    class_distribution(data)
