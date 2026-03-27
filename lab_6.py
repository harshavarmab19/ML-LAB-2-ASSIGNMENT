import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import numpy as np


# A1
def calculate_entropy(data, target_col):
    total = len(data)
    counts = {}

    for val in data[target_col]:
        counts[val] = counts.get(val, 0) + 1

    entropy = 0
    for key in counts:
        p = counts[key] / total
        entropy -= p * math.log(p, 2)

    return entropy


# A2
def calculate_gini(data, target_col):
    total = len(data)
    counts = {}

    for val in data[target_col]:
        counts[val] = counts.get(val, 0) + 1

    gini = 1
    for key in counts:
        p = counts[key] / total
        gini -= p * p

    return gini


# A4
def equal_width_binning(data, column, bins=4):
    col_data = data[column]
    min_val = col_data.min()
    max_val = col_data.max()
    width = (max_val - min_val) / bins

    new_col = []
    for val in col_data:
        if pd.isna(val):
            new_col.append("missing")
        else:
            idx = int((val - min_val) / width)
            if idx >= bins:
                idx = bins - 1
            new_col.append("bin_" + str(idx))

    return new_col


# A3
def information_gain(data, feature, target_col):
    total_entropy = calculate_entropy(data, target_col)
    total = len(data)

    values = data[feature].unique()
    weighted_entropy = 0

    for val in values:
        subset = data[data[feature] == val]
        weight = len(subset) / total
        weighted_entropy += weight * calculate_entropy(subset, target_col)

    return total_entropy - weighted_entropy


# A3
def best_feature(data, features, target_col):
    best = None
    best_gain = -1

    for feature in features:
        gain = information_gain(data, feature, target_col)
        if gain > best_gain:
            best_gain = gain
            best = feature

    return best


# A5
def build_tree(data, features, target_col):
    if len(data[target_col].unique()) == 1:
        return data[target_col].iloc[0]

    if len(features) == 0:
        return data[target_col].mode()[0]

    root = best_feature(data, features, target_col)
    tree = {root: {}}

    for val in data[root].unique():
        subset = data[data[root] == val]
        new_features = [f for f in features if f != root]
        tree[root][val] = build_tree(subset, new_features, target_col)

    return tree


# TABLE PRINT FUNCTION
def print_table(title, df):
    print("\n" + title.center(60))
    print("-" * 60)

    headers = ["Index"] + list(df.columns[:4])
    print("{:<6} {:<10} {:<10} {:<10} {:<10}".format(*headers))

    for i, row in df.iterrows():
        values = [i] + [str(v) if pd.notna(v) else "NA" for v in row.values[:4]]
        print("{:<6} {:<10} {:<10} {:<10} {:<10}".format(*values))


def main():

    df = pd.read_csv(r"C:\Users\harsh\Downloads\P_DIQ_converted (1).csv")
    df = df.drop(columns=["SEQN"], errors="ignore")

    target_col = df.columns[-1]

    # TABLE I
    print_table("TABLE I - ORIGINAL DATA", df.head())

    # A4 - Binning
    for col in df.columns:
        if col != target_col and df[col].dtype != 'object':
            df[col] = equal_width_binning(df, col, 4)

    # TABLE II
    print_table("TABLE II - AFTER BINNING", df.head())

    # A1 & A2
    entropy_value = calculate_entropy(df, target_col)
    gini_value = calculate_gini(df, target_col)

    print("\nEntropy:", round(entropy_value, 4))
    print("Gini Index:", round(gini_value, 4))

    # A3
    features = [col for col in df.columns if col != target_col]
    best = best_feature(df, features, target_col)
    print("\nBest Feature:", best)

    # A5
    tree = build_tree(df, features, target_col)
    print("\nDecision Tree Created")

    # A6
    df_encoded = df.copy()
    le = LabelEncoder()

    for col in df_encoded.columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]

    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(X, y)

    plt.figure(figsize=(12, 6))
    plot_tree(clf, feature_names=X.columns, filled=True)
    plt.savefig("decision_tree.png")
    plt.show()

    # A7
    if X.shape[1] >= 2:
        X2 = X.iloc[:, :2]

        clf2 = DecisionTreeClassifier()
        clf2.fit(X2, y)

        x_min, x_max = X2.iloc[:, 0].min() - 1, X2.iloc[:, 0].max() + 1
        y_min, y_max = X2.iloc[:, 1].min() - 1, X2.iloc[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.1),
            np.arange(y_min, y_max, 0.1)
        )

        Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X2.iloc[:, 0], X2.iloc[:, 1], c=y)

        plt.xlabel(X2.columns[0])
        plt.ylabel(X2.columns[1])

        plt.savefig("decision_boundary.png")
        plt.show()


if __name__ == "__main__":
    main()