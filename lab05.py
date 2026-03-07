import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score


# load dataset
def load_data("C:\Users\harsh\Downloads\P_DIQ_converted (1).csv"):

    data = pd.read_csv("C:\Users\harsh\Downloads\P_DIQ_converted (1).csv")

    # remove ID column if present
    if "SEQN" in data.columns:
        data = data.drop("SEQN", axis=1)

    # remove rows with missing values
    data = data.dropna()

    return data


# split X and y
def split_data(data):

    y = data["DIQ010"]
    X = data.drop("DIQ010", axis=1)

    return X, y


# train linear regression
def train_model(X_train, y_train):

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model


# calculate regression metrics
def get_metrics(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)

    rmse = np.sqrt(mse)

    mape = np.mean(abs((y_true - y_pred) / y_true)) * 100

    r2 = r2_score(y_true, y_pred)

    return mse, rmse, mape, r2


# kmeans clustering
def run_kmeans(X, k):

    kmeans = KMeans(n_clusters=k, random_state=42)

    kmeans.fit(X)

    labels = kmeans.labels_

    centers = kmeans.cluster_centers_

    return labels, centers


# clustering scores
def clustering_scores(X, labels):

    s = silhouette_score(X, labels)

    ch = calinski_harabasz_score(X, labels)

    db = davies_bouldin_score(X, labels)

    return s, ch, db


# check scores for different k
def check_k_values(X):

    k_list = range(2, 10)

    sil_scores = []
    ch_scores = []
    db_scores = []

    for k in k_list:

        kmeans = KMeans(n_clusters=k)

        kmeans.fit(X)

        labels = kmeans.labels_

        sil_scores.append(silhouette_score(X, labels))

        ch_scores.append(calinski_harabasz_score(X, labels))

        db_scores.append(davies_bouldin_score(X, labels))

    return k_list, sil_scores, ch_scores, db_scores


# elbow method
def elbow_plot(X):

    distortions = []

    k_range = range(2, 20)

    for k in k_range:

        kmeans = KMeans(n_clusters=k)

        kmeans.fit(X)

        distortions.append(kmeans.inertia_)

    return k_range, distortions


# main program
def main():

    data = load_data("P_DIQ_converted (1).csv")

    X, y = split_data(data)


    # A1 single attribute regression
    X_one = X[[X.columns[0]]]

    X_train, X_test, y_train, y_test = train_test_split(
        X_one, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)

    train_pred = model.predict(X_train)

    test_pred = model.predict(X_test)

    train_metrics = get_metrics(y_train, train_pred)

    test_metrics = get_metrics(y_test, test_pred)

    print("A1 Single Attribute Regression")
    print("Train:", train_metrics)
    print("Test:", test_metrics)


    # A3 multiple attribute regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)

    train_pred = model.predict(X_train)

    test_pred = model.predict(X_test)

    train_metrics = get_metrics(y_train, train_pred)

    test_metrics = get_metrics(y_test, test_pred)

    print("\nA3 Multiple Attribute Regression")
    print("Train:", train_metrics)
    print("Test:", test_metrics)


    # A4 kmeans clustering
    labels, centers = run_kmeans(X, 2)

    print("\nA4 KMeans Clustering")
    print("Cluster Centers:")
    print(centers)


    # A5 clustering scores
    sil, ch, db = clustering_scores(X, labels)

    print("\nA5 Clustering Scores")
    print("Silhouette:", sil)
    print("CH Score:", ch)
    print("DB Index:", db)


    # A6 k value evaluation
    k_vals, sil_scores, ch_scores, db_scores = check_k_values(X)

    plt.plot(k_vals, sil_scores)
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs k")
    plt.show()


    # A7 elbow method
    k_range, distortions = elbow_plot(X)

    plt.plot(k_range, distortions)
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.title("Elbow Method")
    plt.show()


if __name__ == "__main__":

    main()
