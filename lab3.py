import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import minkowski

def dot_basic(a, b):
    s = 0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s

def len_basic(v):
    t = 0
    for x in v:
        t += x * x
    return math.sqrt(t)

def mean_l(lst):
    return sum(lst) / len(lst)

def var_l(lst):
    m = mean_l(lst)
    return sum((x - m) ** 2 for x in lst) / len(lst)

def center_grp(g):
    return np.mean(g, axis=0)

def center_dist(c1, c2):
    return len_basic(c1 - c2)

def mink_calc(a, b, p):
    t = 0
    for i in range(len(a)):
        t += abs(a[i] - b[i]) ** p
    return t ** (1/p)

def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=1)

def knn_train(X, y, k):
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X, y)
    return m

def knn_acc(m, X, y):
    return m.score(X, y)

def knn_pred(m, X):
    return m.predict(X)

def my_knn(X, y, pt, k):
    temp = []
    for i in range(len(X)):
        d = len_basic(X[i] - pt)
        temp.append((d, y[i]))
    temp.sort(key=lambda x: x[0])
    top = temp[:k]
    labels = [v for _, v in top]
    return max(set(labels), key=labels.count)

def conf_vals(a, p):
    TP = TN = FP = FN = 0
    for x, y in zip(a, p):
        if x == 1 and y == 1: TP += 1
        elif x == 0 and y == 0: TN += 1
        elif x == 0 and y == 1: FP += 1
        else: FN += 1
    return TP, TN, FP, FN

def reg_weights(X, y):
    ones = np.ones((X.shape[0], 1))
    A = np.hstack((ones, X))
    return np.linalg.inv(A.T @ A) @ A.T @ y

def dot_first(a, b):
    s = 0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s

def len_first(v):
    t = 0
    for x in v:
        t += x * x
    return t ** 0.5

def main():
    vA = np.array([2, 5, 7, 9])
    vB = np.array([1, 3, 4, 8])

    print("Manual Dot Product:", dot_first(vA, vB))
    print("NumPy Dot Product:", np.dot(vA, vB))
    print("Manual Length A:", len_first(vA))
    print("NumPy Length A:", np.linalg.norm(vA))
    print("Manual Length B:", len_first(vB))
    print("NumPy Length B:", np.linalg.norm(vB))

    X = np.array([[1,2],[2,3],[3,3],[6,5],[7,7],[8,6]])
    y = np.array([0,0,0,1,1,1])

    p = np.array([1,2,3])
    q = np.array([4,5,6])

    print("Dot custom:", dot_basic(p, q))
    print("Len custom:", len_basic(p))

    g0 = X[y == 0]
    g1 = X[y == 1]

    c0 = center_grp(g0)
    c1 = center_grp(g1)

    print("Center dist:", center_dist(c0, c1))
    print("Minkowski:", mink_calc(p, q, 2), minkowski(p, q, 2))

    Xt, Xs, yt, ys = split_data(X, y)

    model = knn_train(Xt, yt, 3)
    pred = knn_pred(model, Xs)

    print("kNN Acc:", knn_acc(model, Xs, ys))
    print("Custom KNN:", my_knn(Xt, yt, Xs[0], 3))

    for k in range(1, len(Xt)+1):
        mk = knn_train(Xt, yt, k)
        print("k =", k, "Acc =", knn_acc(mk, Xs, ys))

    TP, TN, FP, FN = conf_vals(ys, pred)
    print("TP TN FP FN:", TP, TN, FP, FN)

    print("Regression Weights:", reg_weights(Xt, yt))

main()
