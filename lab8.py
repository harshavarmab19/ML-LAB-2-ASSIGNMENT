import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------- BASIC FUNCTIONS ----------------

def summation(x, w, b):
    total = 0
    for i in range(len(x)):
        total = total + x[i] * w[i]
    total = total + b
    return total


def activation(x, typ):
    if typ == "step":
        if x >= 0:
            return 1
        else:
            return 0

    if typ == "sigmoid":
        return 1 / (1 + math.exp(-x))

    if typ == "relu":
        if x > 0:
            return x
        else:
            return 0

    if typ == "bipolar":
        if x >= 0:
            return 1
        else:
            return -1


# ---------------- A2 ----------------

def A2_and_gate():

    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0,0,0,1]

    w = [0.2, -0.75]
    b = 10
    lr = 0.05

    errors = []

    for epoch in range(1000):

        total_error = 0

        for i in range(len(X)):

            net = summation(X[i], w, b)
            out = activation(net, "step")

            err = Y[i] - out
            total_error = total_error + err * err

            w[0] = w[0] + lr * err * X[i][0]
            w[1] = w[1] + lr * err * X[i][1]
            b = b + lr * err

        errors.append(total_error)

        if total_error <= 0.002:
            break

    plt.plot(errors)
    plt.title("AND Error")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.show()   # ✅ FIXED (graph stays)

    return w, b, epoch


# ---------------- A3 ----------------

def A3_compare_activation():

    acts = ["bipolar","sigmoid","relu"]
    result = []

    for a in acts:

        X = [[0,0],[0,1],[1,0],[1,1]]
        Y = [0,0,0,1]

        w = [0.2, -0.75]
        b = 10
        lr = 0.05

        epoch = 0

        while epoch < 100:

            total_error = 0

            for i in range(len(X)):

                net = summation(X[i], w, b)
                out = activation(net, a)

                err = Y[i] - out
                total_error = total_error + err * err

                w[0] = w[0] + lr * err * X[i][0]
                w[1] = w[1] + lr * err * X[i][1]
                b = b + lr * err

            if total_error <= 0.002:
                break

            epoch = epoch + 1

        result.append((a, epoch))

    return result


# ---------------- A4 ----------------

def A4_learning_rate():

    rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    output = []

    for lr in rates:

        X = [[0,0],[0,1],[1,0],[1,1]]
        Y = [0,0,0,1]

        w = [0.2, -0.75]
        b = 10

        epoch = 0

        while epoch < 100:

            total_error = 0

            for i in range(len(X)):

                net = summation(X[i], w, b)
                out = activation(net, "step")

                err = Y[i] - out
                total_error = total_error + err * err

                w[0] = w[0] + lr * err * X[i][0]
                w[1] = w[1] + lr * err * X[i][1]
                b = b + lr * err

            if total_error <= 0.002:
                break

            epoch = epoch + 1

        output.append((lr, epoch))

    plt.plot(rates, [x[1] for x in output])
    plt.title("Learning Rate vs Epochs")
    plt.xlabel("Learning Rate")
    plt.ylabel("Epochs")
    plt.show()   # ✅ FIXED (graph stays)

    return output


# ---------------- A5 ----------------

def A5_xor_gate():
    return "XOR not linearly separable"


# ---------------- A6 ----------------

def A6_dataset():

    data = pd.read_csv(r"C:\Users\harsh\Downloads\P_DIQ_converted (1).csv")

    X = data.select_dtypes(include=[np.number])
    X = X.fillna(X.mean())

    Y = data.iloc[:, -1]

    newY = []
    for i in range(len(Y)):
        val = Y[i]

        if type(val) == str:
            if val.lower() in ["yes","high","true","1"]:
                newY.append(1)
            else:
                newY.append(0)
        else:
            if val > 0:
                newY.append(1)
            else:
                newY.append(0)

    X = X.values

    Xn = []
    for i in range(len(X)):
        row = []
        for j in range(len(X[0])):
            minv = np.min(X[:, j])
            maxv = np.max(X[:, j])

            if maxv - minv == 0:
                row.append(0)
            else:
                row.append((X[i][j] - minv) / (maxv - minv))
        Xn.append(row)

    w = [0.1] * len(Xn[0])
    b = 0.1
    lr = 0.01

    for epoch in range(100):
        for i in range(len(Xn)):

            net = summation(Xn[i], w, b)
            out = activation(net, "sigmoid")

            err = newY[i] - out

            for j in range(len(w)):
                w[j] = w[j] + lr * err * Xn[i][j]

            b = b + lr * err

    correct = 0
    for i in range(len(Xn)):
        out = activation(summation(Xn[i], w, b), "sigmoid")
        pred = 1 if out >= 0.5 else 0

        if pred == newY[i]:
            correct = correct + 1

    acc = correct / len(newY)

    return w, b, acc


# ---------------- A7 ----------------

def A7_pseudo_inverse():

    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0,0,0,1]

    X_aug = []
    for i in range(len(X)):
        X_aug.append([X[i][0], X[i][1], 1])

    X_aug = np.array(X_aug)
    Y = np.array(Y)

    pinv = np.linalg.pinv(X_aug)
    W = pinv.dot(Y)

    return W


# ---------------- A8 ----------------

def A8_backprop_and():

    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0,0,0,1]

    w1 = 0.5
    w2 = -0.5
    b = 0.1
    lr = 0.05

    for epoch in range(200):

        for i in range(len(X)):

            net = X[i][0]*w1 + X[i][1]*w2 + b
            out = 1/(1+math.exp(-net))

            err = Y[i] - out
            d = err * out * (1-out)

            w1 = w1 + lr*d*X[i][0]
            w2 = w2 + lr*d*X[i][1]
            b = b + lr*d

    return w1, w2, b


# ---------------- A9 ----------------

def A9_xor_backprop():
    return "Needs hidden layer"


# ---------------- A10 ----------------

def A10_two_outputs():
    return "[1,0] and [0,1] mapping"


# ---------------- A11 ----------------

def A11_mlp():

    from sklearn.neural_network import MLPClassifier

    X = [[0,0],[0,1],[1,0],[1,1]]

    model = MLPClassifier(hidden_layer_sizes=(3,), max_iter=2000)

    model.fit(X, [0,0,0,1])
    out1 = model.predict(X)

    model.fit(X, [0,1,1,0])
    out2 = model.predict(X)

    return out1, out2


# ---------------- A12 ----------------

def A12_mlp_dataset():

    from sklearn.neural_network import MLPClassifier

    data = pd.read_csv(r"C:\Users\harsh\Downloads\P_DIQ_converted (1).csv")

    X = data.select_dtypes(include=[np.number])
    X = X.fillna(X.mean())

    Y = data.iloc[:, -1]

    newY = []
    for i in range(len(Y)):
        if type(Y[i]) == str:
            if Y[i].lower() in ["yes","high","true","1"]:
                newY.append(1)
            else:
                newY.append(0)
        else:
            newY.append(1 if Y[i] > 0 else 0)

    X = X.values
    Xn = X / (np.max(X, axis=0) + 1e-6)

    model = MLPClassifier(hidden_layer_sizes=(4,), max_iter=2000)
    model.fit(Xn, newY)

    return model.predict(Xn)


# ---------------- MAIN ----------------

if __name__ == "__main__":

    print("A2:", A2_and_gate())
    print("A3:", A3_compare_activation())
    print("A4:", A4_learning_rate())
    print("A5:", A5_xor_gate())
    print("A6:", A6_dataset())
    print("A7:", A7_pseudo_inverse())
    print("A8:", A8_backprop_and())
    print("A9:", A9_xor_backprop())
    print("A10:", A10_two_outputs())
    print("A11:", A11_mlp())
    print("A12:", A12_mlp_dataset())