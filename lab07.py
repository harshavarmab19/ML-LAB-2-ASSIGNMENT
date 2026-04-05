import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


data = pd.read_csv("P_DIQ_converted (1).csv")

print("Shape:", data.shape)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Classes:", y.unique())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


p_model = RandomizedSearchCV(
    Perceptron(),
    {"alpha": [0.0001, 0.001, 0.01], "max_iter": [500, 1000]},
    n_iter=5, cv=cv
)
p_model.fit(X_train, y_train)


rf_model = RandomizedSearchCV(
    RandomForestClassifier(),
    {"n_estimators": [50, 100, 150], "max_depth": [None, 5, 10]},
    n_iter=5, cv=cv
)
rf_model.fit(X_train, y_train)


svm_model = RandomizedSearchCV(
    SVC(probability=True),
    {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    n_iter=5, cv=cv
)
svm_model.fit(X_train, y_train)


models = [
    ("Perceptron", p_model.best_estimator_),
    ("SVM", svm_model.best_estimator_),
    ("Decision Tree", DecisionTreeClassifier(max_depth=5)),
    ("Random Forest", rf_model.best_estimator_),
    ("AdaBoost", AdaBoostClassifier(n_estimators=100)),
    ("Gradient Boost", GradientBoostingClassifier(n_estimators=100)),
    ("Naive Bayes", GaussianNB()),
    ("MLP", MLPClassifier(max_iter=500))
]


results = []
saved = {}

for name, model in models:
    model.fit(X_train, y_train)

    p1 = model.predict(X_train)
    p2 = model.predict(X_test)

    a1 = accuracy_score(y_train, p1)
    a2 = accuracy_score(y_test, p2)

    pr = precision_score(y_test, p2, average="weighted", zero_division=0)
    rc = recall_score(y_test, p2, average="weighted", zero_division=0)
    f1 = f1_score(y_test, p2, average="weighted", zero_division=0)

    results.append([name, round(a1,4), round(a2,4), round(pr,4), round(rc,4), round(f1,4)])

    saved[name] = model
    print("Done:", name)


df = pd.DataFrame(results, columns=["Model","Train Acc","Test Acc","Precision","Recall","F1"])
df = df.set_index("Model")

print("\nResults:\n")
print(df)


df[["Train Acc","Test Acc"]].plot(kind="bar", figsize=(10,5))
plt.title("Accuracy Comparison")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


best = df["Test Acc"].idxmax()
m = saved[best]

print("\nBest Model:", best)

pred = m.predict(X_test)

cm = confusion_matrix(y_test, pred)
d = ConfusionMatrixDisplay(cm)
d.plot()
plt.title("Confusion Matrix - " + best)
plt.show()

print("\nReport:\n")
print(classification_report(y_test, pred))


print("\nCross Validation:\n")

for name, model in saved.items():
    sc = cross_val_score(model, X_train, y_train, cv=cv)
    print(name, "Mean:", round(sc.mean(),4), "Std:", round(sc.std(),4))


final = df.sort_values("Test Acc", ascending=False)

print("\nFinal Ranking:\n")
print(final)
