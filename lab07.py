# Simple ML Comparison - Beginner Version

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


# -------------------------------
# 1. Load Dataset
# -------------------------------
data = pd.read_csv("P_DIQ_converted (1).csv")

print("Shape:", data.shape)

# assuming last column is target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

print("Classes:", y.unique())


# -------------------------------
# 2. Split Data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -------------------------------
# 3. Models
# -------------------------------
model_list = [
    ("Perceptron", Perceptron(max_iter=1000)),
    ("SVM", SVC()),
    ("Decision Tree", DecisionTreeClassifier(max_depth=5)),
    ("Random Forest", RandomForestClassifier(n_estimators=100)),
    ("AdaBoost", AdaBoostClassifier(n_estimators=100)),
    ("Gradient Boost", GradientBoostingClassifier(n_estimators=100)),
    ("Naive Bayes", GaussianNB()),
    ("MLP", MLPClassifier(max_iter=500))
]


# -------------------------------
# 4. Training + Results
# -------------------------------
results = []
trained = {}

for name, model in model_list:
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    precision = precision_score(y_test, test_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, test_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, test_pred, average="weighted", zero_division=0)

    results.append([
        name,
        round(train_acc, 4),
        round(test_acc, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4)
    ])

    trained[name] = model
    print("Completed:", name)


# -------------------------------
# 5. Results Table
# -------------------------------
df = pd.DataFrame(results, columns=[
    "Model", "Train Acc", "Test Acc", "Precision", "Recall", "F1"
])

df = df.set_index("Model")

print("\nResults (Train vs Test):\n")
print(df)


# -------------------------------
# 6. Plot
# -------------------------------
df[["Train Acc", "Test Acc"]].plot(kind="bar", figsize=(10,5))
plt.title("Train vs Test Accuracy")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# -------------------------------
# 7. Best Model
# -------------------------------
best_model_name = df["Test Acc"].idxmax()
best_model = trained[best_model_name]

print("\nBest Model:", best_model_name)

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Confusion Matrix - " + best_model_name)
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# -------------------------------
# 8. Cross Validation
# -------------------------------
print("\nCross Validation:\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in trained.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv)
    print(name, "Mean:", round(scores.mean(),4), "Std:", round(scores.std(),4))


# -------------------------------
# 9. Final Ranking
# -------------------------------
final = df.sort_values("Test Acc", ascending=False)

print("\nFinal Model Ranking:\n")
print(final)