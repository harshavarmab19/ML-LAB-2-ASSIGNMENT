# -- IMPORTS --
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import lime
import lime.lime_tabular


# -- LOAD DATA --
def load_data():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    return X, y, data.feature_names


# -- SPLIT DATA --
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


# -- BUILD PIPELINE --
def build_pipeline():

    # Base models (A1 requirement)
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True))
    ]

    # Meta model
    final_model = LogisticRegression(max_iter=1000)

    # Stacking model
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=final_model
    )

    # Pipeline (A2 requirement)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', stacking_model)
    ])

    return pipeline


# -- TRAIN MODEL --
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


# -- EVALUATE MODEL --
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


# -- LIME EXPLANATION --
def explain_lime(model, X_train, X_test, feature_names):

    X_train_np = X_train.values
    X_test_np = X_test.values

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=feature_names,
        class_names=['Class 0', 'Class 1'],
        mode='classification'
    )

    explanation = explainer.explain_instance(
        data_row=X_test_np[0],
        predict_fn=model.predict_proba
    )

    return explanation.as_list()


# -- MAIN PROGRAM --
if __name__ == "__main__":

    # Load
    X, y, feature_names = load_data()

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Pipeline
    model_pipeline = build_pipeline()

    # Train
    trained_model = train_model(model_pipeline, X_train, y_train)

    # Accuracy
    accuracy = evaluate_model(trained_model, X_test, y_test)
    print("\nModel Accuracy:", accuracy)

    # LIME (A3 requirement)
    lime_output = explain_lime(trained_model, X_train, X_test, feature_names)

    print("\nLIME Explanation (Top Features):")
    for feature, weight in lime_output:
        print(f"{feature} --> {weight:.4f}")