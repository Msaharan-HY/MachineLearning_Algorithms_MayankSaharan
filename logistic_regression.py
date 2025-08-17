# 1_logistic_regression.py
"""
Logistic Regression classifier on the Iris dataset.
Uses scikit-learn's built-in Iris data for multiclass classification.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def main():
    iris = load_iris()
    X, y = iris.data, iris.target

    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Logistic Regression (Iris) Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

if __name__ == "__main__":
    main()
