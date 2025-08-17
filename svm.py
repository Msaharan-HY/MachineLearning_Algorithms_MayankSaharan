# 4_svm.py
"""
Support Vector Machine (SVM) classifier on Breast Cancer dataset.
Uses scikit-learn's built-in data; classifies tumors as benign/malignant.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def main():
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("SVM (Breast Cancer) Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=cancer.target_names))

if __name__ == "__main__":
    main()
