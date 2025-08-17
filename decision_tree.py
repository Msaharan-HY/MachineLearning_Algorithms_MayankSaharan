# 2_decision_tree.py
"""
Decision Tree classifier on the Titanic dataset.
Requires a 'titanic.csv' file in data/ directory.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

def main():
    df = pd.read_csv("data/titanic.csv")

    df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Decision Tree (Titanic) Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nDecision Tree Rules:\n", export_text(model, feature_names=list(X.columns)))

if __name__ == "__main__":
    main()
