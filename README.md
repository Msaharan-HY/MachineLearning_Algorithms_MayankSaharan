# Machine Learning Algorithms – Mayank Saharan

This repository demonstrates five classic **machine learning algorithms** implemented in Python using **real-world datasets**.  
The project is designed to practice **good coding practices, dataset handling, reproducibility, and public project documentation**.

---

## 📑 Table of Contents

- [Project Overview](#project-overview)  
- [Motivation](#motivation)  
- [Datasets](#datasets)  
- [Repository Structure](#repository-structure)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
- [Usage](#usage)  
- [Results & Evaluation](#results--evaluation)  
- [Known Issues & Future Work](#known-issues--future-work)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)

---

## 📘 Project Overview

The repository contains **five Python scripts**, each implementing a different supervised ML algorithm.  
Each script:
- Loads a **real dataset** (built-in or external).  
- Splits into **training and test sets**.  
- Trains a **model**.  
- Evaluates using **accuracy and classification reports**.  

This project is beginner-friendly and aims to demonstrate ML workflows step by step.

---

## 🎯 Motivation

- To **learn and practice** machine learning concepts with real datasets.  
- To **compare algorithms** across different domains (biology, medicine, text, images).  
- To **build reproducible code** with clear structure and documentation.  
- To showcase work on **GitHub for collaboration and learning**.  

---

## 📂 Datasets

| Script                    | Algorithm                  | Dataset              | Source / Access |
|----------------------------|----------------------------|----------------------|-----------------|
| `1_logistic_regression.py` | Logistic Regression        | Iris                 | [UCI Repository](https://archive.ics.uci.edu/ml/datasets/iris) / scikit-learn built-in |
| `2_decision_tree.py`       | Decision Tree              | Titanic              | [Kaggle](https://www.kaggle.com/competitions/titanic) or [Calmcode CSV](https://calmcode.io/static/data/titanic.csv) |
| `3_knn.py`                 | K-Nearest Neighbors (KNN)  | Digits (handwritten) | scikit-learn built-in |
| `4_svm.py`                 | Support Vector Machine     | Breast Cancer        | [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) / scikit-learn built-in |
| `5_random_forest.py`       | Random Forest              | Wine                 | [UCI Repository](https://archive.ics.uci.edu/ml/datasets/wine) / scikit-learn built-in |

👉 Only the **Titanic dataset** needs to be downloaded manually (`titanic.csv` into `data/` folder).  
All others come built-in with scikit-learn.

---

## 📁 Repository Structure

MachineLearning_Algorithms_MayankSaharan/

|-- 1_logistic_regression.py

|-- 2_decision_tree.py

|-- 3_knn.py

|-- 4_svm.py

|-- 5_random_forest.py

|-- README.md

🚀 Usage

Run any of the scripts from the command line:

- python logistic_regression.py
- python decision_tree.py
- python knn.py
- python svm.py
- python random_forest.py

Each script will:
- Train the model
- Print accuracy score
- Print a classification report
- (Decision Tree also prints model rules)

📊 Results & Evaluation:
Sample results (may vary slightly due to randomness):
- Logistic Regression (Iris): ~97%
- Decision Tree (Titanic): ~80%
- KNN (Digits): ~98%
- SVM (Breast Cancer): ~96%
- Random Forest (Wine): ~99%

🔧 Known Issues & Future Work
- Titanic preprocessing: Currently uses .dropna() → could improve with better missing value handling.
- Hyperparameter tuning: Models use defaults; future versions could implement GridSearchCV or RandomizedSearchCV.
- Visualization: Adding confusion matrices, ROC curves, or feature importance plots would make outputs more interpretable.
