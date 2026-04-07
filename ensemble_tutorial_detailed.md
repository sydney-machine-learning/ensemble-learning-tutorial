# Ensemble Learning Tutorial (ensemble.py)

## 1. Overview

This script benchmarks multiple machine learning models (including
ensemble methods) on classification or regression datasets.

It: - Loads data - Splits into training/testing sets - Trains multiple
models - Evaluates performance - Repeats experiments - Reports mean and
standard deviation

------------------------------------------------------------------------

## 2. Data Loading

Function: `read_data(run_num, prob)`

-   Classification → Pima Indians dataset
-   Regression → Energy efficiency dataset
-   Splits data: 60% train / 40% test

------------------------------------------------------------------------

## 3. Models Implemented

### Classification

-   MLPClassifier (Neural Network)
-   Decision Tree
-   Random Forest
-   AdaBoost
-   Gradient Boosting
-   XGBoost

### Regression

-   MLPRegressor
-   Decision Tree Regressor
-   Random Forest Regressor
-   AdaBoost Regressor
-   Gradient Boosting Regressor
-   XGBoost Regressor

------------------------------------------------------------------------

## 4. Training Pipeline

Each model: 1. Initialized 2. Trained using `.fit()` 3. Predictions made
using `.predict()` 4. Evaluated

------------------------------------------------------------------------

## 5. Evaluation Metrics

-   Classification → Accuracy
-   Regression → RMSE (Root Mean Squared Error)

------------------------------------------------------------------------

## 6. Experiment Loop

Runs multiple experiments:

    for run_num in range(max_expruns):

Purpose: - Reduce randomness - Improve reliability

------------------------------------------------------------------------

## 7. Results

Outputs: - Mean performance - Standard deviation

------------------------------------------------------------------------

## 8. Ensemble Methods Explained

-   Random Forest → Bagging of trees
-   AdaBoost → Focus on hard samples
-   Gradient Boosting → Sequential improvement
-   XGBoost → Optimized gradient boosting

------------------------------------------------------------------------

## 9. Known Issues

-   Typo: 'classifification' → 'classification'
-   Confusion matrix argument order incorrect
-   XGBoost objective deprecated
-   No feature scaling for neural networks

------------------------------------------------------------------------

## 10. How to Run

``` bash
python ensemble.py
```

Ensure datasets exist in `data/` folder.

------------------------------------------------------------------------

## 11. Improvements

-   Add feature scaling (StandardScaler)
-   Increase number of runs
-   Add visualization (plots)
-   Extend with more models

------------------------------------------------------------------------

## 12. Summary

This script is a simple benchmarking framework to compare machine
learning models across multiple runs and datasets.
