"""
models/svm_model.py
-------------------
Traditional ML baseline: LinearSVC with TF-IDF features.  [6]

Design choices (per proposal):
  - LinearSVC with class_weight='balanced'
    → scales the SVM penalty inversely with class frequency,
      directly addressing the 13.4% spam imbalance  [6]
  - CalibratedClassifierCV (Platt scaling, sigmoid)
    → produces probability estimates required for PR-AUC
  - Hyperparameter C tuned via 5-fold cross-validated grid search
    on the training set only

Reference
---------
[6] X. Yang, Q. Song, and Y. Wang, "A weighted support vector machine
    for data classification," IJPRAI, vol. 21, no. 5, pp. 961-976, 2007.
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV


def train_svm(X_train, y_train, tune: bool = True):
    """
    Fit a balanced LinearSVC classifier.

    Parameters
    ----------
    X_train : sparse TF-IDF matrix
    y_train : int array
    tune    : if True, grid-search over C ∈ {0.01, 0.1, 1.0, 10.0}

    Returns
    -------
    Fitted CalibratedClassifierCV wrapping LinearSVC
    """
    base = LinearSVC(
        class_weight="balanced",   # addresses class imbalance [6]
        max_iter=5_000,
        dual=True,
    )
    calibrated = CalibratedClassifierCV(base, cv=3, method="sigmoid")

    if tune:
        print("  Grid-searching best C for SVM ...")
        gs = GridSearchCV(
            calibrated,
            param_grid={"estimator__C": [0.01, 0.1, 1.0, 10.0]},
            cv=5,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        best_C = gs.best_params_["estimator__C"]
        print(f"  Best C = {best_C}  |  CV F1 = {gs.best_score_:.4f}")
        return gs.best_estimator_
    else:
        calibrated.fit(X_train, y_train)
        return calibrated


def predict_svm(model, X):
    """
    Returns
    -------
    y_pred : int array
    y_prob : float array  (P(spam))
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return y_pred, y_prob
