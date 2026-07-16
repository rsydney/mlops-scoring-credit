import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from business_score import (
    business_cost,
    business_score_normalized,
)

# ==========================================================
# Chargement des données
# ==========================================================

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# ==========================================================
# Gestion du déséquilibre avec SMOTE
# ==========================================================

print("Distribution avant SMOTE :", np.bincount(y_train))

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("Distribution après SMOTE :", np.bincount(y_train_smote))

# ==========================================================
# Configuration MLflow
# ==========================================================

mlflow.set_experiment("credit_scoring_models")


# ==========================================================
# Fonction générique
# ==========================================================

def train_and_log(model, param_grid, model_name, X_tr, y_tr, use_smote=False):

    with mlflow.start_run(run_name=model_name):

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
        )

        grid.fit(X_tr, y_tr)

        best_model = grid.best_estimator_

        # -------------------------
        # Prédictions
        # -------------------------

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # -------------------------
        # Métriques
        # -------------------------

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        cost = business_cost(y_test, y_pred)
        cost_norm = business_score_normalized(y_test, y_pred)

        # -------------------------
        # Logging MLflow
        # -------------------------

        mlflow.log_param("model_type", model_name)
        mlflow.log_param("use_smote", use_smote)

        mlflow.log_params(grid.best_params_)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        mlflow.log_metric("business_cost_total", cost)
        mlflow.log_metric("business_cost_normalized", cost_norm)

        # ==========================================
        # Enregistrement du modèle
        # ==========================================

        if isinstance(best_model, XGBClassifier):
            mlflow.xgboost.log_model(
                xgb_model=best_model,
                name="model"
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="model"
            )

        # -------------------------
        # Affichage
        # -------------------------

        print(f"\n===== {model_name} =====")
        print("Meilleurs paramètres :", grid.best_params_)
        print(f"Accuracy : {acc:.3f}")
        print(f"Precision : {precision:.3f}")
        print(f"Recall : {recall:.3f}")
        print(f"F1-score : {f1:.3f}")
        print(f"ROC-AUC : {auc:.3f}")
        print(f"Coût métier : {cost}")
        print(f"Coût métier normalisé : {cost_norm:.3f}")

        return best_model, cost_norm


# ==========================================================
# Logistic Regression
# ==========================================================

lr_params = {
    "C": [0.01, 0.1, 1, 10]
}

lr_model, lr_cost = train_and_log(
    LogisticRegression(
        max_iter=5000,
        random_state=42,
    ),
    lr_params,
    "LogisticRegression_SMOTE",
    X_train_smote,
    y_train_smote,
    use_smote=True,
)

# ==========================================================
# Random Forest
# ==========================================================

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
}

rf_model, rf_cost = train_and_log(
    RandomForestClassifier(
        random_state=42
    ),
    rf_params,
    "RandomForest_SMOTE",
    X_train_smote,
    y_train_smote,
    use_smote=True,
)

# ==========================================================
# XGBoost
# ==========================================================

xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
}

xgb_model, xgb_cost = train_and_log(
    XGBClassifier(
        random_state=42,
        eval_metric="logloss",
    ),
    xgb_params,
    "XGBoost_SMOTE",
    X_train_smote,
    y_train_smote,
    use_smote=True,
)

# ==========================================================
# Comparaison finale
# ==========================================================

print("\n")
print("=" * 60)
print("COMPARAISON FINALE")
print("=" * 60)

results = {
    "LogisticRegression": lr_cost,
    "RandomForest": rf_cost,
    "XGBoost": xgb_cost,
}

for model, score in sorted(results.items(), key=lambda x: x[1]):
    print(f"{model:<20} {score:.4f}")

best_model_name = min(results, key=results.get)

print("\n🏆 Meilleur modèle selon le score métier :", best_model_name)