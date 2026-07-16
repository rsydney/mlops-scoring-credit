import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()

# Ré-entraîner le modèle gagnant avec ses meilleurs hyperparamètres
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model = LogisticRegression(max_iter=5000, C=10, random_state=42)
model.fit(X_train_smote, y_train_smote)

# SHAP pour un modèle linéaire (rapide, pas besoin d'échantillonner)
explainer = shap.LinearExplainer(model, X_train_smote)
shap_values = explainer.shap_values(X_test)

# Graphique résumé : impact de chaque feature sur la prédiction
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
print("Graphique sauvegardé : shap_summary.png")

# Importance moyenne absolue par feature (classement)
import numpy as np
importance = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("\nTop 10 features les plus importantes:")
print(importance.head(10).to_string(index=False))