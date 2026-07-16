import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()

# SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Scaling (corrige le warning de convergence vu au Jour 4)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)

# Modèle final avec les meilleurs hyperparamètres trouvés au Jour 4
model = LogisticRegression(max_iter=5000, C=10, random_state=42)
model.fit(X_train_scaled, y_train_smote)

# Sauvegarder le modèle ET le scaler (indispensable, l'API devra scaler les nouvelles données pareil)
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X_train.columns), "feature_names.pkl")  # pour valider l'ordre des colonnes en entrée

print("Modèle, scaler et noms de features sauvegardés ✅")