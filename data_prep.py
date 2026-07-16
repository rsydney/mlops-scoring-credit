import pandas as pd
import numpy as np

# Charger les données
df = pd.read_csv("data/UCI_Credit_Card.csv", index_col="ID")

print("Shape:", df.shape)
print("\nValeurs manquantes:\n", df.isnull().sum().sum(), "au total")
print("\nDistribution de la cible:\n", df["default.payment.next.month"].value_counts(normalize=True))

# Renommer la cible pour plus de simplicité
df = df.rename(columns={"default.payment.next.month": "target"})

# --- Nettoyage ---

# 1. EDUCATION : les valeurs 0, 5, 6 ne sont pas documentées (hors 1-4) -> on les regroupe dans "autres" (4)
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

# 2. MARRIAGE : la valeur 0 n'est pas documentée (hors 1-3) -> on la regroupe dans "autres" (3)
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

# 3. Vérifier les outliers sur AGE
print("\nAge min/max:", df["AGE"].min(), "/", df["AGE"].max())

# --- Feature engineering simple ---

# Ratio d'utilisation du crédit (montant facturé / limite de crédit)
df["BILL_TO_LIMIT_RATIO"] = df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1)

# Moyenne des montants payés sur les 6 derniers mois
pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
df["AVG_PAY_AMT"] = df[pay_amt_cols].mean(axis=1)

# Nombre de mois en retard de paiement (PAY_0 à PAY_6 : valeurs > 0 = retard)
pay_status_cols = ["PAY_0"] + [f"PAY_{i}" for i in range(2, 7)]
df["MONTHS_LATE"] = (df[pay_status_cols] > 0).sum(axis=1)

print("\nShape après feature engineering:", df.shape)

# --- Split train/test ---
from sklearn.model_selection import train_test_split

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Sauvegarder pour les prochaines étapes
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("\nDonnées préparées et sauvegardées ✅")