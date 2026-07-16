import pandas as pd
from sklearn.linear_model import LogisticRegression
from business_score import business_cost, business_score_normalized, print_cost_breakdown

# Charger les données préparées au Jour 2
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# Modèle baseline simple
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("=== Évaluation avec le score métier ===\n")
print_cost_breakdown(y_test, y_pred)