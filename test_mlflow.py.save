import mlflow

mlflow.set_experiment("test_scoring_credit")

with mlflow.start_run():
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.log_param("C", 1.0)
    
    accuracy = 0.85  # valeur factice pour tester
    mlflow.log_metric("accuracy", accuracy)
    
    with open("test_artifact.txt", "w") as f:
        f.write("Ceci est un artefact de test")
    mlflow.log_artifact("test_artifact.txt")

print("Run enregistré avec succès !")
