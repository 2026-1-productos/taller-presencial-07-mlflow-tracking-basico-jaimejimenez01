"""Prediccion script for the MLflow model.

This script loads the best model from the MLflow experiment and makes
predictions on the full dataset, ordered by lowest test_mse.

$ python3 make_predictions.py

"""

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

FILE_PATH = "data/winequality-red.csv"
EXPERIMENT_NAME = "wine_quality_experiment"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
x = df.drop(columns=["quality"])

client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError(
        f"Experimento '{EXPERIMENT_NAME}' no encontrado. "
        "Ejecuta primero: python3 -m homework --model elasticnet"
    )

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_mse ASC"],
    max_results=1,
)
if not runs:
    raise RuntimeError("No hay runs en el experimento. Ejecuta el pipeline primero.")

best_run = runs[0]
run_id = best_run.info.run_id
logged_model = f"runs:/{run_id}/model"

print(f"Cargando modelo del run: {run_id}")
print(f"  test_mse : {best_run.data.metrics.get('test_mse', 'N/A'):.4f}")
print(f"  test_mae : {best_run.data.metrics.get('test_mae', 'N/A'):.4f}")
print(f"  test_r2  : {best_run.data.metrics.get('test_r2', 'N/A'):.4f}")

loaded_model = mlflow.pyfunc.load_model(logged_model)
predictions = loaded_model.predict(x)

print("\nPredicciones (primeras 10):")
print(predictions[:10])
