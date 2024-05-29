import os
import pickle 
import mlflow
import argparse

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data_path")
parser.add_argument("--top_n")
args = parser.parse_args()

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ["max_depth", "n_estimators", "min_samples_split", "min_samples_leaf", "random_state"]

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    
def train_and_log_model(data_path: str, params):
    X_train, y_train = load_pickle(f"{data_path}/train.pkl")
    X_val, y_val = load_pickle(f"{data_path}/val.pkl")
    X_test, y_test = load_pickle(f"{data_path}/test.pkl")

    with mlflow.start_run():

        new_params = {}
        for param in RF_PARAMS:
            new_params[param] = int(params[param])

        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)

def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME) 
    
    runs = client.search_runs(
        experiment_ids = experiment.experiment_id,
        run_view_type = ViewType.ALL,
        max_results = top_n,
        order_by = ["metrics.rmse ASC"]
    )[0]
    
    for run in runs:
        train_and_log_model(data_path = data_path, params = run.data.params)

    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids = experiment.experiment_id,
        run_view_type = ViewType.ALL,
        max_results = top_n,
        order_by = ["metrics.rmse ASC"]
        )[0]
    

    run_id = best_run.info.run_id
    model_uri = f"runs/{run_id}/model"
    mlflow.register_model(
        model_uri,
        name="rf-best-model"
    )


if __name__ == "__main__":
    run_register_model(args.data_path, args.top_n)