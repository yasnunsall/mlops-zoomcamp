import os
import pickle
import argparse
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default = "./output")
args = parser.parse_args()

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def run_train(data_path: str):
    
    X_train, y_train = load_pickle(f"{data_path}/train.pkl")
    X_val, y_val = load_pickle(f"{data_path}/val.pkl")

    rf = RandomForestRegressor(max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred, squared=False)

if __name__ == "__main__":
        run_train(args.data_path)