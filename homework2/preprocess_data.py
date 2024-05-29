import argparse
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--raw_data_path", type=str)
parser.add_argument("--dest_path", type=str)
args = parser.parse_args()

def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df

def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool = False):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

def run_data_prep(raw_data_path: str, dest_path: str, dataset: str = "green"):
    df_train = read_dataframe(f"{raw_data_path}/{dataset}_tripdata_2024-01.parquet")
    df_val = read_dataframe(f"{raw_data_path}/{dataset}_tripdata_2024-02.parquet")
    df_test = read_dataframe(f"{raw_data_path}/{dataset}_tripdata_2024-03.parquet")

    target = "duration"
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_val[target].values

    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv)
    X_test, _ = preprocess(df_test, dv)

    os.makedirs(dest_path, exist_ok=True)

    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))

if __name__ == "__main__":
    run_data_prep(args.raw_data_path, args.dest_path)