import yaml
import numpy as np

def validate_schema(df):
    if df.isnull().any().any():
        raise ValueError("Data contains missing values.")
    if not all(np.issubdtype(dtype, np.number) for dtype in df.dtypes):
        raise TypeError("All columns must be numeric.")
    return True

def parse_schema(df):
    schema = {}
    for col in df.columns:
        dtype = df[col].dtype
        if np.issubdtype(dtype, np.number):
            schema[col] = "numeric"
        elif np.issubdtype(dtype, np.datetime64):
            schema[col] = "datetime"
        else:
            schema[col] = "categorical"
    return schema

def load_yaml_schema(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_yaml_schema(schema, path):
    with open(path, "w") as f:
        yaml.dump(schema, f)
