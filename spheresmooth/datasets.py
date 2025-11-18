import pandas as pd
import importlib.resources as resources

def load_apw():
    path = resources.files("spheresmooth").joinpath("data/apw.csv")
    return pd.read_csv(path)
