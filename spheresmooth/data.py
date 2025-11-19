import pandas as pd
import importlib.resources as resources

def load_apw():
    path = resources.files("spheresmooth").joinpath("data/apw.csv")
    return pd.read_csv(path)

def load_goni():
    path = resources.files("spheresmooth").joinpath("data/goni.csv")
    return pd.read_csv(path)