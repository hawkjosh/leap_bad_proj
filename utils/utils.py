import pandas as pd
import yaml


def load_data(filepath: str) -> pd.DataFrame:
    data = pd.read_csv(filepath, parse_dates=["TimePeriod"], index_col="TimePeriod")
    data.index = data.index.tz_localize(None)
    return data


def load_config(filepath: str) -> dict:
    with open(filepath, "r") as file:
        return yaml.safe_load(file)
