import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").asfreq("h")
    return df