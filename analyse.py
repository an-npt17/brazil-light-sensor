import pandas as pd
import plotly.express as px
import polars as pl
from matplotlib import pyplot as plt


def analyse_data(df: pd.DataFrame) -> None:
    df["created_at"] = pd.to_datetime(df["created_at"], format="%Y-%m-%d %H:%M:%S %Z")
    df["created_at"] = df["created_at"].dt.tz_convert("Asia/Ho_Chi_Minh")
    print(df.head())
    df.to_csv("data_processed.csv", index=False)
    plt.figure(figsize=(30, 10))
    plt.plot(
        df["created_at"],
        df["field1"],
    )
    plt.title("Value over Time")
    plt.xlabel("Created At")
    plt.ylabel("Value")
    plt.savefig("plot.png")
    fig = px.line(df, x="created_at", y="field1", title="Sensor over Time")
    fig.update_layout(xaxis_title="Created At", yaxis_title="Value")
    fig.show()


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    analyse_data(df)
