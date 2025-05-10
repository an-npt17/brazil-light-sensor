import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from sklearn.linear_model import LinearRegression


def analyse_data(df: pd.DataFrame) -> pd.DataFrame:
    df["created_at"] = pd.to_datetime(df["created_at"], format="%Y-%m-%d %H:%M:%S %Z")
    df["created_at"] = df["created_at"].dt.tz_convert("Asia/Ho_Chi_Minh")
    print(df.head())
    return df


def rank_with_moving_average(series, is_daytime_func=None, window_size=5):
    """
    Rank light intensity using moving averages to detect trends.

    Args:
        series: The light sensor intensity values
        is_daytime_func: Function that takes index and returns True if it's daytime
        window_size: Size of the moving average window

    Returns:
        List of continuous ranks from 1-8
    """
    values = series.values if hasattr(series, "values") else series

    # Calculate moving averages
    moving_avgs = []
    for i in range(len(values)):
        if i < window_size:
            window = values[: i + 1]
        else:
            window = values[i - window_size + 1 : i + 1]

        if len(window) > 0:
            moving_avgs.append(np.mean(window))
        else:
            moving_avgs.append(np.nan)

    # Calculate trends (current value compared to moving average)
    trends = []
    for i, (value, ma) in enumerate(zip(values, moving_avgs)):
        if np.isnan(ma) or ma == 0:
            trends.append(0)
        else:
            trends.append((value - ma) / ma)

    day_indices = [
        i for i in range(len(values)) if is_daytime_func and is_daytime_func(i)
    ]
    night_indices = [
        i for i in range(len(values)) if not (is_daytime_func and is_daytime_func(i))
    ]

    day_values = [values[i] for i in day_indices] if day_indices else []
    night_values = [values[i] for i in night_indices] if night_indices else []

    ranks = []

    for i, (value, trend) in enumerate(zip(values, trends)):
        is_daytime = is_daytime_func(i) if is_daytime_func else False

        # Use appropriate normalization range
        if is_daytime and day_values:
            min_val = np.percentile(day_values, 5)
            max_val = np.percentile(day_values, 95)
            # Scale trend effect for daytime (more sensitive)
            trend_factor = 2.0
            # Higher base for daytime
            base_min = 3
            base_range = 5
        else:
            min_val = np.min(night_values) if night_values else np.min(values)
            max_val = np.max(night_values) if night_values else np.max(values)
            trend_factor = 1.0
            base_min = 1
            base_range = 7

        # Base rank from value
        if max_val > min_val:
            base_rank = base_min + base_range * (value - min_val) / (max_val - min_val)
        else:
            base_rank = base_min

        # Adjust by trend (increasing trend = higher rank)
        trend_adjustment = trend * trend_factor
        adjusted_rank = base_rank + trend_adjustment

        # Ensure rank stays within 1-8 range
        final_rank = max(1, min(8, adjusted_rank))
        ranks.append(final_rank)

    return ranks


def linear_regression_last_6(series):
    result = []
    for i in range(len(series)):
        if i < 6:
            result.append(np.nan)  # Not enough points for regression
        else:
            x = np.arange(6).reshape(-1, 1)  # Independent variable (0 to 5)
            y = series[i - 6 : i].values  # Dependent variable (last 6 points)
            model = LinearRegression().fit(x, y)
            result.append(model.coef_[0])  # Append the slope of the line
    return result


def convert_to_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the field1 to a range of 1 to 8, based on intensity and trends.
    """
    df["field1"] = df["field1"].astype(float)

    is_daytime_func = lambda idx: is_daytime(idx, df)

    # Choose one of the ranking methods
    df["rank"] = rank_with_moving_average(df["field1"], is_daytime_func)

    return df


def is_daytime(index, df):
    """
    Determine if a given index represents daytime based on timestamp or threshold

    Example implementation - you might want to use actual time instead
    """
    # Option 1: Based on time of day if you have timestamps
    if "created_at" in df.columns:
        hour = df.iloc[index]["created_at"].hour
        return 6 <= hour <= 19

    # Option 2: Based on light intensity threshold
    return df.iloc[index]["field1"] > df["field1"].max() * 0.5


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    df = analyse_data(df)
    df = convert_to_range(df)

    fig = px.line(df, x="created_at", y="rank", title="Rank over Time")
    fig.update_layout(xaxis_title="Captured At", yaxis_title="Rank")
    plotly.offline.plot(fig, filename="rank_over_time.html", auto_open=True)
    fig.show()
