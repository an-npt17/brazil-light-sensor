import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression


def analyse_data(df: pd.DataFrame) -> pd.DataFrame:
    df["created_at"] = pd.to_datetime(df["created_at"], format="%Y-%m-%d %H:%M:%S %Z")
    df["created_at"] = df["created_at"].dt.tz_convert("Asia/Ho_Chi_Minh")
    print(df.head())
    return df


def rank_based_on_light_intensity(series, is_daytime_func=None):
    """
    Rank light intensity with special handling for daytime periods.

    Args:
        series: The light sensor intensity values
        is_daytime_func: Function that takes index and returns True if it's daytime

    Returns:
        List of ranks from 1-8 where 1 is lowest light and 8 is highest
    """
    # Get slopes to detect changes
    slopes = linear_regression_last_6(series)
    values = series.values if hasattr(series, "values") else series

    ranks = []

    for i, (value, slope) in enumerate(zip(values, slopes)):
        is_daytime = is_daytime_func(i) if is_daytime_func else False

        # Base rank calculated from absolute value (1-8 scale)
        # Normalize the value to 1-8 range based on min/max of the series
        min_val, max_val = np.nanmin(values), np.nanmax(values)
        base_rank = (
            1 + 7 * (value - min_val) / (max_val - min_val) if max_val > min_val else 1
        )

        # Apply slope-based adjustments differently for day and night
        if is_daytime:
            # During daytime, make small changes more significant
            # Amplify the effect of slope by 3x during day
            slope_adjustment = slope * 3 if not np.isnan(slope) else 0
        else:
            # At night, use regular slope effect
            slope_adjustment = slope if not np.isnan(slope) else 0

        # Adjust rank by slope (positive slope increases rank, negative decreases)
        adjusted_rank = base_rank + slope_adjustment

        # Ensure rank stays within 1-8 range
        final_rank = max(1, min(8, round(adjusted_rank)))
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
    Convert the field1 to a range of 1 to 8, based on intensity and slope.
    """
    df["field1"] = df["field1"].astype(float)
    df["lin_reg"] = linear_regression_last_6(df["field1"])

    # Define a function to check if index is daytime
    is_daytime_func = lambda idx: is_daytime(idx, df)

    df["rank"] = rank_based_on_light_intensity(df["field1"], is_daytime_func)

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
    fig.update_layout(xaxis_title="Created At", yaxis_title="Value")
    fig.show()
