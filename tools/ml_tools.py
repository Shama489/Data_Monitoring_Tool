import pandas as pd

from profiler import forecast_data_health, forecast_metric


def forecast_data(
    frame: pd.DataFrame,
    date_column: str,
    value_column: str | None = None,
    periods: int = 4,
    frequency: str = "W",
    method: str = "auto",
    metric: str | None = None,
) -> dict:
    if not date_column:
        raise ValueError("date_column is required for forecast checks.")
    if metric == "data_health":
        return forecast_data_health(frame, date_column, periods, frequency)
    return forecast_metric(frame, date_column, value_column, periods, frequency, method)