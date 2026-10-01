import pandas as pd

from profiler import analyze_dataset_drift


def analyze_drift(baseline: pd.DataFrame, current: pd.DataFrame) -> dict:
    return analyze_dataset_drift(baseline, current)