import pandas as pd

from profiler import train_and_explain_model


def explain_model(
    frame: pd.DataFrame,
    target_column: str,
    task: str = "classification",
    test_size: float = 0.2,
) -> dict:
    if not target_column:
        raise ValueError("target_column is required for explain checks.")
    return train_and_explain_model(frame, target_column, task, test_size)