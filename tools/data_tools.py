from typing import Any

import pandas as pd

from data_sources import load_source


def load_dataset(payload: dict[str, Any]) -> pd.DataFrame:
    source = payload.get("source")
    if source is not None:
        if not isinstance(source, dict):
            raise ValueError("source must be an object")
        frame = load_source(source)
    else:
        dataset = payload.get("data")
        if dataset is None:
            dataset = payload.get("dataset")
        if dataset is None:
            dataset = payload.get("records")
        if dataset is None:
            raise ValueError("Dataset payload or source is required.")
        try:
            frame = pd.DataFrame(dataset)
        except (TypeError, ValueError) as error:
            raise ValueError("Dataset payload must be list-of-records or column-oriented JSON.") from error

    if frame.empty:
        raise ValueError("Dataset must not be empty.")
    return frame


def dataframe_from_value(dataset: Any, label: str = "Dataset") -> pd.DataFrame:
    try:
        frame = pd.DataFrame(dataset)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} payload must be list-of-records or column-oriented JSON.") from error
    if frame.empty:
        raise ValueError(f"{label} must not be empty.")
    return frame