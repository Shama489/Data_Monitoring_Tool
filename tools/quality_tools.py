from typing import Any

import pandas as pd

from profiler import calculate_data_quality_score, check_data_quality, generate_ai_quality_summary


def analyze_quality(
    frame: pd.DataFrame,
    options: dict[str, Any] | None = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    report = check_data_quality(frame, **(options or {}))
    quality_score = calculate_data_quality_score(frame)
    summary = generate_ai_quality_summary(frame, report, quality_score, use_llm=use_llm)
    return {"report": report, "quality_score": quality_score, "ai_summary": summary}