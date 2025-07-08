"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline  # type: ignore

from .pipelines.benchmark import create_pipeline as benchmark_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    benchmark = benchmark_pipeline()

    return {
        "__default__": benchmark,
        "benchmark": benchmark,
    }
