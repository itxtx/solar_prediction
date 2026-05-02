"""Solar Prediction Package - Optimized Data Pipeline."""

import logging

# Jupyter/Optuna worker threads can outlive the kernel stream they inherited.
# Suppress logging's own handler tracebacks in that case; model exceptions still
# propagate normally.
logging.raiseExceptions = False

# Import key components for easy access
from .benchmark import (
    benchmark,
    benchmark_context,
    configure_benchmark_csv,
    print_benchmark_summary,
)
from .config import get_config
from .data_loader import load_solar_dataset
from .data_prep import prepare_weather_data

__version__ = "1.0.0"
__all__ = [
    "benchmark",
    "benchmark_context",
    "configure_benchmark_csv",
    "print_benchmark_summary",
    "get_config",
    "load_solar_dataset",
    "prepare_weather_data",
]
