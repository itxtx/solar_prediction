"""
Solar Prediction Test Suite

This package contains comprehensive tests for the solar prediction system including:
- Utility function tests (scaling, time parsing)
- TDMC model tests (training, likelihood, transitions)
- Deep learning model tests (LSTM, GRU)
- Memory stability tests (GPU memory tracking)
- Data pipeline tests (end-to-end preparation)
- Performance benchmarking tests

Run all tests with: pytest tests/
Run specific test file: pytest tests/test_utils.py
Run with output: pytest -s tests/
"""

__version__ = "1.0.0"
