"""
Benchmark decorator for timing data pipeline operations.
Supports logging to file and optional CSV export for performance analysis.
"""

import time
import functools
import logging
import csv
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import threading
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class BenchmarkResult:
    """Container for benchmark timing results."""
    function_name: str
    stage_name: str
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    memory_usage: Optional[float] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

class BenchmarkTracker:
    """Thread-safe benchmark tracker for collecting timing results."""
    
    def __init__(self):
        self._results: List[BenchmarkResult] = []
        self._lock = threading.Lock()
        self._csv_file: Optional[str] = None
        self._logger = logging.getLogger(__name__)
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result to the tracker."""
        with self._lock:
            self._results.append(result)
            self._logger.info(f"[BENCHMARK] {result.stage_name}: {result.execution_time:.4f}s")
            
            # Write to CSV if configured
            if self._csv_file:
                self._write_to_csv(result)
    
    def set_csv_file(self, csv_file: str):
        """Set CSV file for logging results."""
        self._csv_file = csv_file
        # Create CSV with headers if it doesn't exist
        if not os.path.exists(csv_file):
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'function_name', 'stage_name', 
                    'execution_time', 'memory_usage', 'additional_info'
                ])
    
    def _write_to_csv(self, result: BenchmarkResult):
        """Write a single result to CSV file."""
        if not self._csv_file:
            return
        
        try:
            with open(self._csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    result.timestamp.isoformat(),
                    result.function_name,
                    result.stage_name,
                    result.execution_time,
                    result.memory_usage,
                    str(result.additional_info) if result.additional_info else ''
                ])
        except Exception as e:
            self._logger.warning(f"Failed to write benchmark result to CSV: {e}")
    
    def get_results(self) -> List[BenchmarkResult]:
        """Get all benchmark results."""
        with self._lock:
            return self._results.copy()
    
    def clear_results(self):
        """Clear all stored results."""
        with self._lock:
            self._results.clear()
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of benchmark results."""
        with self._lock:
            if not self._results:
                return {}
            
            summary = {}
            for result in self._results:
                stage = result.stage_name
                if stage not in summary:
                    summary[stage] = {
                        'total_time': 0.0,
                        'count': 0,
                        'min_time': float('inf'),
                        'max_time': 0.0
                    }
                
                summary[stage]['total_time'] += result.execution_time
                summary[stage]['count'] += 1
                summary[stage]['min_time'] = min(summary[stage]['min_time'], result.execution_time)
                summary[stage]['max_time'] = max(summary[stage]['max_time'], result.execution_time)
            
            # Calculate averages
            for stage_stats in summary.values():
                stage_stats['avg_time'] = stage_stats['total_time'] / stage_stats['count']
            
            return summary

# Global benchmark tracker instance
_benchmark_tracker = BenchmarkTracker()

def get_benchmark_tracker() -> BenchmarkTracker:
    """Get the global benchmark tracker instance."""
    return _benchmark_tracker

def configure_benchmark_csv(csv_file: str):
    """Configure CSV file for benchmark logging."""
    _benchmark_tracker.set_csv_file(csv_file)

def benchmark(stage_name: Optional[str] = None, track_memory: bool = False):
    """
    Decorator to benchmark function execution time.
    
    Args:
        stage_name: Name of the stage being benchmarked. If None, uses function name.
        track_memory: Whether to track memory usage (requires psutil).
    
    Usage:
        @benchmark(stage_name="data_loading")
        def load_data():
            # function implementation
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal stage_name
            if stage_name is None:
                stage_name = func.__name__
            
            # Track memory if requested
            memory_before = None
            memory_after = None
            if track_memory:
                try:
                    import psutil
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                except ImportError:
                    logging.getLogger(__name__).warning(
                        "psutil not available for memory tracking"
                    )
            
            # Execute function with timing
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                
                # Calculate memory usage
                memory_usage = None
                if track_memory and memory_before is not None:
                    try:
                        import psutil
                        process = psutil.Process()
                        memory_after = process.memory_info().rss / 1024 / 1024  # MB
                        memory_usage = memory_after - memory_before
                    except ImportError:
                        pass
                
                # Create and store benchmark result
                benchmark_result = BenchmarkResult(
                    function_name=func.__name__,
                    stage_name=stage_name,
                    execution_time=execution_time,
                    memory_usage=memory_usage
                )
                
                _benchmark_tracker.add_result(benchmark_result)
        
        return wrapper
    return decorator

@contextmanager
def benchmark_context(stage_name: str, track_memory: bool = False):
    """
    Context manager for benchmarking code blocks.
    
    Usage:
        with benchmark_context("data_processing"):
            # code to benchmark
            process_data()
    """
    memory_before = None
    if track_memory:
        try:
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            logging.getLogger(__name__).warning(
                "psutil not available for memory tracking"
            )
    
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Calculate memory usage
        memory_usage = None
        if track_memory and memory_before is not None:
            try:
                import psutil
                process = psutil.Process()
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage = memory_after - memory_before
            except ImportError:
                pass
        
        # Create and store benchmark result
        benchmark_result = BenchmarkResult(
            function_name="context_manager",
            stage_name=stage_name,
            execution_time=execution_time,
            memory_usage=memory_usage
        )
        
        _benchmark_tracker.add_result(benchmark_result)

def print_benchmark_summary():
    """Print a summary of all benchmark results."""
    summary = _benchmark_tracker.get_summary()
    if not summary:
        print("No benchmark results available.")
        return
    
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    total_time = 0.0
    for stage_name, stats in summary.items():
        print(f"\n{stage_name}:")
        print(f"  Total time: {stats['total_time']:.4f}s")
        print(f"  Average time: {stats['avg_time']:.4f}s")
        print(f"  Min time: {stats['min_time']:.4f}s")
        print(f"  Max time: {stats['max_time']:.4f}s")
        print(f"  Executions: {stats['count']}")
        total_time += stats['total_time']
    
    print(f"\nTotal pipeline time: {total_time:.4f}s")
    print("="*60)

def save_benchmark_results(filename: str):
    """Save benchmark results to a file."""
    results = _benchmark_tracker.get_results()
    if not results:
        print("No benchmark results to save.")
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'function_name', 'stage_name', 
            'execution_time', 'memory_usage', 'additional_info'
        ])
        
        for result in results:
            writer.writerow([
                result.timestamp.isoformat(),
                result.function_name,
                result.stage_name,
                result.execution_time,
                result.memory_usage,
                str(result.additional_info) if result.additional_info else ''
            ])
    
    print(f"Benchmark results saved to: {filename}")
