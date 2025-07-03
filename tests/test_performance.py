"""
Test performance: benchmark decorator returns ≤ predefined time budget.
"""

import pytest
import time
import numpy as np
import pandas as pd
from solar_prediction.benchmark import benchmark, benchmark_context, get_benchmark_tracker, configure_benchmark_csv
from solar_prediction.data_prep import prepare_weather_data
from solar_prediction.config import get_config


class TestBenchmarkDecorator:
    """Test the benchmark decorator functionality."""
    
    def test_benchmark_decorator_basic(self, time_budget):
        """Test that benchmark decorator measures execution time correctly."""
        
        @benchmark(stage_name="test_function")
        def test_function():
            time.sleep(0.1)  # Sleep for 100ms
            return "test_result"
        
        # Clear previous results
        tracker = get_benchmark_tracker()
        tracker.clear_results()
        
        # Run the decorated function
        result = test_function()
        
        # Check that function returned correctly
        assert result == "test_result"
        
        # Check that benchmark was recorded
        results = tracker.get_results()
        assert len(results) == 1
        
        benchmark_result = results[0]
        assert benchmark_result.stage_name == "test_function"
        assert benchmark_result.execution_time >= 0.09  # At least 90ms (accounting for precision)
        assert benchmark_result.execution_time <= 0.2   # Less than 200ms (with tolerance)
        
        print(f"Benchmark decorator test passed: {benchmark_result.execution_time:.3f}s")
    
    def test_benchmark_decorator_with_time_budget(self, time_budget):
        """Test that benchmark decorator respects time budgets."""
        
        @benchmark(stage_name="fast_function")
        def fast_function():
            time.sleep(0.01)  # 10ms - should be well within budget
            return True
        
        @benchmark(stage_name="data_prep_simulation")
        def data_prep_simulation():
            # Simulate data preparation work
            time.sleep(0.05)  # 50ms - should be within data_preparation budget
            return True
        
        # Clear previous results
        tracker = get_benchmark_tracker()
        tracker.clear_results()
        
        # Run functions
        result1 = fast_function()
        result2 = data_prep_simulation()
        
        assert result1 is True
        assert result2 is True
        
        # Check results against time budget
        results = tracker.get_results()
        assert len(results) == 2
        
        for result in results:
            if result.stage_name == "fast_function":
                # Should be very fast
                assert result.execution_time <= 0.1
            elif result.stage_name == "data_prep_simulation":
                # Should be within data preparation budget
                assert result.execution_time <= time_budget['data_preparation']
        
        print("Time budget test passed!")
    
    def test_benchmark_context_manager(self, time_budget):
        """Test the benchmark context manager."""
        tracker = get_benchmark_tracker()
        tracker.clear_results()
        
        with benchmark_context("context_test"):
            time.sleep(0.02)  # 20ms
            dummy_work = sum(range(1000))  # Some actual work
        
        results = tracker.get_results()
        assert len(results) == 1
        
        result = results[0]
        assert result.stage_name == "context_test"
        assert result.execution_time >= 0.015  # At least 15ms
        assert result.execution_time <= 0.1    # Less than 100ms
        
        print(f"Context manager test passed: {result.execution_time:.3f}s")
    
    def test_benchmark_multiple_calls(self):
        """Test benchmark decorator with multiple calls to same function."""
        
        @benchmark(stage_name="repeated_function")
        def repeated_function(delay):
            time.sleep(delay)
            return delay
        
        tracker = get_benchmark_tracker()
        tracker.clear_results()
        
        # Call the function multiple times with different delays
        delays = [0.01, 0.02, 0.015]
        results_values = []
        
        for delay in delays:
            result = repeated_function(delay)
            results_values.append(result)
        
        # Check that all calls were recorded
        benchmark_results = tracker.get_results()
        assert len(benchmark_results) == 3
        
        # Check that all results have the same stage name
        for br in benchmark_results:
            assert br.stage_name == "repeated_function"
        
        # Check that execution times are roughly in line with delays
        for i, br in enumerate(benchmark_results):
            expected_delay = delays[i]
            assert br.execution_time >= expected_delay * 0.8  # Allow 20% tolerance
            assert br.execution_time <= expected_delay * 3.0  # Allow overhead
        
        print("Multiple calls test passed!")


class TestPerformanceBudgets:
    """Test that various operations stay within performance budgets."""
    
    def test_data_preparation_performance(self, small_sample_data, time_budget):
        """Test that data preparation stays within time budget."""
        config = get_config()
        
        # Clear benchmark tracker
        tracker = get_benchmark_tracker()
        tracker.clear_results()
        
        @benchmark(stage_name="data_preparation")
        def run_data_prep():
            return prepare_weather_data(
                small_sample_data.copy(),
                config.data,
                config.transformation,
                config.features,
                config.scaling,
                config.sequences
            )
        
        # Run data preparation
        start_time = time.time()
        result = run_data_prep()
        elapsed_time = time.time() - start_time
        
        # Check against time budget
        assert elapsed_time <= time_budget['data_preparation'], \
            f"Data preparation took {elapsed_time:.3f}s > budget {time_budget['data_preparation']}s"
        
        # Check benchmark was recorded
        results = tracker.get_results()
        assert len(results) >= 1  # Should have at least the main benchmark
        
        main_result = next(r for r in results if r.stage_name == "data_preparation")
        assert main_result.execution_time <= time_budget['data_preparation']
        
        print(f"Data preparation performance test passed: {elapsed_time:.3f}s")
    
    def test_inference_performance_simulation(self, time_budget):
        """Test simulated inference performance."""
        
        @benchmark(stage_name="inference_simulation")
        def simulate_inference():
            # Simulate model inference with some computation
            X = np.random.randn(10, 20, 15)  # Small batch
            # Simulate neural network forward pass
            for _ in range(3):  # 3 layers
                X = np.tanh(X @ np.random.randn(X.shape[-1], X.shape[-1]))
            return X.mean()
        
        tracker = get_benchmark_tracker()
        tracker.clear_results()
        
        start_time = time.time()
        result = simulate_inference()
        elapsed_time = time.time() - start_time
        
        # Check against inference budget
        assert elapsed_time <= time_budget['inference'], \
            f"Inference simulation took {elapsed_time:.3f}s > budget {time_budget['inference']}s"
        
        # Check benchmark was recorded
        results = tracker.get_results()
        assert len(results) == 1
        assert results[0].execution_time <= time_budget['inference']
        
        print(f"Inference performance test passed: {elapsed_time:.3f}s")
    
    def test_benchmark_summary_performance(self):
        """Test that benchmark summary generation is fast."""
        
        # Generate some benchmark results
        tracker = get_benchmark_tracker()
        tracker.clear_results()
        
        # Add several benchmark results
        for i in range(10):
            with benchmark_context(f"operation_{i}"):
                time.sleep(0.001)  # 1ms each
        
        # Test summary generation performance
        start_time = time.time()
        summary = tracker.get_summary()
        elapsed_time = time.time() - start_time
        
        # Summary generation should be very fast
        assert elapsed_time <= 0.1, f"Summary generation took {elapsed_time:.3f}s > 0.1s"
        
        # Check that summary contains expected data
        assert len(summary) == 10  # 10 different operations
        for stage_name, stats in summary.items():
            assert 'total_time' in stats
            assert 'avg_time' in stats
            assert 'count' in stats
            assert stats['count'] == 1  # Each operation ran once
        
        print("Benchmark summary performance test passed!")


class TestBenchmarkCSVExport:
    """Test CSV export functionality for benchmarks."""
    
    def test_csv_export_functionality(self, tmp_path):
        """Test that benchmark results can be exported to CSV."""
        csv_file = tmp_path / "benchmark_results.csv"
        
        # Configure CSV export
        configure_benchmark_csv(str(csv_file))
        
        # Clear previous results
        tracker = get_benchmark_tracker()
        tracker.clear_results()
        
        # Run some benchmarked operations
        @benchmark(stage_name="csv_test_operation")
        def csv_test_operation():
            time.sleep(0.01)
            return "done"
        
        # Run operations
        for i in range(3):
            csv_test_operation()
        
        # Check that CSV file was created and contains data
        assert csv_file.exists()
        
        # Read CSV and verify contents
        df = pd.read_csv(csv_file)
        assert len(df) == 3  # 3 operations
        assert 'stage_name' in df.columns
        assert 'execution_time' in df.columns
        assert all(df['stage_name'] == 'csv_test_operation')
        assert all(df['execution_time'] > 0)
        
        print(f"CSV export test passed: {len(df)} records exported")
    
    def test_benchmark_memory_tracking(self):
        """Test benchmark with memory tracking (if psutil is available)."""
        tracker = get_benchmark_tracker()
        tracker.clear_results()
        
        @benchmark(stage_name="memory_test", track_memory=True)
        def memory_intensive_operation():
            # Allocate some memory
            data = np.random.randn(1000, 1000)
            result = np.sum(data)
            del data
            return result
        
        result = memory_intensive_operation()
        assert isinstance(result, (int, float))
        
        # Check benchmark result
        results = tracker.get_results()
        assert len(results) == 1
        
        benchmark_result = results[0]
        assert benchmark_result.stage_name == "memory_test"
        
        # Memory usage might be None if psutil is not available
        if benchmark_result.memory_usage is not None:
            assert isinstance(benchmark_result.memory_usage, (int, float))
            print(f"Memory tracking test passed: {benchmark_result.memory_usage:.1f}MB change")
        else:
            print("Memory tracking test passed (psutil not available)")


class TestPerformanceEdgeCases:
    """Test edge cases for performance measurement."""
    
    def test_benchmark_very_fast_operation(self):
        """Test benchmarking very fast operations."""
        
        @benchmark(stage_name="very_fast")
        def very_fast_operation():
            return 1 + 1
        
        tracker = get_benchmark_tracker()
        tracker.clear_results()
        
        result = very_fast_operation()
        assert result == 2
        
        results = tracker.get_results()
        assert len(results) == 1
        
        # Even very fast operations should be measurable
        assert results[0].execution_time >= 0
        assert results[0].execution_time <= 0.1  # Should be very fast
        
        print(f"Very fast operation test passed: {results[0].execution_time:.6f}s")
    
    def test_benchmark_with_exception(self):
        """Test that benchmark works even when function raises exception."""
        
        @benchmark(stage_name="exception_test")
        def function_with_exception():
            time.sleep(0.01)
            raise ValueError("Test exception")
        
        tracker = get_benchmark_tracker()
        tracker.clear_results()
        
        # Function should raise exception but benchmark should still record
        with pytest.raises(ValueError):
            function_with_exception()
        
        # Check that benchmark was still recorded
        results = tracker.get_results()
        assert len(results) == 1
        assert results[0].stage_name == "exception_test"
        assert results[0].execution_time >= 0.005  # At least 5ms
        
        print("Exception handling test passed!")
    
    def test_nested_benchmarks(self):
        """Test nested benchmark decorators."""
        
        @benchmark(stage_name="outer_function")
        def outer_function():
            time.sleep(0.01)
            return inner_function()
        
        @benchmark(stage_name="inner_function")
        def inner_function():
            time.sleep(0.01)
            return "inner_result"
        
        tracker = get_benchmark_tracker()
        tracker.clear_results()
        
        result = outer_function()
        assert result == "inner_result"
        
        # Should have recorded both functions
        results = tracker.get_results()
        assert len(results) == 2
        
        stage_names = {r.stage_name for r in results}
        assert "outer_function" in stage_names
        assert "inner_function" in stage_names
        
        # Outer function should take longer than inner function
        outer_time = next(r.execution_time for r in results if r.stage_name == "outer_function")
        inner_time = next(r.execution_time for r in results if r.stage_name == "inner_function")
        
        assert outer_time >= inner_time
        
        print("Nested benchmarks test passed!")
    
    def test_concurrent_benchmarks(self):
        """Test that benchmark tracker handles concurrent operations safely."""
        import threading
        
        @benchmark(stage_name="concurrent_operation")
        def concurrent_operation(thread_id):
            time.sleep(0.01)
            return f"thread_{thread_id}"
        
        tracker = get_benchmark_tracker()
        tracker.clear_results()
        
        # Run multiple threads
        threads = []
        results = []
        
        def thread_worker(tid):
            result = concurrent_operation(tid)
            results.append(result)
        
        # Start threads
        for i in range(3):
            thread = threading.Thread(target=thread_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check that all operations completed
        assert len(results) == 3
        assert all("thread_" in r for r in results)
        
        # Check that all benchmarks were recorded
        benchmark_results = tracker.get_results()
        assert len(benchmark_results) == 3
        assert all(br.stage_name == "concurrent_operation" for br in benchmark_results)
        
        print("Concurrent benchmarks test passed!")
