"""
Test memory: run forward pass 100×, check GPU memory stable (pytest -s).
"""

import pytest
import numpy as np
import torch
import time
from solar_prediction.memory_tracker import MemoryTracker
from solar_prediction.lstm import WeatherLSTM, create_model_hyperparameters_from_config
from solar_prediction.gru import WeatherGRU, create_gru_model_hyperparameters_from_config


@pytest.fixture
def device():
    """Get device for testing (prefer CUDA if available, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def memory_tracker(device):
    """Create memory tracker for the test device."""
    return MemoryTracker(device=device, verbose=True)


@pytest.fixture
def test_model_and_data(device):
    """Create a small model and test data for memory testing."""
    # Create small test data
    batch_size = 8
    seq_length = 10
    input_dim = 6
    
    # Use fixed seed for reproducible test data
    torch.manual_seed(42)
    X = torch.randn(batch_size, seq_length, input_dim, dtype=torch.float32, device=device)
    
    # Create small LSTM model
    model_params = create_model_hyperparameters_from_config(
        input_dim=input_dim,
        config_override={'hidden_dim': 16, 'num_layers': 1, 'dropout_prob': 0.0}
    )
    
    model = WeatherLSTM(model_params)
    model.to(device)
    model.eval()  # Set to eval mode to disable dropout
    
    return model, X


class TestMemoryStability:
    """Test memory stability during repeated operations."""
    
    def test_forward_pass_memory_stability_100x(self, test_model_and_data, memory_tracker, time_budget):
        """Run forward pass 100× and check that GPU memory stays stable."""
        model, X = test_model_and_data
        device = X.device
        
        print(f"\nTesting memory stability on device: {device}")
        
        # Initial memory snapshot
        memory_tracker.reset_peak_memory()
        memory_tracker.snapshot("initial", "before any operations")
        initial_memory = memory_tracker.get_memory_info()
        
        # Warm up (some operations allocate memory on first run)
        with torch.no_grad():
            for _ in range(5):
                _ = model(X)
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Memory after warmup
        memory_tracker.snapshot("after_warmup", "after warmup runs")
        warmup_memory = memory_tracker.get_memory_info()
        
        # Store memory usage during 100 iterations
        memory_readings = []
        start_time = time.time()
        
        print(f"Running 100 forward passes...")
        
        with torch.no_grad():
            for i in range(100):
                output = model(X)
                
                # Take memory reading every 10 iterations
                if i % 10 == 0:
                    memory_info = memory_tracker.get_memory_info()
                    memory_readings.append(memory_info['allocated'])
                    if i % 20 == 0:
                        print(f"Iteration {i}: {memory_info['allocated']:.1f}MB allocated")
                
                # Explicitly delete output to free memory
                del output
        
        # Final memory check
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        memory_tracker.snapshot("after_100_iterations", "after 100 forward passes")
        final_memory = memory_tracker.get_memory_info()
        
        # Check timing is within budget
        elapsed_time = time.time() - start_time
        assert elapsed_time <= time_budget['memory_test'], \
            f"Memory test took too long: {elapsed_time:.2f}s > {time_budget['memory_test']}s"
        
        print(f"Completed 100 iterations in {elapsed_time:.2f}s")
        
        # Analyze memory stability
        if len(memory_readings) > 1:
            memory_variance = np.var(memory_readings)
            memory_mean = np.mean(memory_readings)
            memory_std = np.std(memory_readings)
            
            print(f"Memory statistics:")
            print(f"  Mean allocated: {memory_mean:.1f}MB")
            print(f"  Std deviation: {memory_std:.1f}MB")
            print(f"  Variance: {memory_variance:.1f}MB²")
            print(f"  Min/Max: {min(memory_readings):.1f}/{max(memory_readings):.1f}MB")
            
            # Memory should be stable (low variance relative to mean)
            if memory_mean > 0:
                relative_std = memory_std / memory_mean
                assert relative_std < 0.1, \
                    f"Memory usage too variable: {relative_std:.3f} (std/mean) > 0.1"
            
            # Memory should not grow significantly
            memory_growth = max(memory_readings) - min(memory_readings)
            max_growth_mb = 50  # Allow up to 50MB growth
            assert memory_growth <= max_growth_mb, \
                f"Memory grew too much: {memory_growth:.1f}MB > {max_growth_mb}MB"
        
        # Final memory should not be significantly higher than after warmup
        if device.type == 'cuda':
            memory_increase = final_memory['allocated'] - warmup_memory['allocated']
            max_increase_mb = 20  # Allow up to 20MB increase
            assert memory_increase <= max_increase_mb, \
                f"Final memory increased too much: {memory_increase:.1f}MB > {max_increase_mb}MB"
        
        print(f"Memory test passed! Memory usage remained stable.")

    def test_gru_memory_stability(self, memory_tracker, device, time_budget):
        """Test memory stability with GRU model."""
        batch_size = 8
        seq_length = 10
        input_dim = 6
        
        # Create test data
        torch.manual_seed(42)
        X = torch.randn(batch_size, seq_length, input_dim, dtype=torch.float32, device=device)
        
        # Create GRU model
        model_params = create_gru_model_hyperparameters_from_config(
            input_dim=input_dim,
            config_override={'hidden_dim': 16, 'num_layers': 1, 'bidirectional': False, 'dropout_prob': 0.0}
        )
        
        model = WeatherGRU(model_params)
        model.to(device)
        model.eval()
        
        print(f"\nTesting GRU memory stability on device: {device}")
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(X)
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        memory_tracker.snapshot("gru_warmup", "after GRU warmup")
        
        # Run 50 iterations (fewer than LSTM test for speed)
        memory_readings = []
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(50):
                output = model(X)
                
                if i % 5 == 0:
                    memory_info = memory_tracker.get_memory_info()
                    memory_readings.append(memory_info['allocated'])
                
                del output
        
        elapsed_time = time.time() - start_time
        assert elapsed_time <= time_budget['memory_test'] / 2, \
            f"GRU memory test took too long: {elapsed_time:.2f}s"
        
        # Check memory stability
        if len(memory_readings) > 1:
            memory_variance = np.var(memory_readings)
            memory_mean = np.mean(memory_readings)
            
            if memory_mean > 0:
                relative_std = np.std(memory_readings) / memory_mean
                assert relative_std < 0.15, \
                    f"GRU memory too variable: {relative_std:.3f} > 0.15"
        
        print(f"GRU memory test passed in {elapsed_time:.2f}s!")

    def test_memory_leak_detection(self, memory_tracker, device):
        """Test for memory leaks during model operations."""
        if device.type == 'cpu':
            pytest.skip("Memory leak detection test is more relevant for GPU")
        
        print(f"\nTesting memory leak detection on device: {device}")
        
        # Create larger model and data for more sensitive leak detection
        batch_size = 16
        seq_length = 20
        input_dim = 10
        
        model_params = create_model_hyperparameters_from_config(
            input_dim=input_dim,
            config_override={'hidden_dim': 32, 'num_layers': 2}
        )
        
        model = WeatherLSTM(model_params)
        model.to(device)
        model.eval()
        
        # Initial memory state
        torch.cuda.empty_cache()
        memory_tracker.reset_peak_memory()
        memory_tracker.snapshot("leak_test_start", "start of leak detection test")
        initial_memory = memory_tracker.get_memory_info()['allocated']
        
        # Run operations that could potentially leak memory
        for iteration in range(10):
            # Create new data each iteration (potential leak source)
            X = torch.randn(batch_size, seq_length, input_dim, device=device)
            
            with torch.no_grad():
                # Multiple forward passes
                for _ in range(5):
                    output = model(X)
                    del output  # Explicit deletion
            
            # Clean up
            del X
            
            # Check memory every few iterations
            if iteration % 3 == 0:
                torch.cuda.empty_cache()
                current_memory = memory_tracker.get_memory_info()['allocated']
                memory_increase = current_memory - initial_memory
                
                print(f"Iteration {iteration}: Memory increase = {memory_increase:.1f}MB")
                
                # Memory should not grow significantly
                max_acceptable_increase = 30  # 30MB threshold
                assert memory_increase <= max_acceptable_increase, \
                    f"Potential memory leak detected: {memory_increase:.1f}MB > {max_acceptable_increase}MB"
        
        # Final check
        torch.cuda.empty_cache()
        final_memory = memory_tracker.get_memory_info()['allocated']
        total_increase = final_memory - initial_memory
        
        print(f"Total memory increase: {total_increase:.1f}MB")
        
        # Total increase should be minimal
        assert total_increase <= 50, \
            f"Memory leak detected: {total_increase:.1f}MB increase > 50MB"
        
        print("Memory leak test passed!")


class TestMemoryTrackerFunctionality:
    """Test the memory tracker utility itself."""
    
    def test_memory_tracker_basic(self, memory_tracker):
        """Test basic memory tracker functionality."""
        # Take initial snapshot
        memory_tracker.snapshot("test_start", "beginning of tracker test")
        
        # Get memory info
        memory_info = memory_tracker.get_memory_info()
        
        # Check that memory info has expected keys
        assert 'allocated' in memory_info
        assert 'cached' in memory_info
        assert 'max_allocated' in memory_info
        
        # Check that values are non-negative
        assert memory_info['allocated'] >= 0
        assert memory_info['cached'] >= 0
        assert memory_info['max_allocated'] >= 0
        
    def test_memory_tracker_snapshots(self, memory_tracker):
        """Test memory tracker snapshot functionality."""
        # Take multiple snapshots
        memory_tracker.snapshot("snapshot1", "first snapshot")
        
        # Do some memory allocation
        dummy_tensor = torch.randn(1000, 1000)
        
        memory_tracker.snapshot("snapshot2", "after tensor allocation")
        
        # Check that snapshots exist
        assert "snapshot1" in memory_tracker.snapshots
        assert "snapshot2" in memory_tracker.snapshots
        
        # Clean up
        del dummy_tensor
        
    def test_memory_cleanup(self, memory_tracker, device):
        """Test memory cleanup functionality."""
        if device.type == 'cpu':
            pytest.skip("Memory cleanup test is for GPU devices")
        
        # Allocate some memory
        dummy_tensor = torch.randn(1000, 1000, device=device)
        
        memory_before_cleanup = memory_tracker.get_memory_info()['allocated']
        
        # Delete tensor and cleanup
        del dummy_tensor
        memory_tracker.cleanup_memory(force=True)
        
        memory_after_cleanup = memory_tracker.get_memory_info()['allocated']
        
        # Memory should be reduced (or at least not increased)
        assert memory_after_cleanup <= memory_before_cleanup + 1  # Allow 1MB tolerance
        
    def test_memory_context_manager(self, memory_tracker):
        """Test memory tracker context manager."""
        with memory_tracker.track_memory("context_test", cleanup_after=True):
            # Do some operations
            dummy_tensor = torch.randn(500, 500)
            dummy_tensor2 = dummy_tensor * 2
            del dummy_tensor
            del dummy_tensor2
        
        # Context manager should have logged the operation
        # This is mainly a smoke test to ensure no exceptions


class TestMemoryEdgeCases:
    """Test edge cases for memory management."""
    
    def test_large_batch_memory(self, device, memory_budget):
        """Test memory usage with larger batch sizes."""
        if device.type == 'cpu':
            pytest.skip("Large batch memory test is for GPU devices")
        
        print(f"\nTesting large batch memory usage on device: {device}")
        
        # Create model
        model_params = create_model_hyperparameters_from_config(
            input_dim=20,
            config_override={'hidden_dim': 64, 'num_layers': 2}
        )
        model = WeatherLSTM(model_params)
        model.to(device)
        model.eval()
        
        # Test progressively larger batch sizes
        batch_sizes = [32, 64, 128]
        seq_length = 50
        input_dim = 20
        
        for batch_size in batch_sizes:
            try:
                X = torch.randn(batch_size, seq_length, input_dim, device=device)
                
                with torch.no_grad():
                    output = model(X)
                
                memory_info = torch.cuda.memory_allocated() / 1024**2  # MB
                print(f"Batch size {batch_size}: {memory_info:.1f}MB allocated")
                
                # Check against memory budget
                assert memory_info <= memory_budget, \
                    f"Memory usage {memory_info:.1f}MB exceeds budget {memory_budget}MB"
                
                del X, output
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM at batch size {batch_size} (expected for large batches)")
                    break
                else:
                    raise
        
        print("Large batch memory test completed!")
    
    def test_gradient_memory_management(self, device):
        """Test memory management when gradients are involved."""
        # Create model and data
        model_params = create_model_hyperparameters_from_config(
            input_dim=5,
            config_override={'hidden_dim': 16, 'num_layers': 1}
        )
        model = WeatherLSTM(model_params)
        model.to(device)
        model.train()  # Enable gradients
        
        X = torch.randn(8, 10, 5, device=device, requires_grad=True)
        y = torch.randn(8, 1, device=device)
        criterion = torch.nn.MSELoss()
        
        memory_tracker = MemoryTracker(device=device)
        memory_tracker.snapshot("before_forward", "before forward pass with gradients")
        
        # Forward pass with gradients
        output = model(X)
        loss = criterion(output, y)
        
        memory_tracker.snapshot("after_forward", "after forward pass with gradients")
        
        # Backward pass
        loss.backward()
        
        memory_tracker.snapshot("after_backward", "after backward pass")
        
        # Clear gradients
        model.zero_grad()
        
        memory_tracker.snapshot("after_zero_grad", "after clearing gradients")
        
        # Memory should be managed properly throughout
        snapshots = list(memory_tracker.snapshots.values())
        for snapshot in snapshots:
            assert snapshot.allocated >= 0
            print(f"{snapshot.description}: {snapshot.allocated:.1f}MB")
        
        print("Gradient memory management test passed!")
