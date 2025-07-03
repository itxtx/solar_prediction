#!/usr/bin/env python3
"""
Test script for the chunked data loader functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from solar_prediction.data_loader import (
    ChunkedDataLoader,
    ChunkedDataLoaderConfig,
    DataIteratorAdapter,
    create_chunked_loader,
    prepare_weather_data_chunked
)

def create_test_csv(file_path: str, num_rows: int = 1000):
    """Create a test CSV file with sample data."""
    np.random.seed(42)  # For reproducible test data
    
    # Generate sample data
    timestamps = pd.date_range('2023-01-01', periods=num_rows, freq='H')
    data = {
        'Timestamp': timestamps,
        'Radiation': np.random.uniform(0, 800, num_rows),
        'Temperature': np.random.uniform(-10, 40, num_rows),
        'Humidity': np.random.uniform(0, 100, num_rows),
        'WindSpeed': np.random.uniform(0, 20, num_rows),
        'Pressure': np.random.uniform(980, 1030, num_rows)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return df

def test_chunked_loader_config():
    """Test ChunkedDataLoaderConfig."""
    # Test default configuration
    config = ChunkedDataLoaderConfig()
    assert config.chunksize == 1_000_000
    assert config.apply_feature_engineering == True
    assert config.memory_efficient == True
    assert config.skip_empty_chunks == True
    
    # Test custom configuration
    custom_config = ChunkedDataLoaderConfig(
        chunksize=500_000,
        apply_feature_engineering=False,
        memory_efficient=False,
        skip_empty_chunks=False
    )
    assert custom_config.chunksize == 500_000
    assert custom_config.apply_feature_engineering == False
    assert custom_config.memory_efficient == False
    assert custom_config.skip_empty_chunks == False

def test_chunked_loader_basic():
    """Test basic chunked loading functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_file = os.path.join(temp_dir, "test_data.csv")
        original_df = create_test_csv(test_file, num_rows=250)
        
        # Configure small chunks for testing
        config = ChunkedDataLoaderConfig(
            chunksize=100,  # Small chunks for testing
            apply_feature_engineering=False
        )
        
        loader = ChunkedDataLoader(config)
        
        # Test loading chunks
        chunks = list(loader.load_chunks(test_file))
        
        # Should have 3 chunks (250 rows / 100 chunk size = 2.5 -> 3 chunks)
        assert len(chunks) >= 2
        assert len(chunks) <= 3
        
        # Combine chunks and compare with original
        combined_df = pd.concat(chunks, ignore_index=True)
        assert len(combined_df) == len(original_df)
        assert list(combined_df.columns) == list(original_df.columns)

def test_chunked_loader_with_feature_engineering():
    """Test chunked loading with feature engineering."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_file = os.path.join(temp_dir, "test_data.csv")
        create_test_csv(test_file, num_rows=200)
        
        # Configure with feature engineering
        config = ChunkedDataLoaderConfig(
            chunksize=100,
            apply_feature_engineering=True
        )
        
        loader = ChunkedDataLoader(config)
        
        # Test loading chunks with feature engineering
        chunks = list(loader.load_chunks(test_file))
        
        # Check that feature engineering was applied
        for chunk in chunks:
            if 'Timestamp' in chunk.columns:
                # Should have engineered time features
                assert 'HourOfDay' in chunk.columns
                assert 'Month' in chunk.columns
                assert 'HourSin' in chunk.columns
                assert 'HourCos' in chunk.columns
                
            if 'Temperature' in chunk.columns:
                # Should have temperature squared
                assert 'TempSquared' in chunk.columns
                
            if 'Humidity' in chunk.columns:
                # Should have humidity squared
                assert 'HumiditySquared' in chunk.columns

def test_load_and_combine_chunks():
    """Test loading and combining chunks into a single DataFrame."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_file = os.path.join(temp_dir, "test_data.csv")
        original_df = create_test_csv(test_file, num_rows=300)
        
        config = ChunkedDataLoaderConfig(chunksize=100, apply_feature_engineering=False)
        loader = ChunkedDataLoader(config)
        
        # Test combining all chunks
        combined_df = loader.load_and_combine_chunks(test_file)
        assert len(combined_df) == len(original_df)
        
        # Test limiting chunks
        limited_df = loader.load_and_combine_chunks(test_file, max_chunks=2)
        assert len(limited_df) == 200  # 2 chunks * 100 rows each

def test_get_chunk_info():
    """Test getting chunk information."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_file = os.path.join(temp_dir, "test_data.csv")
        create_test_csv(test_file, num_rows=500)
        
        config = ChunkedDataLoaderConfig(chunksize=100)
        loader = ChunkedDataLoader(config)
        
        # Test getting chunk info
        info = loader.get_chunk_info(test_file)
        
        assert 'file_path' in info
        assert 'file_size_mb' in info
        assert 'chunksize' in info
        assert 'estimated_total_rows' in info
        assert 'estimated_chunks' in info
        assert 'sample_columns' in info
        
        assert info['chunksize'] == 100
        assert info['estimated_total_rows'] == 500
        assert info['estimated_chunks'] >= 5

def test_data_iterator_adapter():
    """Test DataIteratorAdapter functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        test_file = os.path.join(temp_dir, "test_data.csv")
        original_df = create_test_csv(test_file, num_rows=200)
        
        config = ChunkedDataLoaderConfig(chunksize=50, apply_feature_engineering=False)
        loader = ChunkedDataLoader(config)
        
        # Test adapter
        adapter = DataIteratorAdapter(loader, test_file)
        
        # Test iteration
        chunks = list(adapter)
        assert len(chunks) == 4  # 200 rows / 50 chunk size = 4 chunks
        
        # Test conversion to DataFrame
        df = adapter.to_dataframe()
        assert len(df) == len(original_df)
        
        # Test with max_chunks
        df_limited = adapter.to_dataframe(max_chunks=2)
        assert len(df_limited) == 100  # 2 chunks * 50 rows each

def test_create_chunked_loader():
    """Test factory function for creating chunked loader."""
    # Test with default config
    loader1 = create_chunked_loader()
    assert loader1.config.chunksize == 1_000_000
    
    # Test with custom config
    custom_config = ChunkedDataLoaderConfig(chunksize=500_000)
    loader2 = create_chunked_loader(custom_config)
    assert loader2.config.chunksize == 500_000

def test_file_not_found():
    """Test error handling for non-existent files."""
    config = ChunkedDataLoaderConfig()
    loader = ChunkedDataLoader(config)
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        list(loader.load_chunks("nonexistent_file.csv"))
    
    with pytest.raises(FileNotFoundError):
        loader.get_chunk_info("nonexistent_file.csv")

def run_tests():
    """Run all tests manually (for systems without pytest)."""
    print("Running chunked data loader tests...")
    
    test_functions = [
        test_chunked_loader_config,
        test_chunked_loader_basic,
        test_chunked_loader_with_feature_engineering,
        test_load_and_combine_chunks,
        test_get_chunk_info,
        test_data_iterator_adapter,
        test_create_chunked_loader,
        test_file_not_found
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}...", end=" ")
            test_func()
            print("PASSED")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    # Try to use pytest if available, otherwise run manual tests
    try:
        import pytest
        pytest.main([__file__])
    except ImportError:
        print("pytest not available, running manual tests...")
        success = run_tests()
        exit(0 if success else 1)
