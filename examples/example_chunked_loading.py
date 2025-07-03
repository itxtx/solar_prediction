#!/usr/bin/env python3
"""
Example script demonstrating the use of chunked data loading.
Shows how to use the new ChunkedDataLoader and prepare_weather_data_chunked function.
"""

import logging
from pathlib import Path
from solar_prediction.data_loader import (
    ChunkedDataLoader, 
    ChunkedDataLoaderConfig, 
    create_chunked_loader,
    prepare_weather_data_chunked
)
from solar_prediction.config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_basic_chunked_loading():
    """Example of basic chunked loading."""
    logger.info("=== Example: Basic Chunked Loading ===")
    
    # Configure chunked loading
    config = ChunkedDataLoaderConfig(
        chunksize=500_000,  # 500k rows per chunk
        apply_feature_engineering=True,
        memory_efficient=True
    )
    
    # Create chunked loader
    loader = create_chunked_loader(config)
    
    # Example file path (adjust as needed)
    data_path = Path("data/solar_weather.csv")
    
    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}")
        logger.info("Using sample data path instead")
        data_path = Path("data/sample/SolarPrediction_sample.csv")
    
    if not data_path.exists():
        logger.error("No data file found. Please check your data paths.")
        return
    
    # Get information about the file
    try:
        chunk_info = loader.get_chunk_info(str(data_path))
        logger.info(f"File info: {chunk_info}")
    except Exception as e:
        logger.error(f"Error getting chunk info: {e}")
        return
    
    # Process chunks one by one
    chunk_count = 0
    total_rows = 0
    
    try:
        for chunk in loader.load_chunks(str(data_path)):
            chunk_count += 1
            total_rows += len(chunk)
            logger.info(f"Processed chunk {chunk_count}: {len(chunk)} rows, "
                       f"columns: {list(chunk.columns)[:5]}...")  # Show first 5 columns
            
            # Process only a few chunks for demonstration
            if chunk_count >= 3:
                logger.info("Stopping after 3 chunks for demonstration")
                break
                
        logger.info(f"Total processed: {chunk_count} chunks, {total_rows} rows")
        
    except Exception as e:
        logger.error(f"Error during chunked processing: {e}")

def example_unified_api_usage():
    """Example of using the unified API with existing prepare_weather_data."""
    logger.info("=== Example: Unified API Usage ===")
    
    # Get configuration objects
    config = get_config()
    
    # Configure chunked loading for large files
    chunked_config = ChunkedDataLoaderConfig(
        chunksize=1_000_000,  # 1M rows per chunk
        apply_feature_engineering=False,  # Let prepare_weather_data handle this
        memory_efficient=True
    )
    
    # Example file path
    data_path = Path("data/sample/SolarPrediction_sample.csv")
    
    if not data_path.exists():
        logger.error("Sample data file not found. Please check your data paths.")
        return
    
    try:
        # Use the unified API - it will automatically handle chunked loading
        logger.info("Processing data with unified API...")
        result = prepare_weather_data_chunked(
            data_source=str(data_path),  # File path
            input_cfg=config.data_input,
            transform_cfg=config.data_transformation,
            feature_cfg=config.features,
            scaling_cfg=config.scaling,
            sequence_cfg=config.sequence,
            chunked_config=chunked_config,
            max_chunks=2  # Limit to 2 chunks for demonstration
        )
        
        X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, transform_details = result
        
        logger.info("Successfully processed data with unified API:")
        logger.info(f"  X_train shape: {X_train.shape}")
        logger.info(f"  X_val shape: {X_val.shape}")
        logger.info(f"  X_test shape: {X_test.shape}")
        logger.info(f"  Feature columns: {len(feature_cols)}")
        logger.info(f"  Scalers: {list(scalers.keys())}")
        
    except Exception as e:
        logger.error(f"Error in unified API usage: {e}")

def example_iterator_usage():
    """Example of using chunked data as an iterator."""
    logger.info("=== Example: Iterator Usage ===")
    
    # Configure chunked loading
    config = ChunkedDataLoaderConfig(
        chunksize=100_000,  # Smaller chunks for demonstration
        apply_feature_engineering=True,
        memory_efficient=True
    )
    
    loader = create_chunked_loader(config)
    data_path = Path("data/sample/SolarPrediction_sample.csv")
    
    if not data_path.exists():
        logger.error("Sample data file not found.")
        return
    
    try:
        # Create an iterator
        chunk_iterator = loader.load_chunks(str(data_path))
        
        # Process chunks with custom logic
        processed_chunks = []
        for i, chunk in enumerate(chunk_iterator):
            # Apply custom processing to each chunk
            logger.info(f"Processing chunk {i+1}: {len(chunk)} rows")
            
            # Example: filter out night-time data (radiation < 10)
            if 'Radiation' in chunk.columns:
                day_data = chunk[chunk['Radiation'] >= 10]
                logger.info(f"  After filtering: {len(day_data)} rows (daylight only)")
                processed_chunks.append(day_data)
            else:
                processed_chunks.append(chunk)
            
            # Process only first few chunks for demonstration
            if i >= 1:
                break
        
        # Combine processed chunks
        if processed_chunks:
            combined_df = pd.concat(processed_chunks, ignore_index=True)
            logger.info(f"Combined processed data: {len(combined_df)} rows")
        
    except Exception as e:
        logger.error(f"Error in iterator usage: {e}")

if __name__ == "__main__":
    import pandas as pd
    
    logger.info("Starting chunked data loading examples")
    
    # Run examples
    example_basic_chunked_loading()
    print("\n" + "="*60 + "\n")
    
    example_unified_api_usage()
    print("\n" + "="*60 + "\n")
    
    example_iterator_usage()
    
    logger.info("Examples completed")
