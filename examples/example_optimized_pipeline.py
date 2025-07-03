#!/usr/bin/env python3
"""
Example script demonstrating the optimized data pipeline with benchmarking.

This script shows how to use the new optimizations:
1. In-place operations with chained assignment protection
2. Vectorized time parsing and feature engineering
3. Pre-allocated arrays with stride-tricks for sequence generation
4. Benchmark decorator timing with CSV logging
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Import the optimized modules
from solar_prediction.data_loader import load_solar_dataset
from solar_prediction.data_prep import prepare_weather_data
from solar_prediction.config import get_config, DataInputConfig
from solar_prediction.benchmark import (
    configure_benchmark_csv, 
    print_benchmark_summary, 
    save_benchmark_results,
    get_benchmark_tracker,
    benchmark
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@benchmark(stage_name="data_loading", track_memory=True)
def load_data():
    """Load the solar dataset with benchmarking."""
    try:
        # Try to load the full dataset, fallback to sample
        df = load_solar_dataset(prefer_sample=False)
        logger.info(f"Loaded dataset with {len(df)} records and {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.warning(f"Failed to load full dataset: {e}")
        # Fallback to sample data
        df = load_solar_dataset(prefer_sample=True)
        logger.info(f"Loaded sample dataset with {len(df)} records and {len(df.columns)} columns")
        return df

@benchmark(stage_name="configuration_setup")
def setup_configuration():
    """Setup the configuration with optimized settings."""
    config = get_config()
    
    # Configure for the specific dataset
    # Adjust based on your actual column names
    data_input_config = DataInputConfig(
        target_col_original_name="GHI",  # Adjust this to your target column
        common_ghi_col="GHI",
        common_temp_col="temp",
        common_pressure_col="pressure", 
        common_humidity_col="humidity",
        common_wind_speed_col="wind_speed",
        time_col="Time",
        unix_time_col="UNIXTime",
        sunrise_col="TimeSunRise",
        sunset_col="TimeSunSet",
        hour_col_raw="hour",
        month_col_raw="month",
        daylength_col_raw="dayLength",
        is_sun_col_raw="isSun",
        sunlight_time_daylength_ratio_raw="SunlightTime/daylength"
    )
    
    return config, data_input_config

def main():
    """Main function demonstrating the optimized pipeline."""
    print("Starting Optimized Solar Data Pipeline Demo")
    print("=" * 50)
    
    # Configure benchmark CSV logging
    benchmark_csv_file = "logs/benchmark_results.csv"
    Path("logs").mkdir(exist_ok=True)
    configure_benchmark_csv(benchmark_csv_file)
    
    try:
        # Load data with benchmarking
        df = load_data()
        
        # Setup configuration
        config, data_input_config = setup_configuration()
        
        # Check if target column exists in the dataset
        if data_input_config.target_col_original_name not in df.columns:
            # Try to find a suitable target column
            possible_targets = ["GHI", "ghi", "Radiation", "radiation", "target"]
            target_found = None
            for target in possible_targets:
                if target in df.columns:
                    target_found = target
                    break
            
            if target_found:
                logger.info(f"Target column '{data_input_config.target_col_original_name}' not found. Using '{target_found}' instead.")
                data_input_config.target_col_original_name = target_found
                data_input_config.common_ghi_col = target_found
            else:
                logger.error(f"No suitable target column found in dataset. Available columns: {list(df.columns)}")
                return
        
        print(f"\nDataset info:")
        print(f"- Shape: {df.shape}")
        print(f"- Target column: {data_input_config.target_col_original_name}")
        print(f"- Columns: {list(df.columns)}")
        
        # Run the optimized preparation pipeline
        print("\nRunning optimized data preparation pipeline...")
        
        try:
            with pd.option_context('mode.chained_assignment', None):
                # Demonstrate the chained assignment optimization context
                (X_train, X_val, X_test, 
                 y_train, y_val, y_test, 
                 scalers, feature_cols, transform_details) = prepare_weather_data(
                    df_input=df,
                    input_cfg=data_input_config,
                    transform_cfg=config.transformation,
                    feature_cfg=config.features,
                    scaling_cfg=config.scaling,
                    sequence_cfg=config.sequences
                )
            
            print(f"\nPipeline completed successfully!")
            print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
            print(f"Validation data shape: X_val={X_val.shape}, y_val={y_val.shape}")
            print(f"Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")
            print(f"Number of features: {len(feature_cols)}")
            print(f"Features used: {feature_cols}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Print benchmark summary
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK RESULTS")
    print_benchmark_summary()
    
    # Save detailed benchmark results
    save_benchmark_results("logs/detailed_benchmark_results.csv")
    
    # Show optimization insights
    print(f"\nOptimization features demonstrated:")
    print(f"1. ✓ In-place operations with chained assignment protection")
    print(f"2. ✓ Vectorized time parsing and feature engineering")
    print(f"3. ✓ Pre-allocated arrays with stride-tricks for sequence generation")
    print(f"4. ✓ Benchmark decorator timing with CSV logging")
    print(f"\nBenchmark results saved to:")
    print(f"- Summary CSV: {benchmark_csv_file}")
    print(f"- Detailed CSV: logs/detailed_benchmark_results.csv")

if __name__ == "__main__":
    main()
