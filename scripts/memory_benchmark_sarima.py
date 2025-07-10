#!/usr/bin/env python3
"""
Memory Benchmarking Script for SARIMA vs LSTM/GRU
==================================================

This script:
1. Generates a realistic 15-minute solar irradiance dataset (~100k rows)
2. Runs SARIMA training/forecasting with MemoryTracker
3. Logs peak & delta memory usage
4. Compares with LSTM/GRU training logs that leverage batching and snapshots
5. Stores plots and metrics for later recommendation justifications

Usage:
    python scripts/memory_benchmark_sarima.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import project modules
from solar_prediction.memory_tracker import MemoryTracker, MemorySnapshot
from solar_prediction.sarima import (
    prepare_time_series_data, fit_sarima_model, auto_sarima_selection,
    create_baseline_models, calculate_metrics, create_metrics_summary
)
from solar_prediction.lstm import WeatherLSTM, create_model_hyperparameters_from_config, create_training_config_from_config
from solar_prediction.gru import WeatherGRU, GRUModelHyperparameters, TrainingConfig as GRUTrainingConfig
from solar_prediction.data_prep import prepare_weather_data
from solar_prediction.config import get_config
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('memory_benchmark_sarima.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MemoryBenchmark:
    """Comprehensive memory benchmarking for time series models."""
    
    def __init__(self, output_dir: str = "memory_benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Memory tracking
        self.memory_tracker = MemoryTracker(verbose=True)
        self.process = psutil.Process()
        
        # Results storage
        self.benchmark_results = {}
        self.memory_logs = {}
        self.timing_logs = {}
        
        logger.info(f"Memory benchmark initialized. Output directory: {output_dir}")
        self.memory_tracker.log_device_info()
    
    def generate_realistic_solar_dataset(self, 
                                       start_date: str = "2023-01-01",
                                       days: int = 70,  # ~100k rows for 15-min data
                                       freq: str = "15T",
                                       noise_level: float = 0.15) -> pd.DataFrame:
        """
        Generate realistic 15-minute solar irradiance dataset with weather patterns.
        
        Args:
            start_date: Start date for the dataset
            days: Number of days to generate
            freq: Data frequency (15T = 15 minutes)
            noise_level: Amount of random noise to add
            
        Returns:
            DataFrame with realistic solar and weather data
        """
        logger.info(f"Generating realistic solar dataset: {days} days, {freq} frequency")
        
        # Create time index
        start = pd.to_datetime(start_date)
        end = start + timedelta(days=days)
        time_index = pd.date_range(start=start, end=end, freq=freq)
        
        # Base solar irradiance pattern
        hour_of_day = time_index.hour + time_index.minute / 60.0
        day_of_year = time_index.dayofyear
        
        # Solar elevation angle approximation
        solar_elevation = np.maximum(0, 
            np.sin(np.pi * (hour_of_day - 6) / 12) * 
            np.sin(np.pi * (day_of_year - 81) / 182))
        
        # Base GHI (Global Horizontal Irradiance)
        base_ghi = 1000 * solar_elevation ** 1.2
        
        # Add seasonal variation
        seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
        base_ghi *= seasonal_factor
        
        # Add weather patterns (clouds, rain)
        np.random.seed(42)  # Reproducible
        cloud_factor = np.random.beta(2, 2, len(time_index))  # 0-1 cloud cover
        rain_events = np.random.exponential(0.1, len(time_index))
        rain_events = np.where(rain_events > 0.5, rain_events, 0)  # Sparse rain
        
        # Apply weather effects
        ghi = base_ghi * (1 - 0.8 * cloud_factor) * (1 - 0.3 * np.minimum(rain_events, 1))
        
        # Add realistic noise
        noise = np.random.normal(0, noise_level * np.maximum(ghi, 10), len(time_index))
        ghi = np.maximum(0, ghi + noise)
        
        # Generate correlated weather features
        temperature = 15 + 15 * seasonal_factor + 10 * solar_elevation + np.random.normal(0, 2, len(time_index))
        humidity = 50 + 30 * cloud_factor + 20 * np.minimum(rain_events, 1) + np.random.normal(0, 5, len(time_index))
        pressure = 1013 + np.random.normal(0, 5, len(time_index))
        wind_speed = 3 + 7 * cloud_factor + np.random.exponential(2, len(time_index))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Time': time_index,
            'GHI': ghi,
            'Temperature': temperature,
            'Humidity': np.clip(humidity, 0, 100),
            'Pressure': pressure,
            'WindSpeed': wind_speed,
            'CloudCover': cloud_factor * 100,
            'Rain': rain_events,
            'HourOfDay': hour_of_day,
            'Month': time_index.month,
            'DayOfYear': day_of_year
        })
        
        logger.info(f"Generated dataset: {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"GHI statistics: min={df['GHI'].min():.1f}, max={df['GHI'].max():.1f}, mean={df['GHI'].mean():.1f}")
        
        # Save dataset
        dataset_path = os.path.join(self.output_dir, "synthetic_solar_dataset.csv")
        df.to_csv(dataset_path, index=False)
        logger.info(f"Dataset saved to {dataset_path}")
        
        return df
    
    def run_sarima_benchmark(self, df: pd.DataFrame) -> Dict:
        """
        Run SARIMA training/forecasting with comprehensive memory tracking.
        
        Args:
            df: Solar dataset DataFrame
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Starting SARIMA memory benchmark")
        
        # Initialize memory tracking
        self.memory_tracker.reset_peak_memory()
        self.memory_tracker.snapshot("sarima_start", "SARIMA benchmark start")
        
        process_memory_start = self.process.memory_info().rss / 1024**2  # MB
        start_time = time.time()
        
        try:
            # Prepare data with memory tracking
            with self.memory_tracker.track_memory("data_preparation"):
                train_series, test_series, seasonality = prepare_time_series_data(
                    df, time_col='Time', target_col='GHI', test_size=0.2
                )
            
            logger.info(f"Data prepared: train={len(train_series)}, test={len(test_series)}, seasonality={seasonality}")
            
            # SARIMA model fitting with memory tracking (using simple parameters for speed)
            with self.memory_tracker.track_memory("sarima_model_selection"):
                # Use a simple SARIMA model for benchmarking purposes
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, seasonality)
                logger.info(f"Using simple SARIMA{order} × {seasonal_order} for benchmark")
                sarima_model = fit_sarima_model(
                    train_series, 
                    order=order,
                    seasonal_order=seasonal_order
                )
            
            if sarima_model is None:
                raise ValueError("SARIMA model selection failed")
            
            # Forecasting with memory tracking
            with self.memory_tracker.track_memory("sarima_forecasting"):
                forecast_steps = len(test_series)
                forecast = sarima_model.forecast(steps=forecast_steps)
                forecast_series = pd.Series(forecast, index=test_series.index)
            
            # Calculate metrics
            with self.memory_tracker.track_memory("metrics_calculation"):
                metrics = calculate_metrics(test_series, forecast_series, "SARIMA", seasonality)
                
                # Create baseline models for comparison
                baselines = create_baseline_models(train_series, test_series, seasonality)
                all_predictions = {"SARIMA": forecast_series, **baselines}
                
                # Calculate metrics for all models manually to include seasonality
                metrics_list = []
                for model_name, predictions in all_predictions.items():
                    model_metrics = calculate_metrics(test_series, predictions, model_name, seasonality)
                    if model_metrics:
                        metrics_list.append(model_metrics)
                
                # Create summary dataframe
                if metrics_list:
                    import pandas as pd
                    metrics_summary = pd.DataFrame(metrics_list).sort_values('RMSE')
                else:
                    metrics_summary = None
            
            # Final memory snapshot
            self.memory_tracker.snapshot("sarima_end", "SARIMA benchmark complete")
            process_memory_end = self.process.memory_info().rss / 1024**2  # MB
            total_time = time.time() - start_time
            
            # Collect memory statistics
            memory_info = self.memory_tracker.get_memory_info()
            memory_peak = memory_info['max_allocated']
            memory_delta = process_memory_end - process_memory_start
            
            # Store results
            results = {
                'model_type': 'SARIMA',
                'dataset_size': len(df),
                'training_size': len(train_series),
                'test_size': len(test_series),
                'seasonality': seasonality,
                'total_time': total_time,
                'peak_gpu_memory_mb': memory_peak,
                'cpu_memory_delta_mb': memory_delta,
                'final_memory_mb': process_memory_end,
                'metrics': metrics,
                'model_order': getattr(sarima_model, 'order', 'unknown'),
                'seasonal_order': getattr(sarima_model, 'seasonal_order', 'unknown'),
                'aic': getattr(sarima_model, 'aic', None),
                'predictions': forecast_series,
                'actual': test_series,
                'memory_snapshots': self.memory_tracker.snapshots.copy()
            }
            
            logger.info(f"SARIMA benchmark completed:")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Peak GPU memory: {memory_peak:.1f}MB")
            logger.info(f"  CPU memory delta: {memory_delta:.1f}MB")
            logger.info(f"  Model AIC: {results['aic']}")
            logger.info(f"  Test RMSE: {metrics['RMSE']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"SARIMA benchmark failed: {e}")
            self.memory_tracker.snapshot("sarima_error", f"SARIMA failed: {str(e)}")
            raise
    
    def run_lstm_benchmark(self, df: pd.DataFrame) -> Dict:
        """
        Run LSTM training with memory tracking for comparison.
        
        Args:
            df: Solar dataset DataFrame
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Starting LSTM memory benchmark")
        
        # Initialize memory tracking
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lstm_memory_tracker = MemoryTracker(device=device, verbose=True)
        lstm_memory_tracker.reset_peak_memory()
        lstm_memory_tracker.snapshot("lstm_start", "LSTM benchmark start")
        
        process_memory_start = self.process.memory_info().rss / 1024**2  # MB
        start_time = time.time()
        
        try:
            # Prepare data for LSTM
            config = get_config()
            
            # Use a subset for LSTM due to memory constraints
            subset_size = min(len(df), 10000)  # Limit for memory efficiency
            df_subset = df.iloc[-subset_size:].copy()
            
            # Simple feature preparation for LSTM
            features = ['GHI', 'Temperature', 'Humidity', 'WindSpeed', 'HourOfDay']
            X = df_subset[features].values
            y = df_subset['GHI'].values.reshape(-1, 1)
            
            # Create sequences
            sequence_length = 24  # 6 hours of 15-min data
            X_sequences, y_sequences = [], []
            
            for i in range(sequence_length, len(X)):
                X_sequences.append(X[i-sequence_length:i])
                y_sequences.append(y[i])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            # Train-validation split
            split_idx = int(0.8 * len(X_sequences))
            X_train, X_val = X_sequences[:split_idx], X_sequences[split_idx:]
            y_train, y_val = y_sequences[:split_idx], y_sequences[split_idx:]
            
            # Create model with smaller architecture for memory efficiency
            model_params = create_model_hyperparameters_from_config(
                input_dim=len(features),
                config_override={'hidden_dim': 32, 'num_layers': 1, 'dropout_prob': 0.2}
            )
            model = WeatherLSTM(model_params)
            
            # Training config with reduced parameters
            train_config = create_training_config_from_config(
                config_override={'epochs': 10, 'batch_size': 16, 'patience': 5}
            )
            
            # Train with memory tracking
            model.fit(
                X_train, y_train, X_val, y_val,
                train_config, device=device,
                memory_tracker=lstm_memory_tracker
            )
            
            # Prediction
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                predictions = model(X_val_tensor).cpu().numpy()
            
            # Final memory snapshot
            lstm_memory_tracker.snapshot("lstm_end", "LSTM benchmark complete")
            process_memory_end = self.process.memory_info().rss / 1024**2  # MB
            total_time = time.time() - start_time
            
            # Collect memory statistics
            memory_info = lstm_memory_tracker.get_memory_info()
            memory_peak = memory_info['max_allocated']
            memory_delta = process_memory_end - process_memory_start
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            rmse = np.sqrt(mean_squared_error(y_val, predictions))
            mae = mean_absolute_error(y_val, predictions)
            r2 = r2_score(y_val, predictions)
            
            results = {
                'model_type': 'LSTM',
                'dataset_size': len(df_subset),
                'training_size': len(X_train),
                'test_size': len(X_val),
                'total_time': total_time,
                'peak_gpu_memory_mb': memory_peak,
                'cpu_memory_delta_mb': memory_delta,
                'final_memory_mb': process_memory_end,
                'metrics': {'RMSE': rmse, 'MAE': mae, 'R²': r2},
                'predictions': predictions.flatten(),
                'actual': y_val.flatten(),
                'memory_snapshots': lstm_memory_tracker.snapshots.copy()
            }
            
            logger.info(f"LSTM benchmark completed:")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Peak GPU memory: {memory_peak:.1f}MB")
            logger.info(f"  CPU memory delta: {memory_delta:.1f}MB")
            logger.info(f"  Test RMSE: {rmse:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"LSTM benchmark failed: {e}")
            lstm_memory_tracker.snapshot("lstm_error", f"LSTM failed: {str(e)}")
            raise
    
    def run_gru_benchmark(self, df: pd.DataFrame) -> Dict:
        """
        Run GRU training with memory tracking for comparison.
        
        Args:
            df: Solar dataset DataFrame
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Starting GRU memory benchmark")
        
        # Initialize memory tracking
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gru_memory_tracker = MemoryTracker(device=device, verbose=True)
        gru_memory_tracker.reset_peak_memory()
        gru_memory_tracker.snapshot("gru_start", "GRU benchmark start")
        
        process_memory_start = self.process.memory_info().rss / 1024**2  # MB
        start_time = time.time()
        
        try:
            # Use a subset for GRU due to memory constraints
            subset_size = min(len(df), 10000)  # Limit for memory efficiency
            df_subset = df.iloc[-subset_size:].copy()
            
            # Simple feature preparation for GRU
            features = ['GHI', 'Temperature', 'Humidity', 'WindSpeed', 'HourOfDay']
            X = df_subset[features].values
            y = df_subset['GHI'].values.reshape(-1, 1)
            
            # Create sequences
            sequence_length = 24  # 6 hours of 15-min data
            X_sequences, y_sequences = [], []
            
            for i in range(sequence_length, len(X)):
                X_sequences.append(X[i-sequence_length:i])
                y_sequences.append(y[i])
            
            X_sequences = np.array(X_sequences)
            y_sequences = np.array(y_sequences)
            
            # Train-validation split
            split_idx = int(0.8 * len(X_sequences))
            X_train, X_val = X_sequences[:split_idx], X_sequences[split_idx:]
            y_train, y_val = y_sequences[:split_idx], y_sequences[split_idx:]
            
            # Create model with smaller architecture for memory efficiency
            model_params = GRUModelHyperparameters(
                input_dim=len(features),
                hidden_dim=32,
                num_layers=1,
                output_dim=1,
                dropout_prob=0.2
            )
            model = WeatherGRU(model_params)
            
            # Training config with reduced parameters
            train_config = GRUTrainingConfig(
                epochs=10,
                batch_size=16,
                learning_rate=0.001,
                patience=5,
                scheduler_type='plateau'
            )
            
            # Train with memory tracking
            model.fit(
                X_train, y_train, X_val, y_val,
                train_config, device=device,
                memory_tracker=gru_memory_tracker
            )
            
            # Prediction
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val).to(device)
                predictions = model(X_val_tensor).cpu().numpy()
            
            # Final memory snapshot
            gru_memory_tracker.snapshot("gru_end", "GRU benchmark complete")
            process_memory_end = self.process.memory_info().rss / 1024**2  # MB
            total_time = time.time() - start_time
            
            # Collect memory statistics
            memory_info = gru_memory_tracker.get_memory_info()
            memory_peak = memory_info['max_allocated']
            memory_delta = process_memory_end - process_memory_start
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            rmse = np.sqrt(mean_squared_error(y_val, predictions))
            mae = mean_absolute_error(y_val, predictions)
            r2 = r2_score(y_val, predictions)
            
            results = {
                'model_type': 'GRU',
                'dataset_size': len(df_subset),
                'training_size': len(X_train),
                'test_size': len(X_val),
                'total_time': total_time,
                'peak_gpu_memory_mb': memory_peak,
                'cpu_memory_delta_mb': memory_delta,
                'final_memory_mb': process_memory_end,
                'metrics': {'RMSE': rmse, 'MAE': mae, 'R²': r2},
                'predictions': predictions.flatten(),
                'actual': y_val.flatten(),
                'memory_snapshots': gru_memory_tracker.snapshots.copy()
            }
            
            logger.info(f"GRU benchmark completed:")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Peak GPU memory: {memory_peak:.1f}MB")
            logger.info(f"  CPU memory delta: {memory_delta:.1f}MB")
            logger.info(f"  Test RMSE: {rmse:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"GRU benchmark failed: {e}")
            gru_memory_tracker.snapshot("gru_error", f"GRU failed: {str(e)}")
            raise
    
    def create_comparison_plots(self, results: Dict[str, Dict]) -> None:
        """
        Create comprehensive comparison plots for memory usage and performance.
        
        Args:
            results: Dictionary of benchmark results by model type
        """
        logger.info("Creating comparison plots")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Memory & Performance Comparison: SARIMA vs LSTM vs GRU', fontsize=16)
        
        # Extract data for plotting
        models = list(results.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        plot_colors = colors[:len(models)]  # Use only needed colors
        
        # Plot 1: Peak Memory Usage
        ax1 = axes[0, 0]
        peak_memories = [results[model]['peak_gpu_memory_mb'] for model in models]
        bars1 = ax1.bar(models, peak_memories, color=plot_colors)
        ax1.set_title('Peak GPU Memory Usage')
        ax1.set_ylabel('Memory (MB)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, peak_memories):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(peak_memories),
                    f'{value:.1f}MB', ha='center', va='bottom')
        
        # Plot 2: CPU Memory Delta
        ax2 = axes[0, 1]
        memory_deltas = [results[model]['cpu_memory_delta_mb'] for model in models]
        bars2 = ax2.bar(models, memory_deltas, color=plot_colors)
        ax2.set_title('CPU Memory Delta')
        ax2.set_ylabel('Memory Change (MB)')
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, memory_deltas):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(memory_deltas),
                    f'{value:.1f}MB', ha='center', va='bottom')
        
        # Plot 3: Training Time
        ax3 = axes[0, 2]
        training_times = [results[model]['total_time'] for model in models]
        bars3 = ax3.bar(models, training_times, color=plot_colors)
        ax3.set_title('Total Training Time')
        ax3.set_ylabel('Time (seconds)')
        ax3.grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, training_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(training_times),
                    f'{value:.1f}s', ha='center', va='bottom')
        
        # Plot 4: RMSE Comparison
        ax4 = axes[1, 0]
        rmse_values = [results[model]['metrics']['RMSE'] for model in models]
        bars4 = ax4.bar(models, rmse_values, color=plot_colors)
        ax4.set_title('Test RMSE')
        ax4.set_ylabel('RMSE')
        ax4.grid(True, alpha=0.3)
        
        for bar, value in zip(bars4, rmse_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(rmse_values),
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 5: Memory vs Time Trade-off
        ax5 = axes[1, 1]
        # Use only the number of colors needed
        scatter_colors = colors[:len(models)]
        ax5.scatter(training_times, peak_memories, c=scatter_colors, s=200, alpha=0.7)
        for i, model in enumerate(models):
            ax5.annotate(model, (training_times[i], peak_memories[i]),
                        xytext=(5, 5), textcoords='offset points')
        ax5.set_xlabel('Training Time (s)')
        ax5.set_ylabel('Peak GPU Memory (MB)')
        ax5.set_title('Memory vs Time Trade-off')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Dataset Size vs Memory Efficiency
        ax6 = axes[1, 2]
        dataset_sizes = [results[model]['dataset_size'] for model in models]
        memory_per_sample = [peak_memories[i] / dataset_sizes[i] * 1000 for i in range(len(models))]
        bars6 = ax6.bar(models, memory_per_sample, color=plot_colors)
        ax6.set_title('Memory per 1000 Samples')
        ax6.set_ylabel('Memory per 1k samples (MB)')
        ax6.grid(True, alpha=0.3)
        
        for bar, value in zip(bars6, memory_per_sample):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(memory_per_sample),
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'memory_performance_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {plot_path}")
        plt.show()
        
        # Create memory timeline plots for each model
        self.create_memory_timeline_plots(results)
    
    def create_memory_timeline_plots(self, results: Dict[str, Dict]) -> None:
        """Create detailed memory timeline plots for each model."""
        
        fig, axes = plt.subplots(len(results), 1, figsize=(15, 4 * len(results)))
        if len(results) == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(results.items()):
            ax = axes[idx]
            
            snapshots = result['memory_snapshots']
            if not snapshots:
                ax.text(0.5, 0.5, f'No memory snapshots for {model_name}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Extract memory data
            times = []
            allocated_memory = []
            cached_memory = []
            
            for name, snapshot in snapshots.items():
                # Convert timestamp to relative time (assuming same day)
                time_parts = snapshot.timestamp.split(':')
                relative_time = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                times.append(relative_time)
                allocated_memory.append(snapshot.allocated)
                cached_memory.append(snapshot.cached)
            
            # Normalize times to start from 0
            if times:
                start_time = min(times)
                times = [(t - start_time) for t in times]
            
            # Plot memory usage over time
            ax.plot(times, allocated_memory, 'o-', label='Allocated Memory', linewidth=2, markersize=6)
            ax.plot(times, cached_memory, 's-', label='Cached Memory', linewidth=2, markersize=6)
            
            ax.set_title(f'{model_name} Memory Usage Timeline')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Memory (MB)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add annotations for key events
            for i, (name, _) in enumerate(snapshots.items()):
                if 'start' in name or 'end' in name:
                    ax.annotate(name, (times[i], allocated_memory[i]),
                               xytext=(0, 10), textcoords='offset points',
                               ha='center', fontsize=8, rotation=45)
        
        plt.tight_layout()
        timeline_path = os.path.join(self.output_dir, 'memory_timeline_plots.png')
        plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
        logger.info(f"Memory timeline plots saved to {timeline_path}")
        plt.show()
    
    def save_detailed_results(self, results: Dict[str, Dict]) -> None:
        """Save detailed benchmark results to files."""
        
        # Save numerical results as CSV
        summary_data = []
        for model_name, result in results.items():
            summary_data.append({
                'Model': model_name,
                'Dataset_Size': result['dataset_size'],
                'Training_Size': result.get('training_size', 'N/A'),
                'Test_Size': result.get('test_size', 'N/A'),
                'Total_Time_s': result['total_time'],
                'Peak_GPU_Memory_MB': result['peak_gpu_memory_mb'],
                'CPU_Memory_Delta_MB': result['cpu_memory_delta_mb'],
                'Final_Memory_MB': result['final_memory_mb'],
                'Test_RMSE': result['metrics']['RMSE'],
                'Test_MAE': result['metrics'].get('MAE', 'N/A'),
                'Test_R2': result['metrics'].get('R²', 'N/A'),
                'Model_Details': str(result.get('model_order', '')) + ' x ' + str(result.get('seasonal_order', ''))
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, 'benchmark_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary results saved to {summary_path}")
        
        # Save detailed memory snapshots
        for model_name, result in results.items():
            snapshots = result['memory_snapshots']
            if snapshots:
                snapshot_data = []
                for name, snapshot in snapshots.items():
                    snapshot_data.append({
                        'snapshot_name': name,
                        'timestamp': snapshot.timestamp,
                        'allocated_mb': snapshot.allocated,
                        'cached_mb': snapshot.cached,
                        'max_allocated_mb': snapshot.max_allocated,
                        'description': snapshot.description
                    })
                
                snapshot_df = pd.DataFrame(snapshot_data)
                snapshot_path = os.path.join(self.output_dir, f'{model_name.lower()}_memory_snapshots.csv')
                snapshot_df.to_csv(snapshot_path, index=False)
                logger.info(f"{model_name} memory snapshots saved to {snapshot_path}")
        
        # Save recommendations based on results
        self.generate_recommendations(results)
    
    def generate_recommendations(self, results: Dict[str, Dict]) -> None:
        """Generate recommendations based on benchmark results."""
        
        recommendations = []
        recommendations.append("# Memory Benchmark Analysis & Recommendations\n")
        recommendations.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Find best model for different criteria
        best_memory = min(results.items(), key=lambda x: x[1]['peak_gpu_memory_mb'])
        best_time = min(results.items(), key=lambda x: x[1]['total_time'])
        best_accuracy = min(results.items(), key=lambda x: x[1]['metrics']['RMSE'])
        
        recommendations.append("## Performance Summary\n")
        recommendations.append(f"- **Most Memory Efficient**: {best_memory[0]} ({best_memory[1]['peak_gpu_memory_mb']:.1f}MB peak)\n")
        recommendations.append(f"- **Fastest Training**: {best_time[0]} ({best_time[1]['total_time']:.1f}s)\n")
        recommendations.append(f"- **Best Accuracy**: {best_accuracy[0]} (RMSE: {best_accuracy[1]['metrics']['RMSE']:.4f})\n")
        
        # Detailed analysis
        recommendations.append("\n## Detailed Analysis\n")
        
        for model_name, result in results.items():
            recommendations.append(f"### {model_name}\n")
            recommendations.append(f"- Dataset processed: {result['dataset_size']:,} samples\n")
            recommendations.append(f"- Training time: {result['total_time']:.1f} seconds\n")
            recommendations.append(f"- Peak GPU memory: {result['peak_gpu_memory_mb']:.1f}MB\n")
            recommendations.append(f"- CPU memory delta: {result['cpu_memory_delta_mb']:.1f}MB\n")
            recommendations.append(f"- Test RMSE: {result['metrics']['RMSE']:.4f}\n")
            
            # Memory efficiency
            memory_per_sample = result['peak_gpu_memory_mb'] / result['dataset_size'] * 1000
            recommendations.append(f"- Memory per 1k samples: {memory_per_sample:.3f}MB\n")
            
            # Time efficiency
            time_per_sample = result['total_time'] / result['dataset_size'] * 1000
            recommendations.append(f"- Time per 1k samples: {time_per_sample:.3f}s\n")
            recommendations.append("\n")
        
        # Usage recommendations
        recommendations.append("## Usage Recommendations\n")
        
        sarima_result = results.get('SARIMA')
        lstm_result = results.get('LSTM')
        gru_result = results.get('GRU')
        
        if sarima_result and lstm_result:
            if sarima_result['peak_gpu_memory_mb'] < lstm_result['peak_gpu_memory_mb']:
                recommendations.append("- **For memory-constrained environments**: SARIMA shows lower memory usage than deep learning models\n")
            
            if sarima_result['total_time'] < lstm_result['total_time']:
                recommendations.append("- **For quick prototyping**: SARIMA provides faster training times\n")
            else:
                recommendations.append("- **For production training**: Deep learning models may offer better scalability\n")
        
        if all(model in results for model in ['SARIMA', 'LSTM', 'GRU']):
            # Compare accuracy vs efficiency
            models_by_accuracy = sorted(results.items(), key=lambda x: x[1]['metrics']['RMSE'])
            models_by_memory = sorted(results.items(), key=lambda x: x[1]['peak_gpu_memory_mb'])
            
            recommendations.append("- **Accuracy ranking**: " + " > ".join([f"{m[0]} ({m[1]['metrics']['RMSE']:.4f})" for m in models_by_accuracy]) + "\n")
            recommendations.append("- **Memory efficiency ranking**: " + " > ".join([f"{m[0]} ({m[1]['peak_gpu_memory_mb']:.1f}MB)" for m in models_by_memory]) + "\n")
        
        recommendations.append("\n## Deployment Considerations\n")
        recommendations.append("- For datasets >100k samples: Consider deep learning models with batching\n")
        recommendations.append("- For real-time inference: SARIMA may offer lower latency\n")
        recommendations.append("- For limited GPU memory: SARIMA provides CPU-based alternative\n")
        recommendations.append("- For ensemble methods: Combine SARIMA with neural networks for robust predictions\n")
        
        # Save recommendations
        rec_path = os.path.join(self.output_dir, 'recommendations.md')
        with open(rec_path, 'w') as f:
            f.writelines(recommendations)
        logger.info(f"Recommendations saved to {rec_path}")
        
        # Print key recommendations
        logger.info("Key Recommendations:")
        logger.info(f"  Most memory efficient: {best_memory[0]}")
        logger.info(f"  Fastest training: {best_time[0]}")
        logger.info(f"  Best accuracy: {best_accuracy[0]}")

def main():
    """Main execution function."""
    logger.info("Starting comprehensive memory benchmark for SARIMA vs LSTM/GRU")
    
    # Initialize benchmark
    benchmark = MemoryBenchmark()
    
    try:
        # Generate realistic dataset
        logger.info("Step 1: Generating realistic 15-minute dataset (~100k rows)")
        df = benchmark.generate_realistic_solar_dataset(
            start_date="2023-01-01",
            days=70,  # ~100k rows with 15-min frequency
            freq="15T"
        )
        
        logger.info(f"Dataset generated: {len(df)} rows")
        
        # Run benchmarks
        results = {}
        
        # SARIMA benchmark
        logger.info("Step 2: Running SARIMA memory benchmark")
        try:
            results['SARIMA'] = benchmark.run_sarima_benchmark(df)
        except Exception as e:
            logger.error(f"SARIMA benchmark failed: {e}")
        
        # LSTM benchmark (with subset due to memory constraints)
        logger.info("Step 3: Running LSTM memory benchmark")
        try:
            results['LSTM'] = benchmark.run_lstm_benchmark(df)
        except Exception as e:
            logger.error(f"LSTM benchmark failed: {e}")
        
        # GRU benchmark (with subset due to memory constraints)
        logger.info("Step 4: Running GRU memory benchmark")
        try:
            results['GRU'] = benchmark.run_gru_benchmark(df)
        except Exception as e:
            logger.error(f"GRU benchmark failed: {e}")
        
        # Generate comparisons and recommendations
        if results:
            logger.info("Step 5: Creating comparison plots and analysis")
            benchmark.create_comparison_plots(results)
            benchmark.save_detailed_results(results)
            
            logger.info("Memory benchmark completed successfully!")
            logger.info(f"Results saved to: {benchmark.output_dir}")
        else:
            logger.error("No successful benchmark results to analyze")
    
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()
