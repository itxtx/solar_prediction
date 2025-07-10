#!/usr/bin/env python3
"""
SARIMA Memory Benchmarking Script
=================================

This script:
1. Generates a realistic 15-minute solar irradiance dataset (~100k rows)
2. Runs SARIMA training/forecasting with MemoryTracker
3. Logs peak & delta memory usage
4. Stores plots and metrics for analysis

Usage:
    python scripts/sarima_memory_benchmark.py
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
    prepare_time_series_data, fit_sarima_model, 
    create_baseline_models, calculate_metrics
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sarima_memory_benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SARIMAMemoryBenchmark:
    """Memory benchmarking for SARIMA time series models."""
    
    def __init__(self, output_dir: str = "sarima_memory_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Memory tracking
        self.memory_tracker = MemoryTracker(verbose=True)
        self.process = psutil.Process()
        
        # Results storage
        self.benchmark_results = {}
        
        logger.info(f"SARIMA memory benchmark initialized. Output directory: {output_dir}")
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
    
    def run_sarima_benchmark(self, df: pd.DataFrame, 
                           orders_to_test: List[Tuple] = None) -> Dict:
        """
        Run SARIMA training/forecasting with comprehensive memory tracking.
        
        Args:
            df: Solar dataset DataFrame
            orders_to_test: List of (order, seasonal_order) tuples to test
            
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Starting SARIMA memory benchmark")
        
        # Default orders to test if none provided
        if orders_to_test is None:
            orders_to_test = [
                ((1, 1, 1), (1, 1, 1, 96)),  # Simple SARIMA
                ((2, 1, 1), (1, 1, 1, 96)),  # More complex AR
                ((1, 1, 2), (1, 1, 1, 96)),  # More complex MA
            ]
        
        # Initialize memory tracking
        self.memory_tracker.reset_peak_memory()
        self.memory_tracker.snapshot("sarima_start", "SARIMA benchmark start")
        
        process_memory_start = self.process.memory_info().rss / 1024**2  # MB
        start_time = time.time()
        
        results = {
            'dataset_info': {},
            'models': {},
            'memory_tracking': {},
            'timing': {}
        }
        
        try:
            # Prepare data with memory tracking
            with self.memory_tracker.track_memory("data_preparation"):
                train_series, test_series, seasonality = prepare_time_series_data(
                    df, time_col='Time', target_col='GHI', test_size=0.2
                )
            
            logger.info(f"Data prepared: train={len(train_series)}, test={len(test_series)}, seasonality={seasonality}")
            
            results['dataset_info'] = {
                'total_samples': len(df),
                'training_samples': len(train_series),
                'test_samples': len(test_series),
                'seasonality': seasonality,
                'frequency': '15T'
            }
            
            # Test different SARIMA configurations
            best_model = None
            best_aic = float('inf')
            best_forecast = None
            
            for i, (order, seasonal_order) in enumerate(orders_to_test):
                model_name = f"SARIMA{order}×{seasonal_order}"
                logger.info(f"Testing {model_name}")
                
                model_start_time = time.time()
                model_memory_start = self.process.memory_info().rss / 1024**2
                
                # Update seasonal_order with actual seasonality
                seasonal_order = (seasonal_order[0], seasonal_order[1], seasonal_order[2], seasonality)
                
                with self.memory_tracker.track_memory(f"model_{i}_fitting"):
                    try:
                        sarima_model = fit_sarima_model(
                            train_series, 
                            order=order,
                            seasonal_order=seasonal_order
                        )
                        
                        if sarima_model is None:
                            logger.warning(f"Failed to fit {model_name}")
                            continue
                        
                        # Forecasting
                        forecast_steps = len(test_series)
                        forecast = sarima_model.forecast(steps=forecast_steps)
                        forecast_series = pd.Series(forecast, index=test_series.index)
                        
                        # Calculate metrics
                        metrics = calculate_metrics(test_series, forecast_series, model_name, seasonality)
                        
                        model_end_time = time.time()
                        model_memory_end = self.process.memory_info().rss / 1024**2
                        
                        # Store results
                        results['models'][model_name] = {
                            'order': order,
                            'seasonal_order': seasonal_order,
                            'aic': getattr(sarima_model, 'aic', None),
                            'bic': getattr(sarima_model, 'bic', None),
                            'metrics': metrics,
                            'forecast': forecast_series,
                            'training_time': model_end_time - model_start_time,
                            'memory_delta': model_memory_end - model_memory_start,
                            'fitted_successfully': True
                        }
                        
                        # Track best model by AIC
                        current_aic = getattr(sarima_model, 'aic', float('inf'))
                        if current_aic < best_aic:
                            best_aic = current_aic
                            best_model = sarima_model
                            best_forecast = forecast_series
                            results['best_model'] = model_name
                        
                        logger.info(f"{model_name} - AIC: {current_aic:.2f}, RMSE: {metrics['RMSE']:.4f}, Time: {model_end_time - model_start_time:.2f}s")
                        
                    except Exception as e:
                        logger.error(f"Error fitting {model_name}: {e}")
                        results['models'][model_name] = {
                            'order': order,
                            'seasonal_order': seasonal_order,
                            'error': str(e),
                            'fitted_successfully': False
                        }
            
            # Create baseline models for comparison
            with self.memory_tracker.track_memory("baseline_models"):
                baselines = create_baseline_models(train_series, test_series, seasonality)
                
                for baseline_name, baseline_forecast in baselines.items():
                    metrics = calculate_metrics(test_series, baseline_forecast, baseline_name, seasonality)
                    results['models'][baseline_name] = {
                        'type': 'baseline',
                        'metrics': metrics,
                        'forecast': baseline_forecast,
                        'fitted_successfully': True
                    }
            
            # Final memory snapshot
            self.memory_tracker.snapshot("sarima_end", "SARIMA benchmark complete")
            process_memory_end = self.process.memory_info().rss / 1024**2  # MB
            total_time = time.time() - start_time
            
            # Collect overall memory statistics
            memory_info = self.memory_tracker.get_memory_info()
            memory_peak = memory_info['max_allocated']
            memory_delta = process_memory_end - process_memory_start
            
            results['memory_tracking'] = {
                'peak_gpu_memory_mb': memory_peak,
                'cpu_memory_delta_mb': memory_delta,
                'initial_memory_mb': process_memory_start,
                'final_memory_mb': process_memory_end,
                'snapshots': self.memory_tracker.snapshots.copy()
            }
            
            results['timing'] = {
                'total_time': total_time,
                'start_time': start_time,
                'end_time': time.time()
            }
            
            # Store actual vs predicted for best model
            if best_forecast is not None:
                results['best_forecast'] = best_forecast
                results['actual_values'] = test_series
            
            logger.info(f"SARIMA benchmark completed:")
            logger.info(f"  Total time: {total_time:.2f}s")
            logger.info(f"  Peak GPU memory: {memory_peak:.1f}MB")
            logger.info(f"  CPU memory delta: {memory_delta:.1f}MB")
            logger.info(f"  Best model: {results.get('best_model', 'None')} (AIC: {best_aic:.2f})")
            
            return results
            
        except Exception as e:
            logger.error(f"SARIMA benchmark failed: {e}")
            self.memory_tracker.snapshot("sarima_error", f"SARIMA failed: {str(e)}")
            raise
    
    def create_analysis_plots(self, results: Dict) -> None:
        """
        Create comprehensive analysis plots for SARIMA benchmark results.
        
        Args:
            results: Dictionary with benchmark results
        """
        logger.info("Creating analysis plots")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SARIMA Memory & Performance Analysis', fontsize=16)
        
        # Extract successful models
        successful_models = {name: data for name, data in results['models'].items() 
                           if data.get('fitted_successfully', False)}
        
        if not successful_models:
            logger.warning("No successful models to plot")
            return
        
        model_names = list(successful_models.keys())
        
        # Plot 1: Model AIC Comparison
        ax1 = axes[0, 0]
        sarima_models = {name: data for name, data in successful_models.items() 
                        if data.get('type') != 'baseline' and 'aic' in data}
        
        if sarima_models:
            names = list(sarima_models.keys())
            aics = [data['aic'] for data in sarima_models.values()]
            bars1 = ax1.bar(names, aics, alpha=0.7)
            ax1.set_title('SARIMA Model AIC Comparison')
            ax1.set_ylabel('AIC (lower is better)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, aic in zip(bars1, aics):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(aics) * 0.01,
                        f'{aic:.0f}', ha='center', va='bottom', fontsize=8)
        else:
            ax1.text(0.5, 0.5, 'No SARIMA models with AIC available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('SARIMA Model AIC (Not Available)')
        
        # Plot 2: RMSE Comparison
        ax2 = axes[0, 1]
        rmse_values = [data['metrics']['RMSE'] for data in successful_models.values()]
        bars2 = ax2.bar(model_names, rmse_values, alpha=0.7)
        ax2.set_title('Model RMSE Comparison')
        ax2.set_ylabel('RMSE (lower is better)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar, rmse in zip(bars2, rmse_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values) * 0.01,
                    f'{rmse:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Training Time vs Memory Usage
        ax3 = axes[0, 2]
        sarima_with_timing = {name: data for name, data in successful_models.items() 
                             if 'training_time' in data and 'memory_delta' in data}
        
        if sarima_with_timing:
            times = [data['training_time'] for data in sarima_with_timing.values()]
            memories = [data['memory_delta'] for data in sarima_with_timing.values()]
            names = list(sarima_with_timing.keys())
            
            scatter = ax3.scatter(times, memories, s=100, alpha=0.7)
            ax3.set_xlabel('Training Time (s)')
            ax3.set_ylabel('Memory Delta (MB)')
            ax3.set_title('Time vs Memory Trade-off')
            ax3.grid(True, alpha=0.3)
            
            # Add labels
            for i, name in enumerate(names):
                ax3.annotate(name, (times[i], memories[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No timing/memory data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Time vs Memory (Not Available)')
        
        # Plot 4: Best Model Forecast vs Actual
        ax4 = axes[1, 0]
        if 'best_forecast' in results and 'actual_values' in results:
            actual = results['actual_values']
            forecast = results['best_forecast']
            
            # Plot subset for visibility
            plot_points = min(len(actual), 200)
            actual_subset = actual.iloc[:plot_points]
            forecast_subset = forecast.iloc[:plot_points]
            
            ax4.plot(actual_subset.index, actual_subset.values, 'o-', 
                    label='Actual', linewidth=2, markersize=3)
            ax4.plot(forecast_subset.index, forecast_subset.values, 's-', 
                    label=f'Forecast ({results.get("best_model", "Best")})', 
                    linewidth=2, markersize=3, alpha=0.8)
            ax4.set_title('Best Model: Actual vs Forecast')
            ax4.set_ylabel('GHI')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No forecast data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Forecast Plot (Not Available)')
        
        # Plot 5: Memory Usage Timeline
        ax5 = axes[1, 1]
        snapshots = results['memory_tracking']['snapshots']
        if snapshots:
            times = []
            allocated_memory = []
            
            for name, snapshot in snapshots.items():
                # Convert timestamp to relative time
                time_parts = snapshot.timestamp.split(':')
                relative_time = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                times.append(relative_time)
                allocated_memory.append(snapshot.allocated)
            
            # Normalize times to start from 0
            if times:
                start_time = min(times)
                times = [(t - start_time) for t in times]
            
            ax5.plot(times, allocated_memory, 'o-', linewidth=2, markersize=6)
            ax5.set_title('GPU Memory Usage Timeline')
            ax5.set_xlabel('Time (seconds)')
            ax5.set_ylabel('Allocated Memory (MB)')
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No memory snapshots available', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Memory Timeline (Not Available)')
        
        # Plot 6: Model Performance Ranking
        ax6 = axes[1, 2]
        rmse_with_names = [(name, data['metrics']['RMSE']) for name, data in successful_models.items()]
        rmse_with_names.sort(key=lambda x: x[1])  # Sort by RMSE
        
        names_sorted = [x[0] for x in rmse_with_names]
        rmse_sorted = [x[1] for x in rmse_with_names]
        
        bars6 = ax6.barh(names_sorted, rmse_sorted, alpha=0.7)
        ax6.set_title('Model Performance Ranking')
        ax6.set_xlabel('RMSE (lower is better)')
        ax6.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, rmse in zip(bars6, rmse_sorted):
            ax6.text(bar.get_width() + max(rmse_sorted) * 0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{rmse:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'sarima_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Analysis plot saved to {plot_path}")
        plt.show()
    
    def save_detailed_results(self, results: Dict) -> None:
        """Save detailed benchmark results to files."""
        
        # Save numerical results as CSV
        summary_data = []
        for model_name, model_data in results['models'].items():
            if model_data.get('fitted_successfully', False):
                row = {
                    'Model': model_name,
                    'Type': model_data.get('type', 'SARIMA'),
                    'Order': str(model_data.get('order', 'N/A')),
                    'Seasonal_Order': str(model_data.get('seasonal_order', 'N/A')),
                    'AIC': model_data.get('aic', 'N/A'),
                    'BIC': model_data.get('bic', 'N/A'),
                    'RMSE': model_data['metrics']['RMSE'],
                    'MAE': model_data['metrics']['MAE'],
                    'MAPE': model_data['metrics']['MAPE'],
                    'R2': model_data['metrics']['R²'],
                    'Training_Time_s': model_data.get('training_time', 'N/A'),
                    'Memory_Delta_MB': model_data.get('memory_delta', 'N/A')
                }
                summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.output_dir, 'sarima_benchmark_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary results saved to {summary_path}")
        
        # Save memory snapshots
        snapshots = results['memory_tracking']['snapshots']
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
            snapshot_path = os.path.join(self.output_dir, 'memory_snapshots.csv')
            snapshot_df.to_csv(snapshot_path, index=False)
            logger.info(f"Memory snapshots saved to {snapshot_path}")
        
        # Save full results as JSON for later analysis
        import json
        
        # Convert non-serializable objects
        results_copy = results.copy()
        for model_name, model_data in results_copy['models'].items():
            if 'forecast' in model_data:
                # Convert Series to list
                results_copy['models'][model_name]['forecast'] = model_data['forecast'].tolist()
        
        if 'best_forecast' in results_copy:
            results_copy['best_forecast'] = results_copy['best_forecast'].tolist()
        if 'actual_values' in results_copy:
            results_copy['actual_values'] = results_copy['actual_values'].tolist()
        
        # Remove memory snapshots from JSON (too complex)
        if 'snapshots' in results_copy['memory_tracking']:
            del results_copy['memory_tracking']['snapshots']
        
        json_path = os.path.join(self.output_dir, 'benchmark_results.json')
        with open(json_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
        logger.info(f"Full results saved to {json_path}")
        
        # Generate summary report
        self.generate_summary_report(results)
    
    def generate_summary_report(self, results: Dict) -> None:
        """Generate a human-readable summary report."""
        
        report = []
        report.append("# SARIMA Memory Benchmark Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Dataset information
        dataset_info = results['dataset_info']
        report.append("## Dataset Information\n")
        report.append(f"- Total samples: {dataset_info['total_samples']:,}\n")
        report.append(f"- Training samples: {dataset_info['training_samples']:,}\n")
        report.append(f"- Test samples: {dataset_info['test_samples']:,}\n")
        report.append(f"- Seasonality period: {dataset_info['seasonality']}\n")
        report.append(f"- Data frequency: {dataset_info['frequency']}\n")
        
        # Memory usage summary
        memory_info = results['memory_tracking']
        report.append("\n## Memory Usage Summary\n")
        report.append(f"- Peak GPU memory: {memory_info['peak_gpu_memory_mb']:.1f}MB\n")
        report.append(f"- CPU memory delta: {memory_info['cpu_memory_delta_mb']:.1f}MB\n")
        report.append(f"- Initial memory: {memory_info['initial_memory_mb']:.1f}MB\n")
        report.append(f"- Final memory: {memory_info['final_memory_mb']:.1f}MB\n")
        
        # Timing information
        timing_info = results['timing']
        report.append("\n## Performance Summary\n")
        report.append(f"- Total benchmark time: {timing_info['total_time']:.2f} seconds\n")
        
        # Model results
        successful_models = {name: data for name, data in results['models'].items() 
                           if data.get('fitted_successfully', False)}
        
        report.append(f"- Successfully fitted models: {len(successful_models)}\n")
        
        if 'best_model' in results:
            best_model_name = results['best_model']
            best_model_data = results['models'][best_model_name]
            report.append(f"- Best model: {best_model_name}\n")
            report.append(f"- Best model AIC: {best_model_data.get('aic', 'N/A')}\n")
            report.append(f"- Best model RMSE: {best_model_data['metrics']['RMSE']:.4f}\n")
        
        # Detailed model results
        report.append("\n## Model Performance Details\n")
        
        # Sort models by RMSE
        model_performance = [(name, data['metrics']['RMSE']) for name, data in successful_models.items()]
        model_performance.sort(key=lambda x: x[1])
        
        for name, rmse in model_performance:
            model_data = successful_models[name]
            report.append(f"### {name}\n")
            report.append(f"- RMSE: {rmse:.4f}\n")
            report.append(f"- MAE: {model_data['metrics']['MAE']:.4f}\n")
            report.append(f"- MAPE: {model_data['metrics']['MAPE']:.2f}%\n")
            report.append(f"- R²: {model_data['metrics']['R²']:.4f}\n")
            
            if 'aic' in model_data:
                report.append(f"- AIC: {model_data['aic']:.2f}\n")
            if 'training_time' in model_data:
                report.append(f"- Training time: {model_data['training_time']:.2f}s\n")
            if 'memory_delta' in model_data:
                report.append(f"- Memory delta: {model_data['memory_delta']:.1f}MB\n")
            report.append("\n")
        
        # Memory efficiency analysis
        report.append("## Memory Efficiency Analysis\n")
        memory_per_sample = memory_info['cpu_memory_delta_mb'] / dataset_info['total_samples'] * 1000
        report.append(f"- Memory usage per 1k samples: {memory_per_sample:.3f}MB\n")
        
        time_per_sample = timing_info['total_time'] / dataset_info['total_samples'] * 1000
        report.append(f"- Processing time per 1k samples: {time_per_sample:.3f}s\n")
        
        # Recommendations
        report.append("\n## Recommendations\n")
        if memory_info['cpu_memory_delta_mb'] < 100:
            report.append("- Memory usage is efficient for this dataset size\n")
        else:
            report.append("- Consider using smaller model orders or data subsets for memory-constrained environments\n")
        
        if timing_info['total_time'] < 60:
            report.append("- Processing time is reasonable for interactive analysis\n")
        else:
            report.append("- Consider simpler models or parallel processing for faster results\n")
        
        # Save report
        report_path = os.path.join(self.output_dir, 'benchmark_report.md')
        with open(report_path, 'w') as f:
            f.writelines(report)
        logger.info(f"Summary report saved to {report_path}")

def main():
    """Main execution function."""
    logger.info("Starting SARIMA memory benchmark")
    
    # Initialize benchmark
    benchmark = SARIMAMemoryBenchmark()
    
    try:
        # Generate realistic dataset
        logger.info("Step 1: Generating realistic 15-minute dataset (~25k rows)")
        df = benchmark.generate_realistic_solar_dataset(
            start_date="2023-01-01",
            days=20,  # ~20k rows for faster benchmarking
            freq="15T"
        )
        
        logger.info(f"Dataset generated: {len(df)} rows")
        
        # Define SARIMA models to test (simplified for speed)
        orders_to_test = [
            ((1, 1, 1), (0, 0, 0, 0)),   # ARIMA (non-seasonal)
            ((2, 1, 1), (0, 0, 0, 0)),   # ARIMA with more AR
            ((1, 1, 2), (0, 0, 0, 0)),   # ARIMA with more MA
            ((1, 1, 1), (1, 0, 1, 24)),  # Simple SARIMA with 24-hour cycle
        ]
        
        # Run SARIMA benchmark
        logger.info("Step 2: Running SARIMA memory benchmark")
        results = benchmark.run_sarima_benchmark(df, orders_to_test)
        
        # Generate analysis and reports
        logger.info("Step 3: Creating analysis plots and reports")
        benchmark.create_analysis_plots(results)
        benchmark.save_detailed_results(results)
        
        logger.info("SARIMA memory benchmark completed successfully!")
        logger.info(f"Results saved to: {benchmark.output_dir}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()
