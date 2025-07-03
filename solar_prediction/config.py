"""
Solar Prediction Project - Centralized Configuration Management

This module provides centralized configuration management for all constants,
hyperparameters, and settings used throughout the solar prediction project.
It uses Pydantic for validation and type safety.

Usage:
    from solar_prediction.config import get_config
    
    config = get_config()
    print(config.data.min_radiation_clip)
    
    # Or for specific sections:
    data_config = get_config().data
    model_config = get_config().models.lstm
"""

from typing import List, Dict, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import os


# =============================================================================
# Data Processing Configuration
# =============================================================================

class DataProcessingConfig(BaseModel):
    """Configuration for data processing operations."""
    
    # Radiation/GHI clipping bounds
    min_radiation_clip: float = Field(0.0, description="Minimum radiation value for clipping")
    max_radiation_clip: float = Field(2000.0, description="Maximum radiation value for clipping")
    
    # Mathematical constants for safety
    max_exp_input: float = Field(700.0, description="Maximum input to np.exp to avoid overflow")
    min_log_input: float = Field(1e-8, description="Minimum input for logarithmic operations")
    
    # Missing data handling
    fill_method: str = Field("ffill", description="Method for filling missing data")
    max_missing_percentage: float = Field(0.3, description="Max percentage of missing data allowed")
    
    # Outlier detection
    outlier_z_threshold: float = Field(3.0, description="Z-score threshold for outlier detection")
    outlier_percentile_lower: float = Field(1.0, description="Lower percentile for outlier clipping")
    outlier_percentile_upper: float = Field(99.0, description="Upper percentile for outlier clipping")


class DataTransformationConfig(BaseModel):
    """Configuration for data transformations."""
    
    # Transformation options
    use_log_transform: bool = Field(False, description="Apply log transformation to target")
    use_power_transform: bool = Field(False, description="Apply Yeo-Johnson power transformation")
    use_piecewise_transform_target: bool = Field(False, description="Apply piecewise transformation for GHI")
    
    # Target preprocessing
    min_target_threshold_initial: Optional[float] = Field(None, description="Initial floor for original target")
    clip_original_target_before_power_transform: bool = Field(False, description="Clip target before power transform")
    original_target_clip_lower_percentile: float = Field(5.0, description="Lower percentile for target clipping")
    original_target_clip_upper_percentile: float = Field(95.0, description="Upper percentile for target clipping")
    
    # Log transformation
    min_radiation_for_log: float = Field(0.1, description="Floor before log(GHI)")
    log_transform_offset: float = Field(1e-6, description="Epsilon for log(x + epsilon)")
    clip_log_transformed_target: bool = Field(False, description="Clip log-transformed target")
    log_clip_lower_percentile: float = Field(1.0, description="Lower percentile for log clipping")
    log_clip_upper_percentile: float = Field(99.0, description="Upper percentile for log clipping")
    
    # Power transformation
    min_radiation_floor_before_power_transform: float = Field(0.0, description="Floor for radiation before power transform")
    
    # Piecewise transformation constants
    piecewise_night_threshold: float = Field(10.0, description="Night threshold for piecewise transform")
    piecewise_moderate_threshold: float = Field(200.0, description="Moderate threshold for piecewise transform")
    piecewise_moderate_slope: float = Field(0.05, description="Slope for moderate region")
    piecewise_high_slope: float = Field(0.002, description="Slope for high region")


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering."""
    
    # Feature creation options
    use_solar_elevation_proxy: bool = Field(True, description="Create solar elevation proxy feature")
    create_low_target_indicator: bool = Field(True, description="Create low target indicator feature")
    low_target_indicator_quantile: float = Field(0.1, description="Quantile for low target indicator")
    
    # Feature selection
    feature_selection_mode: str = Field("all", description="Feature selection mode: 'all', 'basic', 'minimal'")
    
    # Predefined feature sets
    minimal_features: List[str] = Field(
        default_factory=lambda: [
            'Radiation', 'Temperature', 'Humidity', 
            'TimeMinutesSin', 'TimeMinutesCos', 'Cloudcover'
        ],
        description="Minimal feature set"
    )
    
    basic_features: List[str] = Field(
        default_factory=lambda: [
            'Radiation', 'Temperature', 'Pressure', 'Humidity', 'WindSpeed',
            'TimeMinutesSin', 'TimeMinutesCos', 'Cloudcover', 'Rain'
        ],
        description="Basic feature set"
    )
    
    # Time-based feature engineering
    time_slices_per_day: int = Field(24, description="Number of time slices per day")
    minutes_in_day: int = Field(1440, description="Total minutes in a day")


class DataInputConfig(BaseModel):
    """Configuration for input data properties."""
    
    # Column mapping - common input names to standardized names
    target_col_original_name: str = Field(..., description="Original name of target column")
    
    # Common column names
    common_temp_col: str = Field("temp", description="Temperature column name")
    common_pressure_col: str = Field("pressure", description="Pressure column name")
    common_humidity_col: str = Field("humidity", description="Humidity column name") 
    common_wind_speed_col: str = Field("wind_speed", description="Wind speed column name")
    common_ghi_col: str = Field("GHI", description="Global Horizontal Irradiance column name")
    
    # Time columns
    time_col: str = Field("Time", description="Primary timestamp column")
    unix_time_col: str = Field("UNIXTime", description="Fallback timestamp column")
    sunrise_col: str = Field("TimeSunRise", description="Sunrise time column")
    sunset_col: str = Field("TimeSunSet", description="Sunset time column")
    
    # Raw time feature columns
    hour_col_raw: str = Field("hour", description="Raw hour column")
    month_col_raw: str = Field("month", description="Raw month column")
    daylength_col_raw: str = Field("dayLength", description="Day length column")
    is_sun_col_raw: str = Field("isSun", description="Is sun column")
    sunlight_time_daylength_ratio_raw: str = Field("SunlightTime/daylength", description="Sunlight ratio column")


class ScalingConfig(BaseModel):
    """Configuration for data scaling."""
    
    standardize_features: bool = Field(True, description="Use StandardScaler (True) or MinMaxScaler (False)")


class SequenceConfig(BaseModel):
    """Configuration for creating sequences."""
    
    window_size: int = Field(12, description="Sequence window size")
    test_size: float = Field(0.2, description="Test set proportion")
    val_size_from_train_val: float = Field(0.25, description="Validation size as fraction of train+val")
    
    @validator('test_size', 'val_size_from_train_val')
    def validate_proportions(cls, v):
        if not 0 < v < 1:
            raise ValueError("Proportions must be between 0 and 1")
        return v


# =============================================================================
# Model Configuration
# =============================================================================

class LSTMConfig(BaseModel):
    """Configuration for LSTM model."""
    
    # Architecture
    hidden_dim: int = Field(64, description="Hidden dimension size")
    num_layers: int = Field(2, description="Number of LSTM layers")
    output_dim: int = Field(1, description="Output dimension")
    dropout_prob: float = Field(0.3, description="Dropout probability")
    
    # Training
    epochs: int = Field(100, description="Maximum training epochs")
    batch_size: int = Field(32, description="Training batch size")
    learning_rate: float = Field(0.001, description="Initial learning rate")
    patience: int = Field(10, description="Early stopping patience")
    weight_decay: float = Field(1e-5, description="L2 regularization weight decay")
    
    # Learning rate scheduling
    lr_scheduler_factor: float = Field(0.5, description="LR reduction factor")
    lr_scheduler_patience: int = Field(5, description="LR scheduler patience")
    min_lr: float = Field(1e-6, description="Minimum learning rate")
    
    # Gradient clipping
    clip_grad_norm: Optional[float] = Field(1.0, description="Max gradient norm for clipping")
    
    # Scheduler options
    scheduler_type: str = Field("plateau", description="Scheduler type: 'plateau' or 'cosine'")
    t_max_cosine: Optional[int] = Field(None, description="T_max for CosineAnnealingLR")
    
    # Loss function
    loss_type: str = Field("mse", description="Loss type: 'mse', 'combined', 'value_aware'")
    mse_weight: float = Field(0.7, description="MSE weight in combined loss")
    mape_weight: float = Field(0.3, description="MAPE weight in combined loss")
    value_multiplier: float = Field(0.01, description="Value multiplier for value-aware loss")
    
    # Loss function parameters
    mape_epsilon: float = Field(1e-8, description="Epsilon for MAPE calculation")
    mape_clip_percentage: float = Field(100.0, description="MAPE clipping percentage")
    
    # Evaluation
    eval_batch_size: int = Field(256, description="Batch size for evaluation")
    mc_dropout_samples: int = Field(30, description="Number of Monte Carlo dropout samples")
    uncertainty_alpha: float = Field(0.05, description="Alpha for uncertainty confidence intervals")
    
    @validator('dropout_prob')
    def validate_dropout(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Dropout probability must be between 0 and 1")
        return v


class GRUConfig(BaseModel):
    """Configuration for GRU model."""
    
    # Architecture
    hidden_dim: int = Field(64, description="Hidden dimension size")
    num_layers: int = Field(2, description="Number of GRU layers")
    output_dim: int = Field(1, description="Output dimension")
    dropout_prob: float = Field(0.3, description="Dropout probability")
    bidirectional: bool = Field(False, description="Use bidirectional GRU")
    
    # Training (similar to LSTM but simpler loss options)
    epochs: int = Field(100, description="Maximum training epochs")
    batch_size: int = Field(32, description="Training batch size")
    learning_rate: float = Field(0.001, description="Initial learning rate")
    patience: int = Field(10, description="Early stopping patience")
    weight_decay: float = Field(1e-5, description="L2 regularization weight decay")
    
    # Learning rate scheduling
    lr_scheduler_factor: float = Field(0.5, description="LR reduction factor")
    min_lr: float = Field(1e-6, description="Minimum learning rate")
    clip_grad_norm: Optional[float] = Field(1.0, description="Max gradient norm for clipping")
    
    # Scheduler options
    scheduler_type: str = Field("plateau", description="Scheduler type: 'plateau' or 'cosine'")
    t_max_cosine: Optional[int] = Field(None, description="T_max for CosineAnnealingLR")
    
    # Loss function (simpler than LSTM)
    loss_type: str = Field("mse", description="Loss type: 'mse' or 'mae'")
    
    # Evaluation
    eval_batch_size: int = Field(256, description="Batch size for evaluation")
    mape_epsilon: float = Field(1e-8, description="Epsilon for MAPE calculation")
    
    @validator('dropout_prob')
    def validate_dropout(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Dropout probability must be between 0 and 1")
        return v


class TDMCConfig(BaseModel):
    """Configuration for Time-Dynamic Markov Chain model."""
    
    # Model architecture
    n_states: int = Field(4, description="Number of hidden states")
    n_emissions: int = Field(2, description="Number of observable emission variables")
    time_slices: int = Field(24, description="Number of time slices per day")
    
    # Training
    max_iter: int = Field(100, description="Maximum EM iterations")
    tolerance: float = Field(1e-4, description="Convergence tolerance")
    random_state: int = Field(42, description="Random state for reproducibility")
    
    # Regularization
    covariance_regularization: float = Field(1e-4, description="Covariance matrix regularization")
    min_eigenvalue_threshold: float = Field(1e-9, description="Minimum eigenvalue for positive definiteness")
    transition_smoothing_prior: float = Field(1e-6, description="Smoothing prior for transitions")
    initial_state_smoothing_prior: float = Field(1e-6, description="Smoothing prior for initial states")
    
    # Numerical stability
    eigenvalue_min_tolerance: float = Field(1e-9, description="Minimum eigenvalue tolerance")
    min_probability: float = Field(1e-300, description="Minimum probability to avoid numerical issues")
    probability_floor: float = Field(1e-300, description="Floor value for probabilities to avoid underflow")
    
    # Logging
    verbose_logging: bool = Field(False, description="Enable verbose logging for debugging")
    log_likelihood_every_n_iter: int = Field(10, description="Log likelihood every N iterations")
    
    # K-means initialization
    kmeans_n_init: str = Field("auto", description="Number of K-means initializations")


class ModelsConfig(BaseModel):
    """Configuration for all models."""
    
    lstm: LSTMConfig = Field(default_factory=LSTMConfig)
    gru: GRUConfig = Field(default_factory=GRUConfig) 
    tdmc: TDMCConfig = Field(default_factory=TDMCConfig)


# =============================================================================
# File and Path Configuration
# =============================================================================

class PathsConfig(BaseModel):
    """Configuration for file paths and directories."""
    
    # Data directories
    data_dir: str = Field("data", description="Main data directory")
    sample_data_dir: str = Field("data/sample", description="Sample data directory")
    models_dir: str = Field("models", description="Saved models directory")
    logs_dir: str = Field("logs", description="Logs directory")
    plots_dir: str = Field("plots", description="Plots output directory")
    
    # Data files
    #full_dataset_filename: str = Field("SolarPrediction.csv", description="Full dataset filename")
    sample_dataset_filename: str = Field("SolarPrediction_sample.csv", description="Sample dataset filename")
    full_dataset_filename: str = Field("solar_weather.csv", description="Weather dataset filename")
    
    # Model files
    lstm_model_filename: str = Field("lstm_model.pth", description="LSTM model filename")
    gru_model_filename: str = Field("gru_model.pth", description="GRU model filename")
    tdmc_model_filename: str = Field("tdmc_model.pkl", description="TDMC model filename")
    
    # Configuration files
    config_filename: str = Field("config.yaml", description="Configuration file name")
    
    def get_absolute_path(self, relative_path: str, base_path: Optional[Path] = None) -> Path:
        """Convert relative path to absolute path."""
        if base_path is None:
            base_path = Path.cwd()
        return (base_path / relative_path).resolve()


# =============================================================================
# Logging Configuration
# =============================================================================

class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: str = Field("INFO", description="Logging level")
    format_string: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    # File logging
    log_to_file: bool = Field(True, description="Enable file logging")
    log_filename: str = Field("solar_prediction.log", description="Log filename")
    max_log_size_mb: int = Field(10, description="Maximum log file size in MB")
    backup_count: int = Field(5, description="Number of backup log files")
    
    # Console logging
    log_to_console: bool = Field(True, description="Enable console logging")


# =============================================================================
# Plotting Configuration  
# =============================================================================

class PlottingConfig(BaseModel):
    """Configuration for plotting and visualization."""
    
    # Figure settings
    default_figsize: tuple = Field((12, 8), description="Default figure size")
    dashboard_width: int = Field(15, description="Dashboard figure width")
    dashboard_height: int = Field(15, description="Dashboard figure height")
    dpi: int = Field(100, description="Figure DPI")
    
    # Plot styling
    style: str = Field("seaborn-v0_8", description="Matplotlib style")
    color_palette: List[str] = Field(
        default_factory=lambda: ["blue", "red", "green", "orange", "purple", "brown"],
        description="Default color palette"
    )
    
    # Time series specific
    resample_frequency: str = Field("1min", description="Default resampling frequency")
    max_points_scatter: int = Field(10000, description="Maximum points for scatter plots")
    
    # Metrics and annotations
    show_metrics_table: bool = Field(True, description="Show metrics table in plots")
    outlier_std_threshold: float = Field(2.0, description="Standard deviation threshold for outlier highlighting")
    
    # File output
    save_plots: bool = Field(False, description="Save plots to files")
    plot_format: str = Field("png", description="Plot file format")
    plot_quality: int = Field(300, description="Plot DPI for saved files")


# =============================================================================
# Evaluation Configuration
# =============================================================================

class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""
    
    # Metrics calculation
    mape_epsilon: float = Field(1e-8, description="Epsilon for MAPE calculation")
    mape_clip_value: float = Field(1.0, description="Maximum MAPE value for clipping")
    
    # Cross-validation
    cv_folds: int = Field(5, description="Number of cross-validation folds")
    time_series_cv: bool = Field(True, description="Use time series cross-validation")
    
    # Uncertainty quantification
    uncertainty_alpha: float = Field(0.05, description="Alpha for confidence intervals")
    monte_carlo_samples: int = Field(100, description="Number of Monte Carlo samples")
    
    # Evaluation modes
    return_predictions: bool = Field(True, description="Return predictions during evaluation")
    memory_efficient: bool = Field(True, description="Use memory-efficient evaluation")
    batch_size_eval: int = Field(512, description="Batch size for evaluation")


# =============================================================================
# Main Configuration Class
# =============================================================================

class SolarPredictionConfig(BaseModel):
    """Main configuration class that combines all configuration sections."""
    
    # Configuration metadata
    version: str = Field("1.0.0", description="Configuration version")
    description: str = Field(
        "Solar Prediction Project Configuration",
        description="Configuration description"
    )
    
    # Configuration sections
    data: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    transformation: DataTransformationConfig = Field(default_factory=DataTransformationConfig)
    features: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    sequences: SequenceConfig = Field(default_factory=SequenceConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    plotting: PlottingConfig = Field(default_factory=PlottingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# =============================================================================
# Utility Functions
# =============================================================================

# Global configuration instance
_config_instance: Optional[SolarPredictionConfig] = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> SolarPredictionConfig:
    """
    Get the global configuration instance.
    
    Args:
        config_path: Optional path to configuration file. If provided, loads from file.
        
    Returns:
        SolarPredictionConfig: The configuration instance
    """
    global _config_instance
    
    if _config_instance is None or config_path is not None:
        if config_path is not None:
            _config_instance = load_config_from_file(config_path)
        else:
            _config_instance = SolarPredictionConfig()
    
    return _config_instance


def load_config_from_file(config_path: Union[str, Path]) -> SolarPredictionConfig:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        SolarPredictionConfig: Loaded configuration
    """
    import yaml
    import json
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return SolarPredictionConfig(**config_data)


def save_config_to_file(config: SolarPredictionConfig, config_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML or JSON file.
    
    Args:
        config: Configuration instance to save
        config_path: Path where to save the configuration
    """
    import yaml
    import json
    
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.dict()
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yml', '.yaml']:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def reset_config() -> None:
    """Reset the global configuration instance to defaults."""
    global _config_instance
    _config_instance = None


def update_config(**kwargs) -> SolarPredictionConfig:
    """
    Update specific configuration values.
    
    Args:
        **kwargs: Configuration values to update using dot notation
                  (e.g., models__lstm__hidden_dim=128)
        
    Returns:
        SolarPredictionConfig: Updated configuration
    """
    config = get_config()
    
    # Convert double underscore notation to nested dict updates
    for key, value in kwargs.items():
        keys = key.split('__')
        current = config
        
        for k in keys[:-1]:
            current = getattr(current, k)
        
        setattr(current, keys[-1], value)
    
    return config


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config(config: Optional[SolarPredictionConfig] = None) -> bool:
    """
    Validate the configuration for consistency and correctness.
    
    Args:
        config: Configuration to validate. If None, uses global config.
        
    Returns:
        bool: True if configuration is valid
    """
    if config is None:
        config = get_config()
    
    try:
        # Validate that paths exist or can be created
        paths = config.paths
        
        # Validate model configurations
        models = config.models
        
        # Validate data processing parameters
        data = config.data
        if data.min_radiation_clip >= data.max_radiation_clip:
            raise ValueError("min_radiation_clip must be less than max_radiation_clip")
        
        # Validate transformation parameters
        transform = config.transformation
        if transform.use_log_transform and transform.use_power_transform:
            raise ValueError("Cannot use both log and power transforms simultaneously")
        
        # Validate sequence parameters
        seq = config.sequences
        if seq.test_size + seq.val_size_from_train_val >= 1.0:
            raise ValueError("test_size + val_size_from_train_val must be < 1.0")
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


# =============================================================================
# Environment-based Configuration
# =============================================================================

def get_config_for_environment(env: str = "development") -> SolarPredictionConfig:
    """
    Get configuration for a specific environment.
    
    Args:
        env: Environment name ("development", "production", "testing")
        
    Returns:
        SolarPredictionConfig: Environment-specific configuration
    """
    config = get_config()
    
    if env == "development":
        # Development-specific overrides
        config.logging.level = "DEBUG"
        config.plotting.save_plots = True
        config.models.lstm.epochs = 10  # Faster training for development
        config.models.gru.epochs = 10
        
    elif env == "production":
        # Production-specific overrides
        config.logging.level = "INFO"
        config.plotting.save_plots = False
        config.logging.log_to_console = False
        
    elif env == "testing":
        # Testing-specific overrides
        config.logging.level = "ERROR"
        config.plotting.save_plots = False
        config.models.lstm.epochs = 2  # Very fast for testing
        config.models.gru.epochs = 2
        config.sequences.window_size = 5  # Smaller sequences for testing
        
    return config


if __name__ == "__main__":
    # Example usage
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"LSTM hidden dimension: {config.models.lstm.hidden_dim}")
    print(f"Data min radiation clip: {config.data.min_radiation_clip}")
    
    # Validate configuration
    is_valid = validate_config(config)
    print(f"Configuration is valid: {is_valid}")
