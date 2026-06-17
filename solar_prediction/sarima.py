import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import logging

# Statistical analysis
try:
    import statsmodels.api as sm
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, kpss

    from statsmodels.tsa.api import ExponentialSmoothing, Holt, SimpleExpSmoothing
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox

    STATSMODELS_AVAILABLE = True
except ImportError:
    sm = None
    plot_acf = plot_pacf = None
    SARIMAX = None
    adfuller = kpss = None
    ExponentialSmoothing = Holt = SimpleExpSmoothing = None
    ETSModel = None
    seasonal_decompose = None
    acorr_ljungbox = None
    STATSMODELS_AVAILABLE = False

# Machine learning metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from pmdarima import auto_arima

    AUTO_ARIMA_AVAILABLE = True
except ImportError:
    auto_arima = None
    AUTO_ARIMA_AVAILABLE = False


def _require_statsmodels():
    if not STATSMODELS_AVAILABLE:
        raise ImportError(
            "SARIMA/ETS functionality requires optional classical dependencies. "
            "Install them with: pip install -e '.[classical]'"
        )


def determine_seasonality(series, freq_hint=None):
    """Determine seasonality period based on data frequency"""
    if freq_hint:
        freq_str = freq_hint
    else:
        freq_str = pd.infer_freq(series.index)

    if freq_str is None:
        logging.warning("Could not infer frequency. Assuming hourly data (s=24)")
        return 24

    freq_mapping = {
        "15T": 96,  # 15-minute: 4 * 24 = 96
        "15min": 96,
        "H": 24,  # Hourly: 24
        "1H": 24,
        "D": 1,  # Daily: weekly seasonality
        "1D": 1,
    }

    for pattern, s_value in freq_mapping.items():
        if pattern in str(freq_str):
            return s_value

    logging.warning(f"Unknown frequency {freq_str}. Defaulting to s=24")
    return 24


def prepare_time_series_data(df, time_col="Time", target_col="GHI", test_size=0.2):
    """Prepare data for classical time series analysis"""
    df_work = df.copy()

    # Convert time column to datetime
    if time_col in df_work.columns:
        df_work[time_col] = pd.to_datetime(df_work[time_col])
        df_work = df_work.set_index(time_col)
    elif not isinstance(df_work.index, pd.DatetimeIndex):
        raise ValueError(f"No '{time_col}' column found and index is not datetime")

    # Extract target series
    if target_col not in df_work.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    series = df_work[target_col].copy()

    # Handle missing values
    print(f"Missing values before cleaning: {series.isnull().sum()}")
    series = series.ffill().bfill()
    print(f"Missing values after cleaning: {series.isnull().sum()}")

    # Remove any remaining NaN values
    series = series.dropna()

    # CRITICAL: Set frequency for 15-minute data to enable proper seasonality detection
    try:
        # Attempt to set frequency if data is regular 15-minute intervals
        series_with_freq = series.asfreq("15T")
        print("Successfully set frequency to 15T (15-minute intervals)")
        series = series_with_freq.ffill().bfill()  # Handle any NaNs introduced by asfreq
    except Exception as e:
        print(f"Could not set frequency automatically: {e}")
        print("   Proceeding with original series - will manually set seasonality")

    # Determine seasonality
    s = determine_seasonality(series)

    # MANUAL OVERRIDE for 15-minute data (most reliable approach)
    if s != 96:
        print(f"Detected seasonality {s} may be incorrect for 15-minute data")
        s = 96  # Force correct seasonality for 15-minute data
        print(f"Manually overriding seasonality to {s} (daily cycle for 15-min data)")

    # Train-test split
    split_point = int(len(series) * (1 - test_size))
    train_series = series[:split_point]
    test_series = series[split_point:]

    print(f"Data prepared successfully:")
    print(f"   Total samples: {len(series)}")
    print(f"   Training samples: {len(train_series)}")
    print(f"   Test samples: {len(test_series)}")
    print(f"   Seasonality period: {s}")
    print(f"   Data frequency: {series.index.freq or 'Not detected'}")

    return train_series, test_series, s


def check_stationarity(series, name="Series", alpha=0.05):
    """Comprehensive stationarity testing"""
    _require_statsmodels()
    print(f"\nStationarity Analysis for {name}")
    print("=" * 50)

    # Augmented Dickey-Fuller test
    adf_result = adfuller(series.dropna())
    print(f"ADF Test:")
    print(f"  Statistic: {adf_result[0]:.6f}")
    print(f"  p-value: {adf_result[1]:.6f}")
    print(f"  Critical Values: {adf_result[4]}")

    adf_stationary = adf_result[1] <= alpha
    print(f"  Result: {'Stationary' if adf_stationary else 'Non-stationary'}")

    # KPSS test
    try:
        kpss_result = kpss(series.dropna(), regression="c")
        print(f"\nKPSS Test:")
        print(f"  Statistic: {kpss_result[0]:.6f}")
        print(f"  p-value: {kpss_result[1]:.6f}")
        print(f"  Critical Values: {kpss_result[3]}")

        kpss_stationary = kpss_result[1] > alpha
        print(f"  Result: {'Stationary' if kpss_stationary else 'Non-stationary'}")

        # Consensus
        if adf_stationary and kpss_stationary:
            consensus = "Stationary"
        elif not adf_stationary and not kpss_stationary:
            consensus = "Non-stationary"
        else:
            consensus = "Inconclusive"

        print(f"\nConsensus: {consensus}")

    except Exception as e:
        print(f"KPSS test failed: {e}")
        consensus = "Stationary" if adf_stationary else "Non-stationary"
        print(f"Based on ADF only: {consensus}")

    return adf_stationary


def plot_acf_pacf_analysis(series, lags=None, title="ACF/PACF Analysis", seasonality=24):
    """Plot ACF and PACF with interpretation guidance"""
    _require_statsmodels()
    if lags is None:
        lags = min(len(series) // 4, seasonality * 2, 50)

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # ACF
    plot_acf(series.dropna(), ax=axes[0], lags=lags, alpha=0.05)
    axes[0].set_title(f"{title} - Autocorrelation Function (ACF)")
    axes[0].grid(True, alpha=0.3)

    # PACF
    plot_pacf(series.dropna(), ax=axes[1], lags=lags, alpha=0.05)
    axes[1].set_title(f"{title} - Partial Autocorrelation Function (PACF)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Interpretation guidance
    print(f"\nACF/PACF Interpretation Guide:")
    print("AR(p): PACF cuts off after lag p, ACF decays slowly")
    print("MA(q): ACF cuts off after lag q, PACF decays slowly")
    print("ARMA(p,q): Both ACF and PACF decay slowly")
    print(f"Seasonal patterns repeat every {seasonality} lags")


def fit_sarima_model(
    series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False,
):
    """Fit SARIMA model with comprehensive error handling"""
    _require_statsmodels()

    print(f"\nFitting SARIMA{order} × {seasonal_order}...")

    try:
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )

        results = model.fit(disp=False, maxiter=200, method="lbfgs")

        print(f"SARIMA model fitted successfully")
        print(f"   AIC: {results.aic:.2f}")
        print(f"   BIC: {results.bic:.2f}")
        print(f"   Log-likelihood: {results.llf:.2f}")

        return results

    except Exception as e:
        print(f"Error fitting SARIMA model: {e}")
        return None


def auto_sarima_selection(
    series: pd.Series,
    seasonal_period: int,
    max_p_val: int = 1,
    max_q_val: int = 1,  # Further reduced for speed
    max_P_val: int = 1,
    max_Q_val: int = 1,  # Keep at 1
    max_d_val: int = 1,
    max_D_val: int = 1,
):  # Reduced d/D for stability
    """
    Automatic SARIMA model selection using pmdarima with conservative parameters for stability.
    """
    _require_statsmodels()
    if not AUTO_ARIMA_AVAILABLE:
        print("pmdarima not available. Using default parameters for fallback.")
        # Fallback uses order=(1,1,1) and seasonal_order=(1,1,1,seasonal_period)
        return fit_sarima_model(series, seasonal_order=(1, 1, 1, seasonal_period))

    print(f"Running automatic SARIMA model selection (m={seasonal_period})...")
    print(
        f"   Search space: max_p={max_p_val}, max_q={max_q_val}, max_P={max_P_val}, max_Q={max_Q_val}, max_d={max_d_val}, max_D={max_D_val}"
    )
    print("   Using conservative parameters to prevent kernel crashes...")

    try:
        # Use a smaller subset of data for initial model selection to speed up the process
        subset_size = min(len(series), 2000)  # Use at most 2000 points for model selection
        if len(series) > subset_size:
            print(
                f"   Using subset of {subset_size} points for model selection (full dataset: {len(series)})"
            )
            series_subset = series.iloc[-subset_size:]
        else:
            series_subset = series

        auto_model = auto_arima(
            series_subset,
            seasonal=True,
            m=seasonal_period,
            max_p=max_p_val,
            max_q=max_q_val,
            max_P=max_P_val,
            max_Q=max_Q_val,
            max_d=max_d_val,
            max_D=max_D_val,
            start_p=0,
            start_q=0,
            start_P=0,
            start_Q=0,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=True,
            maxiter=50,
            seasonal_test="ch",
            approximation=True,
            method="lbfgs",
            random_state=42,
        )

        print(
            f"🏆 Best model found by auto_arima: SARIMA{auto_model.order} × {auto_model.seasonal_order}"
        )
        print(f"   AIC: {auto_model.aic():.2f}")

        # If we used a subset, refit the model on the full dataset
        if len(series) > subset_size:
            print("   Refitting best model on full dataset...")
            full_model = fit_sarima_model(
                series, order=auto_model.order, seasonal_order=auto_model.seasonal_order
            )
            if full_model is not None:
                print(f"   Full dataset AIC: {full_model.aic:.2f}")
                return full_model
            else:
                print("   Failed to refit on full dataset, returning subset model")
                return auto_model
        else:
            return auto_model

    except Exception as e:
        print(f"❌ Auto ARIMA failed: {e}")
        print("Falling back to manual selection with conservative parameters...")
        # Try a simple, stable model as fallback
        return fit_sarima_model(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period))


def perform_residual_diagnostics(model_results, model_name, plot_size=(15, 12)):
    """Comprehensive residual analysis"""
    _require_statsmodels()

    print(f"\nResidual Diagnostics for {model_name}")
    print("=" * 50)

    try:
        # Plot diagnostics
        fig = model_results.plot_diagnostics(figsize=plot_size)
        plt.suptitle(f"{model_name} Residual Diagnostics", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

        # Extract residuals
        residuals = model_results.resid.dropna()

        # Ljung-Box test for autocorrelation
        lags_to_test = [10, 20, min(len(residuals) // 4, 40)]
        lb_results = acorr_ljungbox(residuals, lags=lags_to_test, return_df=True)

        print("Ljung-Box Test Results (testing for autocorrelation):")
        print(lb_results)
        print("Good: p-values > 0.05 (no significant autocorrelation)")
        print("Concerning: p-values ≤ 0.05 (autocorrelation present)")

        # Normality test
        from scipy import stats

        _, p_value_normality = stats.jarque_bera(residuals)
        print(f"\nJarque-Bera Normality Test:")
        print(f"p-value: {p_value_normality:.6f}")
        if p_value_normality > 0.05:
            print("Residuals appear normally distributed")
        else:
            print("Residuals may not be normally distributed")

        # Summary statistics
        print(f"\nResidual Summary Statistics:")
        print(f"Mean: {residuals.mean():.6f} (should be close to 0)")
        print(f"Std: {residuals.std():.6f}")
        print(f"Skewness: {residuals.skew():.6f}")
        print(f"Kurtosis: {residuals.kurtosis():.6f}")

        return True

    except Exception as e:
        print(f"Error in diagnostics: {e}")
        return False


def fit_ets_model(
    series, seasonal_periods, error="add", trend="add", seasonal="add", damped_trend=True
):
    """Fit ETS model with error handling"""
    _require_statsmodels()

    # Handle None values for trend and seasonal parameters
    # Keep None as None for ETSModel, but handle for string building
    trend_for_string = trend if trend is not None else "none"
    seasonal_for_string = seasonal if seasonal is not None else "none"

    # Build model string safely
    error_char = error[0].upper() if error else "A"
    trend_char = trend_for_string[0].upper() if trend_for_string else "N"
    seasonal_char = seasonal_for_string[0].upper() if seasonal_for_string else "N"

    model_string = f"ETS({error_char},{trend_char}"
    if seasonal_for_string != "none":
        model_string += f",{seasonal_char})"
    else:
        model_string += ",N)"
    if damped_trend and trend_for_string != "none":
        model_string = model_string.replace(")", "d)")

    print(f"\nFitting {model_string} model...")

    try:
        # Ensure series has proper frequency
        if series.index.freq is None:
            freq = pd.infer_freq(series.index)
            if freq:
                series = series.asfreq(freq)
            else:
                print("Could not infer frequency, using original series")

        model = ETSModel(
            series,
            error=error,
            trend=trend,  # Pass None as None to ETSModel
            seasonal=seasonal,  # Pass None as None to ETSModel
            seasonal_periods=seasonal_periods,
            damped_trend=damped_trend,
        )

        results = model.fit(disp=False)

        print(f"ETS model fitted successfully")
        print(f"   AIC: {results.aic:.2f}")
        print(f"   BIC: {results.bic:.2f}")
        print(f"   Log-likelihood: {results.llf:.2f}")

        return results

    except Exception as e:
        print(f"Error fitting ETS model: {e}")
        return None


def select_best_ets_model(series, seasonal_periods):
    """Try different ETS configurations and select the best one"""

    print("ETS Model Selection")
    print("=" * 30)

    # Define model configurations to try
    configurations = [
        ("add", "add", "add", True),  # ETS(A,Ad,A)
        ("add", "add", "add", False),  # ETS(A,A,A)
        ("add", "add", "mul", True),  # ETS(A,Ad,M)
        ("add", "add", "mul", False),  # ETS(A,A,M)
        ("add", None, "add", False),  # ETS(A,N,A)
        ("add", None, "mul", False),  # ETS(A,N,M)
        ("mul", "add", "mul", True),  # ETS(M,Ad,M)
        ("mul", "add", "mul", False),  # ETS(M,A,M)
    ]

    best_aic = float("inf")
    best_model = None
    best_config = None

    for error, trend, seasonal, damped in configurations:
        # Skip multiplicative models if series has zeros or negative values
        if error == "mul" or seasonal == "mul":
            if (series <= 0).any():
                continue

        model = fit_ets_model(series, seasonal_periods, error, trend, seasonal, damped)

        if model is not None and model.aic < best_aic:
            best_aic = model.aic
            best_model = model
            best_config = (error, trend, seasonal, damped)

    if best_model is not None:
        print(f"\nBest ETS Model: {best_config}")
        print(f"   AIC: {best_aic:.2f}")

    return best_model


def create_baseline_models(train_series, test_series, seasonal_period):
    """Create naive baseline models for comparison"""

    print("\nCreating Baseline Models")
    print("=" * 35)

    baselines = {}

    # 1. Naive (Random Walk) - last value carries forward
    naive_pred = pd.Series(
        [train_series.iloc[-1]] * len(test_series), index=test_series.index, name="Naive"
    )
    baselines["Naive"] = naive_pred
    print("Naive model created")

    # 2. Seasonal Naive - same time from previous season
    if len(train_series) >= seasonal_period:
        seasonal_naive_values = []
        for i in range(len(test_series)):
            # Get the value from the same position in the previous season
            seasonal_lag_idx = -(seasonal_period - (i % seasonal_period))
            if abs(seasonal_lag_idx) <= len(train_series):
                seasonal_naive_values.append(train_series.iloc[seasonal_lag_idx])
            else:
                # If not enough history, use the last available value
                seasonal_naive_values.append(train_series.iloc[-1])

        seasonal_naive_pred = pd.Series(
            seasonal_naive_values, index=test_series.index, name="Seasonal_Naive"
        )
        baselines["Seasonal_Naive"] = seasonal_naive_pred
        print("Seasonal Naive model created")

    # 3. Linear Trend - simple linear extrapolation
    try:
        from scipy import stats

        # Fit linear trend to last portion of training data
        trend_window = min(len(train_series), seasonal_period * 4)
        recent_data = train_series[-trend_window:]
        x_trend = np.arange(len(recent_data))
        slope, intercept, _, _, _ = stats.linregress(x_trend, recent_data.values)

        # Project trend forward
        trend_pred_values = []
        for i in range(len(test_series)):
            trend_value = intercept + slope * (len(recent_data) + i)
            trend_pred_values.append(trend_value)

        trend_pred = pd.Series(trend_pred_values, index=test_series.index, name="Linear_Trend")
        baselines["Linear_Trend"] = trend_pred
        print("Linear Trend model created")

    except Exception as e:
        print(f"Could not create Linear Trend model: {e}")

    # 4. Moving Average
    ma_window = min(seasonal_period, len(train_series) // 4)
    ma_value = train_series[-ma_window:].mean()

    ma_pred = pd.Series(
        [ma_value] * len(test_series), index=test_series.index, name="Moving_Average"
    )
    baselines["Moving_Average"] = ma_pred
    print("Moving Average model created")

    return baselines


def calculate_metrics(actual, predicted, model_name, seasonality):
    """Calculate comprehensive evaluation metrics"""

    # Ensure no NaN values
    mask = ~(pd.isna(actual) | pd.isna(predicted))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]

    if len(actual_clean) == 0:
        print(f"No valid data points for {model_name}")
        return None

    # Basic metrics
    mae = mean_absolute_error(actual_clean, predicted_clean)
    rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))

    # R-squared
    r2 = r2_score(actual_clean, predicted_clean)

    # MAPE variants (handling division by zero)
    epsilon = 1e-8

    # Standard MAPE (capped at 100% for extreme outliers)
    mape = (
        np.mean(
            np.clip(
                np.abs((actual_clean - predicted_clean) / (np.abs(actual_clean) + epsilon)), 0, 1.0
            )
        )
        * 100
    )

    # Symmetric MAPE
    smape = (
        np.mean(
            2
            * np.abs(actual_clean - predicted_clean)
            / (np.abs(actual_clean) + np.abs(predicted_clean) + epsilon)
        )
        * 100
    )

    # Mean Absolute Scaled Error (MASE) - requires seasonal naive benchmark
    try:
        if len(actual_clean) > seasonality:
            naive_errors = np.abs(actual_clean[seasonality:] - actual_clean[:-seasonality])
            mae_naive = np.mean(naive_errors)
            mase = mae / (mae_naive + epsilon)
        else:
            mase = np.nan
    except:
        mase = np.nan

    # Additional metrics
    bias = np.mean(predicted_clean - actual_clean)
    max_error = np.max(np.abs(actual_clean - predicted_clean))

    metrics = {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "sMAPE": smape,
        "MASE": mase,
        "R²": r2,
        "Bias": bias,
        "Max_Error": max_error,
        "N_Points": len(actual_clean),
    }

    return metrics

