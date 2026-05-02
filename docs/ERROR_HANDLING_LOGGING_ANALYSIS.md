# Error-Handling and Logging Patterns Analysis

## Executive Summary

This analysis compares SARIMA's ad-hoc `print` statements with the structured `logging` patterns used elsewhere in the codebase, identifies unhandled exceptions, and proposes standardized logging and error handling improvements.

## Current State Analysis

### 1. SARIMA Module Logging Patterns

**Current Approach**: Mixed `print` statements and minimal structured logging

#### Print Statements in SARIMA (sarima.py)
```python
# Examples of print-based logging
print(f"Missing values before cleaning: {series.isnull().sum()}")
print(f"Missing values after cleaning: {series.isnull().sum()}")
print("Successfully set frequency to 15T (15-minute intervals)")
print(f"Could not set frequency automatically: {e}")
print(f"Detected seasonality {s} may be incorrect for 15-minute data")
print(f"Data prepared successfully:")
print(f"   Total samples: {len(series)}")
print(f"SARIMA model fitted successfully")
print(f"   AIC: {results.aic:.2f}")
print(f"Error fitting SARIMA model: {e}")
print(f"Running automatic SARIMA model selection (m={seasonal_period})...")
print(f"🏆 Best model found by auto_arima: SARIMA{auto_model.order} × {auto_model.seasonal_order}")
print(f"❌ Auto ARIMA failed: {e}")
```

**Limited Structured Logging**:
```python
# Only 2 instances of proper logging in SARIMA
logging.warning("Could not infer frequency. Assuming hourly data (s=24)")
logging.warning(f"Unknown frequency {freq_str}. Defaulting to s=24")
```

### 2. Structured Logging in Other Modules

**Consistent Pattern**: Proper logging setup with hierarchical levels

#### Data Preparation Module (data_prep.py)
```python
# Proper logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Structured usage
logging.info("Starting weather data preparation pipeline v2.")
logging.info(f"Applied column renames: {actual_rename_map}")
logging.warning(f"Could not parse or sort by '{STD_TIME_COL}': {e}. Trying UNIXTime.")
logging.debug("Performing initial DataFrame setup.")
```

#### GRU Model (gru.py)
```python
# Structured error handling with context
logging.error(f"Input validation failed for GRU fit: {e}")
logging.info(f"GRU Training started. Device: {device}, Config: {train_config}")
logging.warning(f"Unknown loss type '{config.loss_type}' for GRU. Defaulting to MSE.")
```

#### Memory Tracker (memory_tracker.py)
```python
# Logger configuration with formatting
self.logger = logging.getLogger(f'MemoryTracker.{self.device}')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
self.logger.debug(f"Memory snapshot '{name}': {snapshot.allocated:.1f}MB allocated")
self.logger.warning(f"Cannot find snapshots for memory diff: {before_name}, {after_name}")
```

## Identified Problems

### 1. Unhandled Exceptions in SARIMA

#### Silent `return None` Patterns
```python
# auto_sarima_selection function
except Exception as e:
    print(f"❌ Auto ARIMA failed: {e}")
    print("Falling back to manual selection with conservative parameters...")
    return fit_sarima_model(series, order=(1,1,1), seasonal_order=(1,1,1,seasonal_period))

# fit_sarima_model function  
except Exception as e:
    print(f"Error fitting SARIMA model: {e}")
    return None  # Silent failure!

# fit_ets_model function
except Exception as e:
    print(f"Error fitting ETS model: {e}")
    return None  # Silent failure!
```

#### Unspecific Exception Handling
```python
# Broad exception catching without proper classification
try:
    series_with_freq = series.asfreq('15T')
    print("Successfully set frequency to 15T (15-minute intervals)")
    series = series_with_freq.ffill().bfill()
except Exception as e:  # Too broad!
    print(f"Could not set frequency automatically: {e}")
    print("   Proceeding with original series - will manually set seasonality")
```

### 2. Missing Error Context

#### Insufficient Error Information
```python
# Current pattern - lacks context
print(f"Error fitting SARIMA model: {e}")

# Better pattern would include:
# - Model parameters being fitted
# - Data characteristics (size, range, etc.)
# - Suggested remediation steps
# - Error classification
```

### 3. Inconsistent Error Reporting

**SARIMA**: Mixed print/logging, no structured error levels
**Other modules**: Consistent logging levels with proper formatting

## Proposed Standardized Approach

### 1. Logging Configuration

#### Centralized Logger Setup
```python
import logging
from typing import Optional

def setup_sarima_logger(level: str = "INFO") -> logging.Logger:
    """Setup standardized logger for SARIMA module."""
    logger = logging.getLogger('solar_prediction.sarima')
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(getattr(logging, level.upper()))
    return logger

# Module-level logger
logger = setup_sarima_logger()
```

### 2. Standardized Logging Levels

#### INFO Level - Progress and Success
```python
# Replace print statements for progress
logger.info(f"Starting SARIMA data preparation for {len(series)} samples")
logger.info(f"Successfully fitted SARIMA{order} × {seasonal_order} (AIC: {results.aic:.2f})")
logger.info(f"Auto-ARIMA found best model: SARIMA{auto_model.order} × {auto_model.seasonal_order}")
```

#### WARNING Level - Non-fatal Issues
```python
# Replace print statements for warnings
logger.warning(f"Could not infer frequency, assuming hourly data (s=24)")
logger.warning(f"High missing data percentage: {missing_pct:.1%}, applying forward/backward fill")
logger.warning(f"Model convergence may be poor due to near-constant data (variance: {variance:.6f})")
```

#### ERROR Level - Recoverable Failures
```python
# Replace print statements for errors
logger.error(f"SARIMA model fitting failed: {e}. Parameters: order={order}, seasonal_order={seasonal_order}")
logger.error(f"Auto-ARIMA selection failed after {max_attempts} attempts: {e}")
logger.error(f"ETS model fitting failed with configuration {config}: {e}")
```

#### DEBUG Level - Detailed Information
```python
# Add debug information for troubleshooting
logger.debug(f"Input data characteristics: mean={series.mean():.2f}, std={series.std():.2f}, skew={series.skew():.2f}")
logger.debug(f"Stationarity test results: ADF p-value={adf_pvalue:.6f}, KPSS p-value={kpss_pvalue:.6f}")
logger.debug(f"Model search space: max_p={max_p}, max_q={max_q}, max_P={max_P}, max_Q={max_Q}")
```

### 3. Structured Error Handling

#### Error Classification System
```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

class SARIMAErrorType(Enum):
    """Classification of SARIMA-specific errors."""
    DATA_INSUFFICIENT = "insufficient_data"
    DATA_INVALID = "invalid_data"
    CONVERGENCE_FAILURE = "convergence_failure"
    NUMERICAL_INSTABILITY = "numerical_instability"
    PARAMETER_INVALID = "invalid_parameters"
    DEPENDENCY_MISSING = "missing_dependency"

@dataclass
class SARIMAError(Exception):
    """Structured SARIMA error with context."""
    error_type: SARIMAErrorType
    message: str
    context: Optional[Dict[str, Any]] = None
    original_exception: Optional[Exception] = None
    suggested_action: Optional[str] = None
    
    def __str__(self) -> str:
        base_msg = f"[{self.error_type.value}] {self.message}"
        if self.suggested_action:
            base_msg += f" Suggestion: {self.suggested_action}"
        return base_msg
```

#### Improved Exception Handling
```python
def fit_sarima_model_improved(series, order=(1,1,1), seasonal_order=(1,1,1,24), 
                             enforce_stationarity=False, enforce_invertibility=False):
    """Improved SARIMA model fitting with structured error handling."""
    
    logger.info(f"Fitting SARIMA{order} × {seasonal_order}")
    logger.debug(f"Data characteristics: n={len(series)}, mean={series.mean():.2f}, std={series.std():.2f}")
    
    # Validate inputs
    if len(series) < max(order) + max(seasonal_order[:3]) * seasonal_order[3] + 10:
        raise SARIMAError(
            error_type=SARIMAErrorType.DATA_INSUFFICIENT,
            message=f"Insufficient data for model complexity: {len(series)} samples",
            context={"series_length": len(series), "required_minimum": max(order) + max(seasonal_order[:3]) * seasonal_order[3] + 10},
            suggested_action="Reduce model complexity or provide more data"
        )
    
    if series.var() < 1e-10:
        raise SARIMAError(
            error_type=SARIMAErrorType.DATA_INVALID,
            message=f"Near-constant data (variance: {series.var():.2e})",
            context={"variance": series.var(), "mean": series.mean()},
            suggested_action="Check data preprocessing or add noise regularization"
        )
    
    try:
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility
        )
        
        results = model.fit(disp=False, maxiter=200, method='lbfgs')
        
        logger.info(f"SARIMA model fitted successfully (AIC: {results.aic:.2f}, BIC: {results.bic:.2f})")
        return results
        
    except np.linalg.LinAlgError as e:
        raise SARIMAError(
            error_type=SARIMAErrorType.NUMERICAL_INSTABILITY,
            message="Matrix operation failed during fitting",
            context={"order": order, "seasonal_order": seasonal_order},
            original_exception=e,
            suggested_action="Try enforce_stationarity=True or reduce model complexity"
        ) from e
        
    except ValueError as e:
        if "invertible" in str(e).lower():
            raise SARIMAError(
                error_type=SARIMAErrorType.CONVERGENCE_FAILURE,
                message="Model parameters not invertible",
                context={"order": order, "seasonal_order": seasonal_order},
                original_exception=e,
                suggested_action="Try enforce_invertibility=True or different parameters"
            ) from e
        else:
            raise SARIMAError(
                error_type=SARIMAErrorType.PARAMETER_INVALID,
                message=f"Invalid model parameters: {e}",
                context={"order": order, "seasonal_order": seasonal_order},
                original_exception=e,
                suggested_action="Validate parameter ranges and data compatibility"
            ) from e
            
    except Exception as e:
        logger.error(f"Unexpected error in SARIMA fitting: {e}")
        raise SARIMAError(
            error_type=SARIMAErrorType.CONVERGENCE_FAILURE,
            message=f"Model fitting failed: {e}",
            context={"order": order, "seasonal_order": seasonal_order},
            original_exception=e,
            suggested_action="Try simpler model or different optimization method"
        ) from e
```

### 4. Error Recovery Strategies

#### Graceful Degradation
```python
def auto_sarima_with_fallback(series: pd.Series, seasonal_period: int) -> Optional[Any]:
    """Auto-ARIMA with structured fallback strategy."""
    
    fallback_configs = [
        {"max_p": 2, "max_q": 2, "max_P": 1, "max_Q": 1},  # Moderate complexity
        {"max_p": 1, "max_q": 1, "max_P": 1, "max_Q": 1},  # Conservative
        {"max_p": 1, "max_q": 0, "max_P": 0, "max_Q": 0},  # Minimal AR
        {"max_p": 0, "max_q": 1, "max_P": 0, "max_Q": 0},  # Minimal MA
    ]
    
    for i, config in enumerate(fallback_configs):
        try:
            logger.info(f"Attempting auto-ARIMA with configuration {i+1}/{len(fallback_configs)}: {config}")
            
            result = auto_arima_selection(series, seasonal_period, **config)
            if result is not None:
                logger.info(f"Auto-ARIMA succeeded with fallback configuration {i+1}")
                return result
                
        except SARIMAError as e:
            logger.warning(f"Auto-ARIMA configuration {i+1} failed: {e}")
            if i == len(fallback_configs) - 1:  # Last attempt
                logger.error(f"All auto-ARIMA configurations failed. Final error: {e}")
                raise e
            else:
                logger.info(f"Trying next fallback configuration...")
                continue
                
    return None
```

### 5. Diagnostic Information

#### Enhanced Error Context
```python
def add_diagnostic_context(series: pd.Series, error: SARIMAError) -> SARIMAError:
    """Add diagnostic information to SARIMA errors."""
    
    diagnostics = {
        "data_summary": {
            "length": len(series),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "missing_count": int(series.isnull().sum()),
            "zero_count": int((series == 0).sum()),
            "negative_count": int((series < 0).sum())
        },
        "stationarity_hints": {
            "constant_variance": series.std() > 1e-6,
            "reasonable_range": series.min() >= 0 and series.max() < 10000,
            "no_extreme_outliers": (series.quantile(0.99) - series.quantile(0.01)) < 1000
        }
    }
    
    error.context = {**(error.context or {}), **diagnostics}
    return error
```

## Implementation Recommendations

### Phase 1: Replace Print Statements
1. **Immediate**: Replace all `print()` calls with appropriate `logger.info()`, `logger.warning()`, or `logger.error()` calls
2. **Add context**: Include relevant parameters and data characteristics in log messages
3. **Consistent formatting**: Use structured message formats with consistent parameter naming

### Phase 2: Implement Structured Errors
1. **Define error types**: Create `SARIMAErrorType` enum and `SARIMAError` class
2. **Replace generic exceptions**: Convert broad `except Exception` blocks to specific error types
3. **Add error context**: Include diagnostic information and suggested actions

### Phase 3: Error Recovery
1. **Implement fallback strategies**: Add graceful degradation for auto-ARIMA failures
2. **Surface errors appropriately**: Replace `return None` with proper exception propagation where appropriate
3. **Add retry logic**: Implement retry mechanisms for transient failures

### Phase 4: Integration with Monitoring
1. **Structured metrics**: Log key metrics (AIC, convergence status, etc.) in structured format
2. **Performance tracking**: Add timing and memory usage logging
3. **Error analytics**: Enable error pattern analysis through structured logging

## Benefits of Proposed Changes

1. **Consistency**: Aligns SARIMA logging with rest of codebase
2. **Debuggability**: Structured errors provide clear failure context
3. **Maintainability**: Centralized error handling simplifies debugging
4. **Robustness**: Fallback strategies improve model reliability
5. **Monitoring**: Structured logs enable better operational visibility
6. **User Experience**: Clear error messages with actionable suggestions

## Conclusion

The current SARIMA implementation's reliance on `print` statements and silent `return None` patterns represents a significant deviation from the structured logging and error handling used elsewhere in the codebase. Implementing the proposed standardized approach will improve consistency, debuggability, and robustness while maintaining the existing functionality.

The phased implementation allows for gradual improvement without disrupting current operations, while the structured error system provides clear pathways for diagnosing and resolving issues in production environments.
