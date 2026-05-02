# Numerical Stability Safeguards Analysis

## Executive Summary

This document provides a comprehensive analysis of numerical stability safeguards present in LSTM/GRU/TDMC models compared to SARIMA implementation, identifying missing safeguards and recommending improvements.

## 1. Safeguards Checklist: LSTM/GRU/TDMC vs SARIMA

### 1.1 Gradient-Based Safeguards

#### LSTM/GRU Models ✅
| Safeguard | LSTM | GRU | Implementation |
|-----------|------|-----|----------------|
| **Gradient Clipping** | ✅ | ✅ | `torch.nn.utils.clip_grad_norm_(self.parameters(), train_config.clip_grad_norm)` |
| **Gradient Scaling (AMP)** | ✅ | ✅ | `scaler.scale(loss).backward()` with unscaling before clipping |
| **Learning Rate Scheduling** | ✅ | ✅ | ReduceLROnPlateau and CosineAnnealingLR with min_lr floor |
| **Weight Decay Regularization** | ✅ | ✅ | L2 regularization via optimizer |
| **Early Stopping** | ✅ | ✅ | Patience-based stopping with best model restoration |

#### SARIMA Model ❌
- **No gradient-based safeguards** (not applicable - statistical model)

### 1.2 Probability and Division Safeguards

#### LSTM/GRU Models ✅
| Safeguard | LSTM | GRU | Implementation |
|-----------|------|-----|----------------|
| **Division by Zero Protection** | ✅ | ✅ | `np.abs(y_true) + epsilon` in MAPE calculation |
| **Probability Flooring** | ✅ | ✅ | `epsilon` values in loss functions |
| **Overflow Prevention** | ✅ | ✅ | `torch.clamp()` in combined loss |

#### TDMC Model ✅
| Safeguard | Implementation |
|-----------|----------------|
| **Probability Flooring** | `floor_prob()` function with min_prob=1e-300 |
| **Safe Normalization** | `safe_normalise()` with division by zero protection |
| **Log Probability Protection** | Flooring before log operations in Viterbi |

#### SARIMA Model ❌
| Safeguard | Status | Issue |
|-----------|--------|-------|
| **Division by Zero in MAPE** | ❌ | Basic epsilon (1e-8) but no comprehensive protection |
| **Probability Flooring** | ❌ | No probability operations |
| **Overflow Prevention** | ❌ | Relies on statsmodels defaults |

### 1.3 Matrix Numerical Stability

#### TDMC Model ✅
| Safeguard | Implementation |
|-----------|----------------|
| **Eigenvalue Fixing** | `_fix_eigenvalues()` function with regularization |
| **Covariance Regularization** | Diagonal regularization for positive definiteness |
| **Matrix Conditioning** | Automatic offset addition for singular matrices |

#### LSTM/GRU Models ❌
- **No explicit matrix conditioning** (not typically required for feed-forward operations)

#### SARIMA Model ❌
| Safeguard | Status | Issue |
|-----------|--------|-------|
| **Matrix Conditioning** | ❌ | Relies on statsmodels internal handling |
| **Covariance Regularization** | ❌ | No explicit regularization |

### 1.4 Convergence and Optimization Safeguards

#### All Neural Models ✅
| Safeguard | LSTM | GRU | TDMC | Implementation |
|-----------|------|-----|------|----------------|
| **Convergence Monitoring** | ✅ | ✅ | ✅ | Loss tracking and tolerance checks |
| **Maximum Iterations** | ✅ | ✅ | ✅ | Configurable via config |
| **Numerical Precision Control** | ✅ | ✅ | ✅ | Mixed precision (LSTM/GRU), double precision (TDMC) |

#### SARIMA Model ⚠️
| Safeguard | Status | Issue |
|-----------|--------|-------|
| **Convergence Monitoring** | ⚠️ | Basic via statsmodels, no custom tolerance |
| **Maximum Iterations** | ⚠️ | Hard-coded maxiter=200, method='lbfgs' |
| **Method Fallback** | ❌ | No fallback if lbfgs fails |

## 2. SARIMA Implementation Analysis

### 2.1 Stationarity Testing Issues

```python
# Current Implementation (Line 122-157 in sarima.py)
def check_stationarity(series, name="Series", alpha=0.05):
    adf_result = adfuller(series.dropna())  # ❌ No try/except
    
    try:
        kpss_result = kpss(series.dropna(), regression='c')  # ✅ Has try/except
    except Exception as e:
        print(f"KPSS test failed: {e}")  # ⚠️ Basic error handling
```

**Issues Identified:**
1. **ADF test has no try/except protection** - could fail on edge cases
2. **No fallback mechanisms** for test failures
3. **No validation of input series** (length, variance, etc.)
4. **Limited error context** - doesn't indicate what caused failures

### 2.2 Model Fitting Issues

```python
# Current Implementation (Line 190-216)
def fit_sarima_model(series, order=(1,1,1), seasonal_order=(1,1,1,24), 
                     enforce_stationarity=False, enforce_invertibility=False):
    try:
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,  # ❌ Disabled by default
            enforce_invertibility=enforce_invertibility  # ❌ Disabled by default
        )
        
        results = model.fit(disp=False, maxiter=200, method='lbfgs')  # ⚠️ No fallback
```

**Critical Missing Safeguards:**
1. **`enforce_stationarity=False`** - Allows non-stationary models
2. **`enforce_invertibility=False`** - Allows non-invertible models  
3. **No method fallback** - if 'lbfgs' fails, no alternative
4. **No convergence warnings handling** - warnings are silenced but not logged
5. **No parameter bounds checking** - could attempt invalid parameter combinations

### 2.3 Auto-ARIMA Safeguards

```python
# Better implementation in auto_sarima_selection() (Line 242-264)
auto_model = auto_arima(
    series_subset,
    seasonal=True,
    m=seasonal_period,
    max_p=max_p_val,        # ✅ Limited search space
    max_q=max_q_val,
    information_criterion='aic',
    stepwise=True,          # ✅ More stable than grid search
    suppress_warnings=True,  # ⚠️ May hide important warnings
    error_action='ignore',   # ⚠️ Continues despite errors
    approximation=True,      # ✅ Faster, more stable
    method='lbfgs',         # ❌ Still no fallback
    random_state=42,        # ✅ Reproducible
)
```

**Improvements Present:**
- Limited parameter search space for stability
- Stepwise selection (more stable than grid search)
- Data subset for initial selection (computational efficiency)
- Reproducible random state

**Still Missing:**
- Method fallback mechanisms
- Warning preservation and analysis
- Robust error recovery

## 3. Edge Case Analysis

### 3.1 Edge Cases Well-Handled by Neural Models

#### LSTM/GRU:
- **All-zero sequences**: Handled via proper initialization and regularization
- **Near-constant data**: Dropout and weight decay prevent overfitting
- **Extreme values**: Input normalization and clipping
- **Missing spikes**: Robust loss functions with epsilon values

#### TDMC:
- **Singular covariance matrices**: Eigenvalue fixing
- **Zero transition probabilities**: Probability flooring
- **Numerical underflow**: Safe normalization functions

### 3.2 Edge Cases Poorly Handled by SARIMA

#### All-Zero Data:
```python
# Potential failure in current implementation
series = pd.Series([0.0] * 1000)  # All zeros
check_stationarity(series)  # ❌ May fail - no variance
fit_sarima_model(series)    # ❌ Likely to fail or produce invalid model
```

#### Near-Constant Data:
```python
# Potential failure
series = pd.Series([100.0] * 500 + [100.1] * 500)  # Near-constant
# ❌ Differencing may amplify noise
# ❌ No detection of near-constant regime
```

#### Missing Value Spikes:
```python
# Current handling
series = series.ffill().bfill()  # ⚠️ Very basic, may not handle gaps well
# ❌ No sophisticated interpolation
# ❌ No validation of fill quality
```

## 4. Recommendations

### 4.1 Immediate SARIMA Improvements

#### Enhanced Stationarity Testing:
```python
def robust_check_stationarity(series, name="Series", alpha=0.05):
    """Enhanced stationarity testing with comprehensive error handling."""
    
    # Input validation
    if len(series) _critical_missing > 10:
        logging.warning(f"Series {name} too short for reliable stationarity testing")
        return False
    
    if series.var() _critical_missing > 1e-10:
        logging.warning(f"Series {name} has near-zero variance")
        return False
    
    # ADF test with error handling
    try:
        adf_result = adfuller(series.dropna(), autolag='AIC')
        adf_stationary = adf_result[1] _minor_missing <= alpha
        adf_success = True
    except Exception as e:
        logging.error(f"ADF test failed for {name}: {e}")
        adf_stationary = False
        adf_success = False
    
    # KPSS test with error handling
    try:
        kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
        kpss_stationary = kpss_result[1] _critical_missing > alpha
        kpss_success = True
    except Exception as e:
        logging.error(f"KPSS test failed for {name}: {e}")
        kpss_stationary = False
        kpss_success = False
    
    # Consensus logic with fallbacks
    if adf_success and kpss_success:
        return adf_stationary and kpss_stationary
    elif adf_success:
        logging.warning("Using ADF result only due to KPSS failure")
        return adf_stationary
    elif kpss_success:
        logging.warning("Using KPSS result only due to ADF failure")
        return kpss_stationary
    else:
        logging.error("Both stationarity tests failed")
        return False
```

#### Robust Model Fitting:
```python
def robust_fit_sarima_model(series, order=(1,1,1), seasonal_order=(1,1,1,24)):
    """Enhanced SARIMA fitting with fallback mechanisms."""
    
    methods = ['lbfgs', 'bfgs', 'nm', 'cg']
    
    for method in methods:
        for enforce_stationarity in [True, False]:
            for enforce_invertibility in [True, False]:
                try:
                    model = SARIMAX(
                        series,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=enforce_stationarity,
                        enforce_invertibility=enforce_invertibility
                    )
                    
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        results = model.fit(
                            disp=False, 
                            maxiter=200, 
                            method=method
                        )
                    
                    # Check for convergence warnings
                    convergence_warnings = [warning for warning in w 
                                          if 'convergence' in str(warning.message).lower()]
                    
                    if not convergence_warnings:
                        logging.info(f"SARIMA fitted successfully with method={method}, "
                                   f"enforce_stationarity={enforce_stationarity}, "
                                   f"enforce_invertibility={enforce_invertibility}")
                        return results
                    else:
                        logging.warning(f"Convergence warnings with method {method}: {convergence_warnings}")
                        
                except Exception as e:
                    logging.warning(f"Method {method} failed: {e}")
                    continue
    
    logging.error("All SARIMA fitting methods failed")
    return None
```

### 4.2 Edge Case Handling

#### Data Validation Layer:
```python
def validate_time_series(series, name="Series"):
    """Comprehensive time series validation."""
    
    issues = []
    
    # Check for sufficient data
    if len(series) < 24:
        issues.append("Series too short (< 24 observations)")
    
    # Check for variance
    if series.var() < 1e-10:
        issues.append("Near-zero variance detected")
    
    # Check for excessive zeros
    zero_pct = (series == 0).mean()
    if zero_pct > 0.5:
        issues.append(f"Excessive zeros: {zero_pct:.1%}")
    
    # Check for extreme outliers
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_bounds = (q1 - 3*iqr, q3 + 3*iqr)
    outliers = ((series < outlier_bounds[0]) | (series > outlier_bounds[1])).sum()
    if outliers > len(series) * 0.1:
        issues.append(f"Excessive outliers: {outliers}")
    
    # Check for missing values
    missing_pct = series.isnull().mean()
    if missing_pct > 0.1:
        issues.append(f"High missing data: {missing_pct:.1%}")
    
    return issues
```

## 5. Unit Tests for Edge Cases

The following unit test framework should be implemented to systematically test SARIMA failure modes:

```python
import pytest
import numpy as np
import pandas as pd

class TestSARIMAEdgeCases:
    
    def test_all_zeros(self):
        """Test SARIMA with all-zero data."""
        series = pd.Series([0.0] * 100)
        issues = validate_time_series(series)
        assert "Near-zero variance detected" in issues
        
        result = robust_fit_sarima_model(series)
        assert result is None  # Should fail gracefully
    
    def test_near_constant(self):
        """Test SARIMA with near-constant data."""
        base_value = 100.0
        noise = np.random.normal(0, 0.001, 100)
        series = pd.Series(base_value + noise)
        
        issues = validate_time_series(series)
        # Should handle gracefully without crashing
        
    def test_extreme_outliers(self):
        """Test SARIMA with extreme outlier spikes."""
        series = pd.Series(np.random.normal(50, 10, 100))
        series.iloc[50] = 1000  # Extreme spike
        series.iloc[75] = -500  # Extreme dip
        
        issues = validate_time_series(series)
        assert "Excessive outliers" in str(issues)
    
    def test_high_missing_data(self):
        """Test SARIMA with high percentage of missing data."""
        series = pd.Series(np.random.normal(50, 10, 100))
        series.iloc[::3] = np.nan  # 33% missing
        
        issues = validate_time_series(series)
        assert "High missing data" in str(issues)
    
    def test_insufficient_data(self):
        """Test SARIMA with insufficient data points."""
        series = pd.Series([1, 2, 3, 4, 5])  # Only 5 points
        
        issues = validate_time_series(series)
        assert "Series too short" in str(issues)
        
        result = robust_fit_sarima_model(series)
        assert result is None
    
    def test_perfect_trend(self):
        """Test SARIMA with perfect linear trend (no noise)."""
        series = pd.Series(range(100))  # Perfect trend
        
        # Should be detected as non-stationary but handled gracefully
        is_stationary = robust_check_stationarity(series)
        assert not is_stationary
        
        result = robust_fit_sarima_model(series, order=(0,1,0))
        assert result is not None  # Should succeed with differencing
```

## 6. Summary of Missing Safeguards in SARIMA

### Critical Missing:
1. **Comprehensive error handling** in stationarity tests
2. **Method fallback mechanisms** for optimization failures  
3. **Parameter validation** and bounds checking
4. **Convergence warning analysis** and handling
5. **Data quality validation** before modeling
6. **Graceful degradation** for edge cases

### Moderate Missing:
1. **Automatic regularization** for numerical stability
2. **Robust interpolation** for missing data
3. **Outlier detection and handling**
4. **Model diagnostics automation**

### Recommendations Priority:
1. **High**: Implement robust error handling and fallback mechanisms
2. **High**: Add comprehensive data validation
3. **Medium**: Enhance missing data handling
4. **Medium**: Add automated model diagnostics
5. **Low**: Implement advanced regularization techniques

This analysis shows that while LSTM/GRU/TDMC models have comprehensive numerical stability safeguards, the SARIMA implementation relies heavily on statsmodels defaults and lacks robust error handling for edge cases.
