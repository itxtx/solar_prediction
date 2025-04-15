import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pmdarima as pm

class SolarARIMA:
    def __init__(self, order=(1,1,1), seasonal_order=None, method='css-mle', trend='c'):
        """
        Initialize ARIMA model for solar radiance forecasting
        
        Args:
            order: ARIMA order (p,d,q) 
                p: AR order (autoregression)
                d: Differencing
                q: MA order (moving average)
            seasonal_order: Seasonal ARIMA order (P,D,Q,s)
                Only used if not None (makes it SARIMA)
            method: Fitting method
            trend: Trend component ('c' for constant, 'n' for none)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.method = method
        self.trend = trend
        self.model = None
        self.model_fit = None
        self.is_seasonal = seasonal_order is not None
        
    def check_stationarity(self, series):
        """Test for stationarity using Augmented Dickey-Fuller test"""
        result = adfuller(series)
        print(f'ADF Statistic: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4f}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.4f}')
        
        # Rule of thumb: if p-value < 0.05, series is stationary
        is_stationary = result[1] < 0.05
        print(f'Series is {"stationary" if is_stationary else "non-stationary"}')
        return is_stationary
    
    def suggest_differencing(self, series, max_d=2):
        """Suggest differencing order based on stationarity tests"""
        for d in range(max_d + 1):
            if d == 0:
                test_series = series
            else:
                test_series = np.diff(series, n=d)
            
            result = adfuller(test_series)
            is_stationary = result[1] < 0.05
            
            if is_stationary:
                print(f"Series becomes stationary after {d} differencing")
                return d
        
        print(f"Series remains non-stationary after {max_d} differencing")
        return max_d
    
    def fit(self, series, exog=None):
        """
        Fit ARIMA model to the data
        
        Args:
            series: Time series data (pandas Series with DatetimeIndex or numpy array)
            exog: Exogenous variables (optional)
        
        Returns:
            self: Fitted model
        """
        # If series is not a pandas Series, convert it
        if not isinstance(series, pd.Series):
            if isinstance(series, np.ndarray) and series.ndim == 1:
                series = pd.Series(series)
            else:
                raise ValueError("Series must be a 1D array or pandas Series")
        
        # Create and fit model
        if self.is_seasonal:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            self.model = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                exog=exog,
                trend=self.trend
            )
        else:
            self.model = ARIMA(
                series,
                order=self.order,
                exog=exog,
                trend=self.trend
            )
        
        self.model_fit = self.model.fit(method=self.method)
        print(self.model_fit.summary())
        
        return self
    
    def forecast(self, steps=24, exog=None, alpha=0.05, return_conf_int=True):
        """
        Generate forecasts
        
        Args:
            steps: Number of steps to forecast
            exog: Exogenous variables for forecasting period
            alpha: Significance level for confidence intervals
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            forecast: Forecasted values
            confidence_intervals: If return_conf_int is True
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Generate forecast
        if hasattr(self.model_fit, 'get_forecast'):
            # For newer statsmodels versions
            forecast_result = self.model_fit.get_forecast(steps=steps, exog=exog)
            predicted_mean = forecast_result.predicted_mean
            
            if return_conf_int:
                confidence_intervals = forecast_result.conf_int(alpha=alpha)
                return predicted_mean, confidence_intervals
        else:
            # For older statsmodels versions
            forecast_result = self.model_fit.forecast(steps=steps, exog=exog)
            if return_conf_int:
                # This is a simplified approach - actual implementation depends on statsmodels version
                forecast, stderr, conf_int = self.model_fit.forecast(steps=steps, exog=exog, alpha=alpha)
                return forecast, conf_int
        
        return predicted_mean if 'predicted_mean' in locals() else forecast_result
    
    def evaluate(self, test_data, steps=None, exog=None):
        """
        Evaluate model on test data
        
        Args:
            test_data: Actual values to compare against
            steps: Forecast steps (if None, use length of test_data)
            exog: Exogenous variables for test period
            
        Returns:
            Dictionary of evaluation metrics
        """
        if steps is None:
            steps = len(test_data)
        
        # Generate forecast
        forecast = self.forecast(steps=steps, exog=exog, return_conf_int=False)
        if isinstance(forecast, tuple):
            forecast = forecast[0]  # Extract predictions if confidence intervals included
        
        # Convert to array for comparison
        if isinstance(forecast, (pd.Series, pd.DataFrame)):
            forecast = forecast.values
        if isinstance(test_data, (pd.Series, pd.DataFrame)):
            test_data = test_data.values
        
        # Trim if necessary
        min_len = min(len(forecast), len(test_data))
        forecast = forecast[:min_len]
        test_data = test_data[:min_len]
        
        # Calculate metrics
        mse = mean_squared_error(test_data, forecast)
        rmse = np.sqrt(mse)
        
        # Handle zeros in test data for MAPE
        nonzero_idx = test_data != 0
        if np.any(nonzero_idx):
            mape = mean_absolute_percentage_error(
                test_data[nonzero_idx], forecast[nonzero_idx]) * 100
        else:
            mape = np.nan
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }
        
        print(f"Evaluation Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return metrics, forecast, test_data
    
    def plot_forecast(self, train_data, test_data=None, forecast=None, 
                    steps=None, exog=None, figsize=(12, 6)):
        """
        Plot the forecast against actual values
        
        Args:
            train_data: Training data
            test_data: Test data (optional)
            forecast: Pre-computed forecast (optional)
            steps: Forecast steps if forecast not provided
            exog: Exogenous variables for forecast period
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        if forecast is None:
            if steps is None and test_data is not None:
                steps = len(test_data)
            elif steps is None:
                steps = 24  # Default forecast horizon
                
            forecast, conf_int = self.forecast(steps=steps, exog=exog)
        else:
            conf_int = None
        
        # Create time index for forecast
        if isinstance(train_data, pd.Series) and isinstance(train_data.index, pd.DatetimeIndex):
            # If we have a datetime index, continue it
            last_date = train_data.index[-1]
            freq = pd.infer_freq(train_data.index)
            forecast_index = pd.date_range(start=last_date, periods=steps+1, freq=freq)[1:]
            
            if isinstance(forecast, np.ndarray):
                forecast = pd.Series(forecast, index=forecast_index)
            
            if conf_int is not None and isinstance(conf_int, np.ndarray):
                conf_int = pd.DataFrame(conf_int, index=forecast_index, 
                                       columns=['lower', 'upper'])
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot training data
        if isinstance(train_data, pd.Series):
            train_data.plot(ax=ax, label='Training Data')
        else:
            ax.plot(np.arange(len(train_data)), train_data, label='Training Data')
        
        # Plot test data if provided
        if test_data is not None:
            if isinstance(test_data, pd.Series):
                test_data.plot(ax=ax, label='Test Data', color='green')
            else:
                offset = len(train_data)
                ax.plot(np.arange(offset, offset + len(test_data)), 
                       test_data, label='Test Data', color='green')
        
        # Plot forecast
        if isinstance(forecast, pd.Series):
            forecast.plot(ax=ax, label='Forecast', color='red')
        else:
            offset = len(train_data)
            ax.plot(np.arange(offset, offset + len(forecast)), 
                   forecast, label='Forecast', color='red')
        
        # Plot confidence intervals if available
        if conf_int is not None:
            if isinstance(conf_int, pd.DataFrame):
                ax.fill_between(conf_int.index, conf_int['lower'], conf_int['upper'],
                               color='pink', alpha=0.3)
            else:
                offset = len(train_data)
                ax.fill_between(
                    np.arange(offset, offset + len(conf_int)),
                    conf_int[:, 0], conf_int[:, 1],
                    color='pink', alpha=0.3
                )
        
        ax.set_title('ARIMA Forecast for Solar Radiance')
        ax.set_ylabel('Solar Radiance')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def auto_arima(self, series, max_p=3, max_d=2, max_q=3, exog=None, seasonal=False):
        """
        Find optimal ARIMA parameters using information criteria
        
        Args:
            series: Time series data
            max_p, max_d, max_q: Maximum values for ARIMA parameters
            exog: Exogenous variables
            seasonal: Whether to fit SARIMA models
            
        Returns:
            Best ARIMA order based on AIC
        """
        try:

            
            # Use pmdarima's auto_arima
            auto_model = pm.auto_arima(
                series,
                exogenous=exog,
                start_p=0, max_p=max_p,
                start_q=0, max_q=max_q,
                d=None, max_d=max_d,
                seasonal=seasonal,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            print(f"Best ARIMA order: {auto_model.order}")
            if seasonal:
                print(f"Best seasonal order: {auto_model.seasonal_order}")
                
            # Update model parameters
            self.order = auto_model.order
            if seasonal:
                self.seasonal_order = auto_model.seasonal_order
                self.is_seasonal = True
            
            return auto_model.order
            
        except ImportError:
            print("pmdarima not installed. Please install it for auto_arima functionality.")
            print("Falling back to manual parameter selection.")
            
            # Simple grid search
            best_aic = float('inf')
            best_order = None
            
            for p in range(max_p + 1):
                for d in range(max_d + 1):
                    for q in range(max_q + 1):
                        try:
                            model = ARIMA(series, order=(p, d, q), exog=exog)
                            results = model.fit(method=self.method)
                            aic = results.aic
                            
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)
                                
                            print(f"ARIMA({p},{d},{q}) AIC: {aic}")
                        except:
                            continue
            
            print(f"Best ARIMA order: {best_order} (AIC: {best_aic})")
            self.order = best_order
            return best_order
        
        