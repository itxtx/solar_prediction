import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

class SolarTDMC:
    """
    Time-dynamic Markov Chain (TDMC) implementation for solar irradiance prediction.
    This is a specialized Hidden Markov Model with time-dependent transition probabilities.
    """
    
    def __init__(self, n_states=4, n_emissions=2, time_slices=24):
        """
        Initialize the TDMC model.
        
        Parameters:
        -----------
        n_states : int
            Number of hidden states in the model
        n_emissions : int
            Number of observable emission variables (e.g., irradiance, temperature)
        time_slices : int
            Number of time slices for the day (e.g., 24 for hourly)
        """
        self.n_states = n_states
        self.n_emissions = n_emissions
        self.time_slices = time_slices
        
        # Initialize time-dependent transition matrices (one for each time slice)
        self.transitions = np.zeros((time_slices, n_states, n_states))
        for t in range(time_slices):
            # Initialize with equal probabilities
            self.transitions[t] = np.ones((n_states, n_states)) / n_states
        
        # Initialize emission parameters (means and covariances for each state)
        self.emission_means = np.zeros((n_states, n_emissions))
        self.emission_covars = np.zeros((n_states, n_emissions, n_emissions))
        for s in range(n_states):
            self.emission_covars[s] = np.eye(n_emissions)
        
        # Initial state distribution
        self.initial_probs = np.ones(n_states) / n_states
        
        # Scaling for standardizing input data
        self.scaler = StandardScaler()
        
        # For tracking training
        self.trained = False
        self.state_names = [f"State_{i}" for i in range(n_states)]
    
    def _preprocess_data(self, X, timestamps=None):
        """
        Preprocess input data for the model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_emissions)
            Observable emissions data (e.g., irradiance, temperature)
        timestamps : array-like, shape (n_samples,)
            Timestamps for each observation (used to determine time slice)
            
        Returns:
        --------
        X_scaled : array-like
            Scaled emissions data
        time_indices : array-like
            Time slice indices for each observation
        """
        # Scale the data
        if not self.trained:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Process timestamps into time slice indices
        if timestamps is not None:
            # Convert timestamps to hours and map to time slices
            if isinstance(timestamps[0], (pd.Timestamp, np.datetime64)):
                time_indices = np.array([ts.hour for ts in pd.DatetimeIndex(timestamps)])
            else:
                # Assume timestamps are already hour indicators (0-23)
                time_indices = np.array(timestamps)
        else:
            # Default to time slice 0 for all observations if no timestamps provided
            time_indices = np.zeros(len(X), dtype=int)
            
        return X_scaled, time_indices
    
    def fit(self, X, timestamps=None, max_iter=100, tol=1e-4, state_names=None):
        """
        Fit the TDMC model to data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_emissions)
            Observable emissions data (e.g., irradiance, temperature)
        timestamps : array-like, shape (n_samples,)
            Timestamps for each observation
        max_iter : int
            Maximum number of iterations for Baum-Welch
        tol : float
            Convergence tolerance
        state_names : list
            Optional names for the hidden states
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Preprocess data
        X_scaled, time_indices = self._preprocess_data(X, timestamps)
        
        # Initialize parameters using K-means
        self._initialize_parameters(X_scaled, time_indices)
        
        # Run Baum-Welch algorithm
        self._baum_welch_update(X_scaled, time_indices, max_iter, tol)
        
        # Set state names if provided
        if state_names is not None:
            if len(state_names) == self.n_states:
                self.state_names = state_names
        
        self.trained = True
        return self
    
    def _initialize_parameters(self, X_scaled, time_indices):
        """Initialize model parameters using clustering."""
        # Use K-means to initialize state assignments
        kmeans = KMeans(n_clusters=self.n_states, random_state=42)
        state_assignments = kmeans.fit_predict(X_scaled)
        
        # Initialize emission parameters based on clusters
        for s in range(self.n_states):
            state_data = X_scaled[state_assignments == s]
            if len(state_data) > 0:
                self.emission_means[s] = np.mean(state_data, axis=0)
                # Add stronger regularization to ensure non-singular covariance
                cov = np.cov(state_data.T)
                # Ensure positive definiteness
                min_eig = np.min(np.real(np.linalg.eigvals(cov)))
                if min_eig < 0:
                    cov -= min_eig * np.eye(self.n_emissions)
                self.emission_covars[s] = cov + 1e-3 * np.eye(self.n_emissions)
            else:
                # If no data for this state, use global statistics
                self.emission_means[s] = np.mean(X_scaled, axis=0)
                self.emission_covars[s] = np.cov(X_scaled.T) + 1e-3 * np.eye(self.n_emissions)
        
        # Initialize time-dependent transition matrices with small uniform prior
        self.transitions = np.ones((self.time_slices, self.n_states, self.n_states)) / self.n_states
        
        for t in range(self.time_slices):
            # Get data for this time slice
            time_slice_data = (time_indices == t)
            if np.sum(time_slice_data) > 1:
                # Count transitions within this time slice
                states_t = state_assignments[time_slice_data]
                for i in range(len(states_t) - 1):
                    s1, s2 = states_t[i], states_t[i+1]
                    self.transitions[t, s1, s2] += 1
                
                # Normalize to get probabilities with smoothing
                row_sums = self.transitions[t].sum(axis=1, keepdims=True)
                self.transitions[t] = (self.transitions[t] + 1e-6) / (row_sums + self.n_states * 1e-6)
        
        # Initialize initial state distribution with smoothing
        initial_states = np.ones(self.n_states)  # Start with uniform prior
        for t in range(self.time_slices):
            time_slice_data = (time_indices == t)
            if np.sum(time_slice_data) > 0:
                first_state = state_assignments[time_slice_data][0]
                initial_states[first_state] += 1
        
        # Normalize with smoothing
        self.initial_probs = (initial_states + 1e-6) / (np.sum(initial_states) + self.n_states * 1e-6)
    
    def _forward_backward(self, X_scaled, time_indices):
        """Perform forward-backward algorithm for the TDMC."""
        n_samples = len(X_scaled)
        
        # Compute emission probabilities for all observations and states
        emission_probs = np.zeros((n_samples, self.n_states))
        for s in range(self.n_states):
            try:
                mvn = multivariate_normal(
                    mean=self.emission_means[s],
                    cov=self.emission_covars[s]
                )
                emission_probs[:, s] = mvn.pdf(X_scaled)
            except:
                # If multivariate normal fails, use a simple Gaussian approximation
                emission_probs[:, s] = np.exp(-0.5 * np.sum(
                    (X_scaled - self.emission_means[s])**2, axis=1
                ))
        
        # Add small constant to avoid numerical underflow
        emission_probs = np.maximum(emission_probs, 1e-300)
        
        # Initialize forward and backward variables
        alpha = np.zeros((n_samples, self.n_states))
        beta = np.zeros((n_samples, self.n_states))
        scale = np.zeros(n_samples)
        
        # Forward pass with scaling
        alpha[0] = self.initial_probs * emission_probs[0]
        scale[0] = np.sum(alpha[0])
        alpha[0] /= scale[0]
        
        for t in range(1, n_samples):
            for s2 in range(self.n_states):
                alpha[t, s2] = np.sum(
                    alpha[t-1] * self.transitions[time_indices[t-1], :, s2]
                ) * emission_probs[t, s2]
            scale[t] = np.sum(alpha[t])
            alpha[t] /= scale[t]
        
        # Backward pass with scaling
        beta[-1] = 1.0
        for t in range(n_samples-2, -1, -1):
            for s1 in range(self.n_states):
                beta[t, s1] = np.sum(
                    beta[t+1] * self.transitions[time_indices[t], s1, :] * emission_probs[t+1]
                )
            beta[t] /= scale[t+1]
        
        return alpha, beta, scale, emission_probs
    
    def _baum_welch_update(self, X_scaled, time_indices, max_iter=100, tol=1e-4):
        """Perform Baum-Welch algorithm to update model parameters."""
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iter):
            # E-step: Forward-backward algorithm
            alpha, beta, scale, emission_probs = self._forward_backward(X_scaled, time_indices)
            
            # Compute log-likelihood from scaling factors
            log_likelihood = np.sum(np.log(scale))
            
            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood
            
            # M-step: Update parameters
            n_samples = len(X_scaled)
            
            # Update initial state distribution
            gamma = alpha * beta
            self.initial_probs = gamma[0] / np.sum(gamma[0])
            
            # Update transition matrices
            for t in range(self.time_slices):
                # Get indices where time_indices equals t, but exclude the last element
                # since we need to look ahead one step
                time_slice_indices = np.where(time_indices[:-1] == t)[0]
                if len(time_slice_indices) == 0:
                    continue
                    
                # Compute xi for this time slice
                xi = np.zeros((self.n_states, self.n_states))
                for s1 in range(self.n_states):
                    for s2 in range(self.n_states):
                        # Use the valid indices to compute xi
                        xi[s1, s2] = np.sum(
                            alpha[time_slice_indices, s1] * 
                            self.transitions[t, s1, s2] * 
                            emission_probs[time_slice_indices + 1, s2] * 
                            beta[time_slice_indices + 1, s2]
                        )
                
                # Update transitions with smoothing
                row_sums = np.sum(xi, axis=1, keepdims=True)
                row_sums = np.maximum(row_sums, 1e-300)  # Avoid division by zero
                self.transitions[t] = (xi + 1e-6) / (row_sums + self.n_states * 1e-6)
            
            # Update emission parameters
            for s in range(self.n_states):
                gamma_s = gamma[:, s]
                gamma_s = np.maximum(gamma_s, 1e-300)  # Avoid numerical issues
                
                # Update mean
                self.emission_means[s] = np.sum(gamma_s[:, np.newaxis] * X_scaled, axis=0) / np.sum(gamma_s)
                
                # Update covariance with regularization
                diff = X_scaled - self.emission_means[s]
                cov = np.dot(gamma_s * diff.T, diff) / np.sum(gamma_s)
                # Ensure positive definiteness
                min_eig = np.min(np.real(np.linalg.eigvals(cov)))
                if min_eig < 0:
                    cov -= min_eig * np.eye(self.n_emissions)
                self.emission_covars[s] = cov + 1e-3 * np.eye(self.n_emissions)
        
        return log_likelihood
    
    def predict_states(self, X, timestamps=None):
        """
        Predict the most likely hidden state sequence using Viterbi algorithm.
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
        
        # Preprocess data
        X_scaled, time_indices = self._preprocess_data(X, timestamps)
        n_samples = len(X_scaled)
        
        # Compute emission probabilities
        emission_probs = np.zeros((n_samples, self.n_states))
        for s in range(self.n_states):
            mvn = multivariate_normal(
                mean=self.emission_means[s],
                cov=self.emission_covars[s]
            )
            emission_probs[:, s] = mvn.pdf(X_scaled)
        
        # Initialize Viterbi variables
        viterbi = np.zeros((n_samples, self.n_states))
        backpointers = np.zeros((n_samples, self.n_states), dtype=int)
        
        # Initialize first step
        viterbi[0] = np.log(self.initial_probs + 1e-10) + np.log(emission_probs[0] + 1e-10)
        
        # Recursion
        for t in range(1, n_samples):
            time_slice = time_indices[t-1]
            for s in range(self.n_states):
                # Calculate probabilities and find the best previous state
                probs = viterbi[t-1] + np.log(self.transitions[time_slice, :, s] + 1e-10)
                backpointers[t, s] = np.argmax(probs)
                viterbi[t, s] = probs[backpointers[t, s]] + np.log(emission_probs[t, s] + 1e-10)
        
        # Backtrack to find the most likely sequence
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(viterbi[-1])
        for t in range(n_samples-2, -1, -1):
            states[t] = backpointers[t+1, states[t+1]]
        
        return states
    
    def _inverse_transform_target(self, y, target_scaler, transform_info):
        """
        Apply inverse transformations to recover original target values
        
        Args:
            y: Transformed target values
            target_scaler: Scaler used for the target
            transform_info: Combined transform info dictionary
            
        Returns:
            Original scale target values
        """
        # Make a copy to avoid modifying the original
        y_transformed = y.copy()
        
        # Get the list of transforms in reverse order (to undo in reverse)
        transforms = transform_info.get('transforms', [])[::-1]
        
        # Apply inverse transformations in reverse order
        for transform in transforms:
            transform_type = transform.get('type')
            
            if transform_type == 'log' and transform.get('applied', False):
                # Undo log transform
                if transform.get('offset', 0) > 0:
                    # If log1p was used: exp(y) - offset
                    y_transformed = np.exp(y_transformed) - transform.get('offset')
                else:
                    # If simple log was used
                    y_transformed = np.exp(y_transformed)
                    
            elif transform_type == 'scale' and transform.get('applied', False):
                # No direct inverse needed as the scaler will handle this
                pass
        
        # Apply inverse scaling if scaler is provided
        if target_scaler is not None:
            # Create dummy array with the right shape for inverse_transform
            if len(y_transformed.shape) > 1 and y_transformed.shape[1] == 1:
                y_transformed = y_transformed.squeeze(axis=1)
                
            # Check if we need a dummy array (if y is just the target column)
            if hasattr(target_scaler, 'n_features_in_') and target_scaler.n_features_in_ > 1:
                # Create dummy array with zeros except for target column
                dummy = np.zeros((y_transformed.shape[0], target_scaler.n_features_in_))
                
                # Find target column index from transform_info
                target_col = transform_info.get('target_col_original', -1)
                if isinstance(target_col, str) and hasattr(target_scaler, 'feature_names_in_'):
                    # If we have column names, find the index
                    try:
                        target_idx = np.where(target_scaler.feature_names_in_ == target_col)[0][0]
                    except:
                        # Default to last column if name not found
                        target_idx = -1
                else:
                    # Default to the provided index or last column
                    target_idx = -1 if isinstance(target_col, str) else target_col
                
                # Place values in the correct column
                dummy[:, target_idx] = y_transformed
                
                # Apply inverse transform
                transformed_dummy = target_scaler.inverse_transform(dummy)
                
                # Extract target column
                y_transformed = transformed_dummy[:, target_idx]
            else:
                # If scaler was fitted only on target, just inverse transform directly
                y_transformed = target_scaler.inverse_transform(
                    y_transformed.reshape(-1, 1)).squeeze()
        
        return y_transformed

    def forecast(self, X_last, timestamps_last, forecast_horizon, weather_forecasts=None):
        """
        Forecast future solar irradiance.
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
        
        # Determine current state
        X_last_scaled = self.scaler.transform(X_last.reshape(1, -1))
        
        # Calculate emission probabilities for current state
        state_probs = np.zeros(self.n_states)
        for s in range(self.n_states):
            mvn = multivariate_normal(
                mean=self.emission_means[s],
                cov=self.emission_covars[s]
            )
            state_probs[s] = mvn.pdf(X_last_scaled)
        
        current_state = np.argmax(state_probs)
        
        # Generate future timestamps
        if isinstance(timestamps_last, (pd.Timestamp, np.datetime64)):
            future_timestamps = [timestamps_last + pd.Timedelta(hours=i+1) for i in range(forecast_horizon)]
            future_time_indices = np.array([ts.hour for ts in future_timestamps])
        else:
            # Assume timestamps_last is an hour (0-23)
            future_time_indices = [(timestamps_last + i + 1) % self.time_slices for i in range(forecast_horizon)]
        
        # Initialize forecasts and confidence intervals
        forecasts = np.zeros((forecast_horizon, self.n_emissions))
        confidence_lower = np.zeros((forecast_horizon, self.n_emissions))
        confidence_upper = np.zeros((forecast_horizon, self.n_emissions))
        
        # Current state distribution
        state_distribution = np.zeros(self.n_states)
        state_distribution[current_state] = 1.0
        
        # Forecast for each step ahead
        for step in range(forecast_horizon):
            time_idx = future_time_indices[step]
            
            # Update state distribution using transition matrix
            state_distribution = np.dot(state_distribution, self.transitions[time_idx])
            
            # Calculate expected emission and variance for each variable
            forecast_mean = np.zeros(self.n_emissions)
            forecast_var = np.zeros(self.n_emissions)
            
            for s in range(self.n_states):
                forecast_mean += state_distribution[s] * self.emission_means[s]
                
                # Add variance contribution from each state
                forecast_var += state_distribution[s] * np.diag(self.emission_covars[s])
                
                # Add variance from state uncertainty
                for dim in range(self.n_emissions):
                    forecast_var[dim] += state_distribution[s] * (self.emission_means[s, dim] - forecast_mean[dim])**2
            
            # Incorporate external weather forecasts if available
            if weather_forecasts is not None and step < len(weather_forecasts):
                # Simple weighted average between model prediction and weather forecast
                alpha = 0.7  # Weight for external forecast
                forecasts[step] = (1 - alpha) * forecast_mean + alpha * weather_forecasts[step]
            else:
                forecasts[step] = forecast_mean
            
            # Calculate confidence intervals (95%)
            forecast_std = np.sqrt(forecast_var)
            confidence_lower[step] = forecasts[step] - 1.96 * forecast_std
            confidence_upper[step] = forecasts[step] + 1.96 * forecast_std
        
        # Inverse transform to original scale using the new method
        transform_info = {
            'transforms': [
                {'type': 'scale', 'applied': True}
            ],
            'target_col_original': -1  # Use all columns
        }
        
        forecasts = self._inverse_transform_target(forecasts, self.scaler, transform_info)
        confidence_lower = self._inverse_transform_target(confidence_lower, self.scaler, transform_info)
        confidence_upper = self._inverse_transform_target(confidence_upper, self.scaler, transform_info)
        
        return forecasts, (confidence_lower, confidence_upper)
    
    def get_state_characteristics(self):
        """
        Get characteristics of each hidden state.
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
        
        state_info = {}
        
        for s in range(self.n_states):
            # Transform means back to original scale
            orig_means = self.scaler.inverse_transform(self.emission_means[s].reshape(1, -1))[0]
            
            state_info[self.state_names[s]] = {
                'mean_emissions': orig_means,
                'most_likely_next_states': {
                    f'hour_{t}': [
                        (self.state_names[i], self.transitions[t, s, i])
                        for i in np.argsort(-self.transitions[t, s])[:3]
                    ]
                    for t in range(self.time_slices)
                }
            }
        
        return state_info
    
    def plot_state_transitions(self, time_slice=12):
        """
        Plot transition probabilities for a specific time slice.
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(self.transitions[time_slice], cmap='Blues')
        
        # Add labels
        ax.set_xticks(np.arange(self.n_states))
        ax.set_yticks(np.arange(self.n_states))
        ax.set_xticklabels(self.state_names)
        ax.set_yticklabels(self.state_names)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        
        # Add text annotations
        for i in range(self.n_states):
            for j in range(self.n_states):
                text = ax.text(j, i, f"{self.transitions[time_slice, i, j]:.2f}",
                              ha="center", va="center", color="black" if self.transitions[time_slice, i, j] < 0.5 else "white")
        
        ax.set_title(f"Transition Probabilities at Time Slice {time_slice}")
        ax.set_xlabel("To State")
        ax.set_ylabel("From State")
        fig.tight_layout()
        
        return fig
    
    def plot_emissions_by_state(self):
        """
        Plot emission distributions for each state.
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
        
        # Only implemented for 1D or 2D emissions
        if self.n_emissions > 2:
            raise ValueError("Plotting only supported for 1 or 2 emission dimensions")
        
        if self.n_emissions == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.linspace(
                min(self.emission_means[:, 0] - 3*np.sqrt(self.emission_covars[:, 0, 0])),
                max(self.emission_means[:, 0] + 3*np.sqrt(self.emission_covars[:, 0, 0])),
                1000
            )
            
            for s in range(self.n_states):
                y = multivariate_normal.pdf(
                    x, mean=self.emission_means[s, 0], cov=self.emission_covars[s, 0, 0]
                )
                ax.plot(x, y, label=self.state_names[s])
            
            ax.set_title("Emission Distributions by State")
            ax.set_xlabel("Emission Value (Standardized)")
            ax.set_ylabel("Probability Density")
            ax.legend()
            
        else:  # 2D emissions
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create grid for contour plot
            x, y = np.mgrid[
                min(self.emission_means[:, 0] - 3*np.sqrt(self.emission_covars[:, 0, 0])):
                max(self.emission_means[:, 0] + 3*np.sqrt(self.emission_covars[:, 0, 0])):100j,
                min(self.emission_means[:, 1] - 3*np.sqrt(self.emission_covars[:, 1, 1])):
                max(self.emission_means[:, 1] + 3*np.sqrt(self.emission_covars[:, 1, 1])):100j
            ]
            pos = np.dstack((x, y))
            
            for s in range(self.n_states):
                rv = multivariate_normal(
                    mean=self.emission_means[s],
                    cov=self.emission_covars[s]
                )
                ax.contour(x, y, rv.pdf(pos), levels=5, alpha=0.6, colors=[f'C{s}'])
                ax.scatter(self.emission_means[s, 0], self.emission_means[s, 1], 
                          c=[f'C{s}'], marker='x', s=100, label=self.state_names[s])
            
            ax.set_title("Emission Distributions by State")
            ax.set_xlabel("Radiation (Standardized)")
            ax.set_ylabel("Temperature (Standardized)")
            ax.legend()
            
        fig.tight_layout()
        return fig

    def save_model(self, filepath):
        """Save model to file"""
        model_data = {
            'n_states': self.n_states,
            'n_emissions': self.n_emissions,
            'time_slices': self.time_slices,
            'transitions': self.transitions,
            'emission_means': self.emission_means,
            'emission_covars': self.emission_covars,
            'initial_probs': self.initial_probs,
            'state_names': self.state_names,
            'scaler_mean_': self.scaler.mean_,
            'scaler_scale_': self.scaler.scale_,
            'trained': self.trained
        }
        np.save(filepath, model_data)
    
    @classmethod
    def load_model(cls, filepath):
        """Load model from file"""
        model_data = np.load(filepath, allow_pickle=True).item()
        
        model = cls(
            n_states=model_data['n_states'],
            n_emissions=model_data['n_emissions'],
            time_slices=model_data['time_slices']
        )
        
        model.transitions = model_data['transitions']
        model.emission_means = model_data['emission_means']
        model.emission_covars = model_data['emission_covars']
        model.initial_probs = model_data['initial_probs']
        model.state_names = model_data['state_names']
        
        model.scaler = StandardScaler()
        model.scaler.mean_ = model_data['scaler_mean_']
        model.scaler.scale_ = model_data['scaler_scale_']
        
        model.trained = model_data['trained']
        
        return model
