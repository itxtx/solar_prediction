import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
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
    
    def _initialize_parameters(self, X_scaled, time_indices):
        """
        Initialize model parameters using clustering.
        
        Parameters:
        -----------
        X_scaled : array-like
            Scaled emissions data
        time_indices : array-like
            Time slice indices for each observation
        """
        # Use K-means to initialize state assignments
        kmeans = KMeans(n_clusters=self.n_states, random_state=42)
        state_assignments = kmeans.fit_predict(X_scaled)
        
        # Initialize emission parameters based on clusters
        for s in range(self.n_states):
            state_data = X_scaled[state_assignments == s]
            if len(state_data) > 0:
                self.emission_means[s] = np.mean(state_data, axis=0)
                # Add small regularization to ensure non-singular covariance
                self.emission_covars[s] = np.cov(state_data.T) + 1e-6 * np.eye(self.n_emissions)
        
        # Initialize time-dependent transition matrices
        for t in range(self.time_slices):
            # Get data for this time slice
            time_slice_data = (time_indices == t)
            if np.sum(time_slice_data) > 1:
                # Count transitions within this time slice
                states_t = state_assignments[time_slice_data]
                for i in range(len(states_t) - 1):
                    s1, s2 = states_t[i], states_t[i+1]
                    self.transitions[t, s1, s2] += 1
                
                # Normalize to get probabilities
                row_sums = self.transitions[t].sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                self.transitions[t] /= row_sums
            else:
                # Not enough data for this time slice, use uniform distribution
                self.transitions[t] = np.ones((self.n_states, self.n_states)) / self.n_states
        
        # Initialize initial state distribution based on first observations for each time slice
        initial_states = np.zeros(self.n_states)
        for t in range(self.time_slices):
            time_slice_data = (time_indices == t)
            if np.sum(time_slice_data) > 0:
                first_state = state_assignments[time_slice_data][0]
                initial_states[first_state] += 1
        
        # Normalize to get probabilities
        self.initial_probs = initial_states / np.sum(initial_states)
        if np.sum(initial_states) == 0:
            self.initial_probs = np.ones(self.n_states) / self.n_states
    
    def _forward_backward(self, X_scaled, time_indices):
        """
        Perform forward-backward algorithm for the TDMC.
        
        Parameters:
        -----------
        X_scaled : array-like, shape (n_samples, n_emissions)
            Scaled emissions data
        time_indices : array-like, shape (n_samples,)
            Time slice indices for each observation
            
        Returns:
        --------
        alpha : array-like
            Forward probabilities
        beta : array-like
            Backward probabilities
        scale : array-like
            Scaling factors
        emission_probs : array-like
            Emission probabilities
        """
        n_samples = len(X_scaled)
        
        # Compute emission probabilities for all observations and states
        emission_probs = np.zeros((n_samples, self.n_states))
        for s in range(self.n_states):
            mvn = multivariate_normal(
                mean=self.emission_means[s],
                cov=self.emission_covars[s]
            )
            emission_probs[:, s] = mvn.pdf(X_scaled)
        
        # Initialize forward and backward variables
        alpha = np.zeros((n_samples, self.n_states))
        beta = np.zeros((n_samples, self.n_states))
        scale = np.zeros(n_samples)
        
        # Forward pass
        alpha[0] = self.initial_probs * emission_probs[0]
        scale[0] = np.sum(alpha[0])
        alpha[0] /= scale[0]
        
        for t in range(1, n_samples):
            time_slice = time_indices[t-1]  # Use previous time's slice for transition
            for s in range(self.n_states):
                alpha[t, s] = np.sum(alpha[t-1] * self.transitions[time_slice, :, s]) * emission_probs[t, s]
            
            scale[t] = np.sum(alpha[t])
            if scale[t] > 0:
                alpha[t] /= scale[t]
        
        # Backward pass
        beta[-1] = 1.0 / scale[-1]
        
        for t in range(n_samples-2, -1, -1):
            time_slice = time_indices[t]
            for s in range(self.n_states):
                beta[t, s] = np.sum(self.transitions[time_slice, s, :] * emission_probs[t+1, :] * beta[t+1, :])
            
            beta[t] /= scale[t]
        
        return alpha, beta, scale, emission_probs
    
    def _baum_welch_update(self, X_scaled, time_indices, max_iter=100, tol=1e-4):
        """
        Perform Baum-Welch algorithm to estimate model parameters.
        
        Parameters:
        -----------
        X_scaled : array-like
            Scaled emissions data
        time_indices : array-like
            Time slice indices for each observation
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
            
        Returns:
        --------
        log_likelihood : float
            Final log-likelihood of the model
        """
        n_samples = len(X_scaled)
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
            
            # Compute posteriors (gamma) and transition posteriors (xi)
            gamma = alpha * beta  # State posteriors
            
            # M-step: Update parameters
            # Update emission parameters
            for s in range(self.n_states):
                # Weighted sum for mean
                weighted_sum = np.sum(gamma[:, s].reshape(-1, 1) * X_scaled, axis=0)
                self.emission_means[s] = weighted_sum / np.sum(gamma[:, s])
                
                # Weighted covariance
                diff = X_scaled - self.emission_means[s]
                weighted_cov = np.zeros((self.n_emissions, self.n_emissions))
                for i in range(n_samples):
                    weighted_cov += gamma[i, s] * np.outer(diff[i], diff[i])
                self.emission_covars[s] = weighted_cov / np.sum(gamma[:, s])
                # Add regularization
                self.emission_covars[s] += 1e-6 * np.eye(self.n_emissions)
            
            # Update transition matrices for each time slice
            for t in range(self.time_slices):
                time_slice_indices = np.where(time_indices == t)[0]
                if len(time_slice_indices) > 1:
                    for i in range(len(time_slice_indices) - 1):
                        idx = time_slice_indices[i]
                        next_idx = time_slice_indices[i + 1]
                        
                        for s1 in range(self.n_states):
                            for s2 in range(self.n_states):
                                xi = alpha[idx, s1] * self.transitions[t, s1, s2] * \
                                     emission_probs[next_idx, s2] * beta[next_idx, s2]
                                
                                # Accumulate transitions
                                self.transitions[t, s1, s2] += xi
                    
                    # Normalize transitions for this time slice
                    row_sums = self.transitions[t].sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1  # Avoid division by zero
                    self.transitions[t] /= row_sums
            
            # Update initial state distribution
            self.initial_probs = gamma[0] / np.sum(gamma[0])
            
        return log_likelihood
    
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
        
        # Initialize parameters
        self._initialize_parameters(X_scaled, time_indices)
        
        # Run Baum-Welch algorithm
        log_likelihood = self._baum_welch_update(X_scaled, time_indices, max_iter, tol)
        
        # Set state names if provided
        if state_names is not None:
            if len(state_names) == self.n_states:
                self.state_names = state_names
        
        self.trained = True
        return self
    
    def predict_states(self, X, timestamps=None):
        """
        Predict the most likely hidden state sequence using Viterbi algorithm.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_emissions)
            Observable emissions data
        timestamps : array-like, shape (n_samples,)
            Timestamps for each observation
            
        Returns:
        --------
        states : array-like
            Most likely state sequence
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
    
    def forecast(self, X_last, timestamps_last, forecast_horizon, weather_forecasts=None):
        """
        Forecast future solar irradiance.
        
        Parameters:
        -----------
        X_last : array-like, shape (n_emissions,)
            Last observed emissions
        timestamps_last : int or timestamp
            Last timestamp
        forecast_horizon : int
            Number of steps ahead to forecast
        weather_forecasts : array-like, optional
            External weather forecasts to improve predictions
            
        Returns:
        --------
        forecasts : array-like
            Predicted emissions for forecast horizon
        confidence : array-like
            Confidence intervals for predictions
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
        
        # Inverse transform to original scale
        forecasts = self.scaler.inverse_transform(forecasts)
        confidence_lower = self.scaler.inverse_transform(confidence_lower)
        confidence_upper = self.scaler.inverse_transform(confidence_upper)
        
        return forecasts, (confidence_lower, confidence_upper)
    
    def get_state_characteristics(self):
        """
        Get characteristics of each hidden state.
        
        Returns:
        --------
        state_info : dict
            Dictionary containing information about each state
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
        
        Parameters:
        -----------
        time_slice : int
            Time slice to visualize (e.g., hour of day)
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
            ax.set_xlabel("Emission 1 (Standardized)")
            ax.set_ylabel("Emission 2 (Standardized)")
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


# Example usage with synthetic data
def create_synthetic_solar_data(n_days=30, time_points_per_day=24, noise_level=0.2):
    """Create synthetic solar irradiance data for testing"""
    n_samples = n_days * time_points_per_day
    
    # Time indices (hour of day)
    time_indices = np.tile(np.arange(time_points_per_day), n_days)
    
    # Base irradiance pattern (bell curve centered at noon)
    hour_pattern = np.sin(np.pi * np.arange(time_points_per_day) / (time_points_per_day - 1))
    base_irradiance = np.tile(hour_pattern, n_days)
    
    # Add weather state patterns (hidden states)
    # 4 weather states: Sunny, Partly Cloudy, Cloudy, Rainy
    state_durations = [24, 24, 12, 12]  # Each state lasts this many hours on average
    transition_probs = [
        [0.8, 0.2, 0.0, 0.0],  # Sunny -> Sunny (0.8), Partly Cloudy (0.2)
        [0.3, 0.6, 0.1, 0.0],  # Partly Cloudy -> Sunny (0.3), Partly Cloudy (0.6), Cloudy (0.1)
        [0.1, 0.3, 0.5, 0.1],  # Cloudy -> Sunny (0.1), Partly Cloudy (0.3), Cloudy (0.5), Rainy (0.1)
        [0.0, 0.2, 0.3, 0.5]   # Rainy -> Partly Cloudy (0.2), Cloudy (0.3), Rainy (0.5)
    ]
    
    # State modifiers for irradiance and temperature
    state_irradiance_factor = [1.0, 0.7, 0.4, 0.2]  # How much each state affects irradiance
    state_temp_modifier = [5, 2, 0, -3]  # How much each state affects temperature (°C)
    
    # Generate underlying weather states
    states = np.zeros(n_samples, dtype=int)
    states[0] = np.random.choice(4, p=[0.4, 0.3, 0.2, 0.1])  # Initial state
    
    for i in range(1, n_samples):
        # Time-dependent transition probabilities (simplified)
        hour = time_indices[i]
        time_factor = 1.0
        
        # Higher chance of weather changes in the morning/evening
        if 5 <= hour <= 9 or 17 <= hour <= 20:
            time_factor = 1.5
        
        # Calculate transition probabilities for current state
        current_state = states[i-1]
        probs = np.array(transition_probs[current_state])
        
        # Apply time factor (more likely to change at certain hours)
        if current_state != np.argmax(probs):  # If not self-transition
            probs = probs * time_factor
            probs = probs / np.sum(probs)  # Re-normalize
        
        # Determine next state
        states[i] = np.random.choice(4, p=probs)
    
    # Generate observed data based on states
    irradiance = np.zeros(n_samples)
    temperature = np.zeros(n_samples)
    
    for i in range(n_samples):
        state = states[i]
        hour = time_indices[i]
        
        # Base irradiance from time of day
        irradiance[i] = base_irradiance[i] * state_irradiance_factor[state]
        
        # Add seasonal component (assuming 30 days covers a month)
        day = i // time_points_per_day
        seasonal_factor = 0.9 + 0.2 * np.sin(2 * np.pi * day / n_days)
        irradiance[i] *= seasonal_factor
        
        # Temperature model: daily pattern + state modifier
        base_temp = 20 + 5 * np.sin(np.pi * hour / 12 - np.pi/2)  # Daily cycle centered at noon
        temperature[i] = base_temp + state_temp_modifier[state]
        
        # Add noise
        irradiance[i] += np.random.normal(0, noise_level * irradiance[i])
        temperature[i] += np.random.normal(0, 2)  # 2°C noise for temperature
    
    # Ensure non-negative irradiance
    irradiance = np.maximum(irradiance, 0)
    
    # Create dataframe
    dates = []
    for day in range(n_days):
        for hour in range(time_points_per_day):
            dates.append(pd.Timestamp('2023-06-01') + pd.Timedelta(days=day, hours=hour))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'irradiance': irradiance,
        'temperature': temperature,
        'true_state': states  # Keep for evaluation
    })
    
    return df

# Example function to train and evaluate the model
def train_and_evaluate_tdmc():
    """Train and evaluate the TDMC model with synthetic data"""
    # Generate synthetic data
    print("Generating synthetic solar data...")
    data = create_synthetic_solar_data(n_days=60, noise_level=0.15)
    
    # Extract features and timestamps
    X = data[['irradiance', 'temperature']].values
    timestamps = data['timestamp'].values
    
    # Split into train/test sets
    train_days = 45
    train_idx = train_days * 24
    X_train, X_test = X[:train_idx], X[train_idx:]
    timestamps_train, timestamps_test = timestamps[:train_idx], timestamps[train_idx:]
    true_states_train, true_states_test = data['true_state'].values[:train_idx], data['true_state'].values[train_idx:]
    
    # Initialize and train the model
    print("Training TDMC model...")
    model = SolarTDMC(n_states=4, n_emissions=2, time_slices=24)
    model.fit(
        X_train, timestamps_train, 
        max_iter=50, 
        state_names=['Sunny', 'Partly Cloudy', 'Cloudy', 'Rainy']
    )
    
    # Predict states
    print("Predicting states...")
    predicted_states = model.predict_states(X_test, timestamps_test)
    
    # Evaluate state prediction accuracy
    # Note: HMM states may not align with true states, so this is simplified
    state_accuracy = np.mean(predicted_states == true_states_test)
    print(f"State prediction accuracy: {state_accuracy:.2f}")
    print("(Note: State accuracy may be low due to label permutation in HMMs)")
    
    # Generate forecast
    print("Generating forecasts...")
    last_obs = X_test[0]
    last_timestamp = timestamps_test[0]
    forecast_horizon = 48  # 48 hours ahead
    
    forecasts, (conf_lower, conf_upper) = model.forecast(
        last_obs, last_timestamp, forecast_horizon
    )
    
    # Evaluate forecast error
    actual = X_test[1:min(forecast_horizon+1, len(X_test))]
    pred = forecasts[:min(forecast_horizon, len(X_test)-1)]
    
    mse = np.mean((actual[:, 0] - pred[:, 0])**2)  # MSE for irradiance
    print(f"Mean Squared Error for irradiance forecast: {mse:.4f}")
    
    # Get state characteristics
    state_info = model.get_state_characteristics()
    print("\nState Characteristics:")
    for state, info in state_info.items():
        print(f"- {state}: Mean Irradiance = {info['mean_emissions'][0]:.2f}, "
              f"Mean Temperature = {info['mean_emissions'][1]:.2f}°C")
    
    # Plot results
    print("Plotting results...")
    
    # Actual vs Predicted irradiance
    plt.figure(figsize=(12, 6))
    plt.plot(actual[:, 0], label='Actual Irradiance')
    plt.plot(pred[:, 0], label='Predicted Irradiance')
    plt.fill_between(
        range(len(pred)), 
        conf_lower[:len(pred), 0], 
        conf_upper[:len(pred), 0], 
        alpha=0.3, label='95% Confidence Interval'
    )
    plt.title('Irradiance Forecast vs Actual')
    plt.xlabel('Hours ahead')
    plt.ylabel('Irradiance')
    plt.legend()
    plt.grid(True)
    
    # Plot state transitions
    model.plot_state_transitions(time_slice=12)  # Noon transitions
    
    # Plot emissions by state
    model.plot_emissions_by_state()
    
    return model, data

# Example of how to use the model for solar production prediction
def solar_production_prediction(model, solar_capacity_kw=5.0):
    """Forecast solar energy production using the TDMC model"""
    print("\nSolar Production Prediction Example")
    
    # Generate synthetic data for current conditions
    current_data = create_synthetic_solar_data(n_days=1, time_points_per_day=24)
    current_obs = current_data[['irradiance', 'temperature']].values[-1]
    current_time = current_data['timestamp'].values[-1]
    
    # Generate forecast for next 3 days
    forecast_horizon = 24 * 3
    irradiance_forecast, confidence = model.forecast(current_obs, current_time, forecast_horizon)
    
    # Convert irradiance to energy production (kWh)
    # Simple model: production = irradiance_ratio * capacity * efficiency
    efficiency = 0.15  # 15% panel efficiency
    production_forecast = irradiance_forecast[:, 0] * solar_capacity_kw * efficiency
    
    # Time series of forecast days
    forecast_times = [current_time + pd.Timedelta(hours=i+1) for i in range(forecast_horizon)]
    
    # Plot production forecast
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_times, production_forecast)
    plt.title(f'Solar Energy Production Forecast ({solar_capacity_kw} kW System)')
    plt.xlabel('Date')
    plt.ylabel('Energy Production (kWh)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    
    # Daily production summary
    daily_production = {}
    for day_idx in range(3):
        start_idx = day_idx * 24
        end_idx = start_idx + 24
        if end_idx <= len(production_forecast):
            day_production = sum(production_forecast[start_idx:end_idx])
            day_date = forecast_times[start_idx].date()
            daily_production[day_date] = day_production
    
    print("Daily Production Forecast:")
    for date, prod in daily_production.items():
        print(f"- {date}: {prod:.2f} kWh")
    
    return production_forecast, forecast_times

if __name__ == "__main__":
    # Train and evaluate the model
    model, data = train_and_evaluate_tdmc()
    
    # Run production prediction example
    solar_production_prediction(model, solar_capacity_kw=7.5)
    
    plt.show()