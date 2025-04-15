import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from data_prep import pca_transform

class SolarTDMC:
    """
    Time-dynamic Markov Chain (TDMC) implementation for solar irradiance prediction.
    This is a specialized Hidden Markov Model with time-dependent transition probabilities.
    """
    
    def __init__(self, n_states=4, n_emissions=None, time_slices=24, n_components=None):
        """
        Initialize the TDMC model.
        
        Parameters:
        -----------
        n_states : int
            Number of hidden states in the model
        n_emissions : int or None
            Number of observable emission variables (e.g., irradiance, temperature)
            If None, will be determined from the data during fit()
        time_slices : int
            Number of time slices for the day (e.g., 24 for hourly)
        n_components : int or None
            Number of PCA components to use. If None, will use enough components
            to explain 95% of the variance.
        """
        self.n_states = n_states
        self.n_emissions = n_emissions
        self.time_slices = time_slices
        self.n_components = n_components
        
        # Initialize PCA
        self.pca = PCA(n_components=n_components)
        
        # Initialize time-dependent transition matrices (one for each time slice)
        self.transitions = np.zeros((time_slices, n_states, n_states))
        for t in range(time_slices):
            # Initialize with equal probabilities
            self.transitions[t] = np.ones((n_states, n_states)) / n_states
        
        # Initialize emission parameters (means and covariances for each state)
        # These will be properly initialized in fit() when we know n_emissions
        self.emission_means = None
        self.emission_covars = None
        
        # Initial state distribution
        self.initial_probs = np.ones(n_states) / n_states
        
        # For tracking training
        self.trained = False
        self.state_names = [f"State_{i}" for i in range(n_states)]
    
    def fit(self, X, timestamps=None, max_iter=100, tol=1e-4, state_names=None):
        """
        Fit the TDMC model to data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, sequence_length, n_features)
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
        # Data is already transformed, just use it directly
        X_scaled = X
        
        # Convert timestamps to hour values and reshape to match sequence length
        sequence_length = X.shape[1]
        if timestamps is not None:
            # Convert timestamps to hour values (0-23)
            if isinstance(timestamps[0], (pd.Timestamp, np.datetime64)):
                hours = np.array([ts.hour for ts in timestamps])
            else:
                hours = timestamps
            # Reshape to match sequence length
            time_indices = np.repeat(hours[:, np.newaxis], sequence_length, axis=1)
        else:
            time_indices = np.zeros((X.shape[0], sequence_length), dtype=int)
        
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
        print(f"Input data shape: {X_scaled.shape}")
        
        # If n_emissions wasn't specified, determine it from the data
        if self.n_emissions is None:
            self.n_emissions = X_scaled.shape[-1]
            print(f"Setting n_emissions to: {self.n_emissions}")
            
        # Initialize emission parameters with correct dimensions
        self.emission_means = np.zeros((self.n_states, self.n_emissions))
        self.emission_covars = np.zeros((self.n_states, self.n_emissions, self.n_emissions))
        print(f"Initialized emission_means shape: {self.emission_means.shape}")
        print(f"Initialized emission_covars shape: {self.emission_covars.shape}")
        
        # Reshape 3D sequence data into 2D for clustering
        # X_scaled shape: (n_sequences, sequence_length, n_features)
        n_sequences, sequence_length, n_features = X_scaled.shape
        print(f"n_sequences: {n_sequences}, sequence_length: {sequence_length}, n_features: {n_features}")
        
        X_reshaped = X_scaled.reshape(-1, n_features)  # Flatten sequences
        print(f"Reshaped data shape: {X_reshaped.shape}")
        
        # Use K-means to initialize state assignments
        kmeans = KMeans(n_clusters=self.n_states, random_state=42)
        state_assignments = kmeans.fit_predict(X_reshaped)
        print(f"State assignments shape: {state_assignments.shape}")
        
        # Reshape state assignments back to match sequences
        state_assignments = state_assignments.reshape(n_sequences, sequence_length)
        print(f"Reshaped state assignments shape: {state_assignments.shape}")
        
        # Initialize emission parameters based on clusters
        for s in range(self.n_states):
            # Get all observations assigned to this state
            state_data = X_reshaped[state_assignments.flatten() == s]
            print(f"State {s} data shape: {state_data.shape}")
            
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
                self.emission_means[s] = np.mean(X_reshaped, axis=0)
                self.emission_covars[s] = np.cov(X_reshaped.T) + 1e-3 * np.eye(self.n_emissions)
        
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
        sequence_length = X_scaled.shape[1]
        
        # Initialize emission probabilities
        emission_probs = np.zeros((n_samples, sequence_length, self.n_states))
        
        # Compute emission probabilities for all observations and states
        for s in range(self.n_states):
            try:
                mvn = multivariate_normal(
                    mean=self.emission_means[s],
                    cov=self.emission_covars[s]
                )
                # Calculate probabilities for each time step in each sequence
                for t in range(sequence_length):
                    emission_probs[:, t, s] = mvn.pdf(X_scaled[:, t, :])
            except:
                # If multivariate normal fails, use a simple Gaussian approximation
                for t in range(sequence_length):
                    emission_probs[:, t, s] = np.exp(-0.5 * np.sum(
                        (X_scaled[:, t, :] - self.emission_means[s])**2, axis=1
                    ))
        
        # Add small constant to avoid numerical underflow
        emission_probs = np.maximum(emission_probs, 1e-300)
        
        # Initialize forward and backward variables
        alpha = np.zeros((n_samples, sequence_length, self.n_states))
        beta = np.zeros((n_samples, sequence_length, self.n_states))
        scale = np.zeros((n_samples, sequence_length))
        
        # Forward pass with scaling
        alpha[:, 0] = self.initial_probs * emission_probs[:, 0]
        scale[:, 0] = np.sum(alpha[:, 0], axis=1)
        alpha[:, 0] = (alpha[:, 0].T / scale[:, 0]).T
        
        for t in range(1, sequence_length):
            for s2 in range(self.n_states):
                alpha[:, t, s2] = np.sum(
                    alpha[:, t-1] * self.transitions[time_indices[:, t-1], :, s2],
                    axis=1
                ) * emission_probs[:, t, s2]
            scale[:, t] = np.sum(alpha[:, t], axis=1)
            alpha[:, t] = (alpha[:, t].T / scale[:, t]).T
        
        # Backward pass with scaling
        beta[:, -1] = 1.0
        for t in range(sequence_length-2, -1, -1):
            for s1 in range(self.n_states):
                beta[:, t, s1] = np.sum(
                    beta[:, t+1] * self.transitions[time_indices[:, t], s1, :] * emission_probs[:, t+1],
                    axis=1
                )
            beta[:, t] = (beta[:, t].T / scale[:, t+1]).T
        
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
            n_samples, sequence_length = X_scaled.shape[:2]
            
            # Update initial state distribution
            gamma = alpha * beta
            self.initial_probs = gamma[:, 0].sum(axis=0) / np.sum(gamma[:, 0])
            
            # Update transition matrices
            for t in range(self.time_slices):
                # Get indices where time_indices equals t, but exclude the last element
                # since we need to look ahead one step
                time_slice_mask = (time_indices[:, :-1] == t)
                if np.sum(time_slice_mask) == 0:
                    continue
                    
                # Compute xi for this time slice
                xi = np.zeros((self.n_states, self.n_states))
                for s1 in range(self.n_states):
                    for s2 in range(self.n_states):
                        # Use the valid indices to compute xi
                        xi[s1, s2] = np.sum(
                            alpha[:, :-1, s1][time_slice_mask] * 
                            self.transitions[t, s1, s2] * 
                            emission_probs[:, 1:, s2][time_slice_mask] * 
                            beta[:, 1:, s2][time_slice_mask]
                        )
                
                # Update transitions with smoothing
                row_sums = np.sum(xi, axis=1, keepdims=True)
                row_sums = np.maximum(row_sums, 1e-300)  # Avoid division by zero
                self.transitions[t] = (xi + 1e-6) / (row_sums + self.n_states * 1e-6)
            
            # Update emission parameters
            for s in range(self.n_states):
                gamma_s = gamma[:, :, s]
                gamma_s = np.maximum(gamma_s, 1e-300)  # Avoid numerical issues
                
                # Update mean
                self.emission_means[s] = np.sum(
                    gamma_s[:, :, np.newaxis] * X_scaled, 
                    axis=(0, 1)
                ) / np.sum(gamma_s)
                
                # Update covariance with regularization
                diff = X_scaled - self.emission_means[s]
                cov = np.sum(
                    gamma_s[:, :, np.newaxis, np.newaxis] * 
                    np.einsum('...i,...j->...ij', diff, diff),
                    axis=(0, 1)
                ) / np.sum(gamma_s)
                
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
        
        # Preprocess data using the new pca_transform function
        X_scaled, time_indices, _ = pca_transform(X, timestamps, self.n_components, self.pca)
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
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
        
        # Handle input shape
        if len(X_last.shape) == 3:
            # If input is already in (n_samples, sequence_length, n_features) format
            X_last_scaled = X_last
        else:
            # If input is in (sequence_length, n_features) format, add batch dimension
            X_last_scaled = X_last.reshape(1, X_last.shape[0], X_last.shape[1])
        
        # Preprocess the data using the new pca_transform function
        X_last_scaled, _, _ = pca_transform(X_last_scaled, None, self.n_components, self.pca)
        
        # Take the last time step for state probability calculation
        X_last_scaled = X_last_scaled[0, -1, :]  # Shape: (n_components,)
        
        # Calculate emission probabilities for current state
        state_probs = np.zeros(self.n_states)
        for s in range(self.n_states):
            try:
                mvn = multivariate_normal(
                    mean=self.emission_means[s],
                    cov=self.emission_covars[s]
                )
                state_probs[s] = mvn.pdf(X_last_scaled)
            except:
                # Fallback to simple Gaussian approximation
                state_probs[s] = np.exp(-0.5 * np.sum(
                    (X_last_scaled - self.emission_means[s])**2
                ))
        
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
        
        # Return only the first emission (radiation) and its confidence intervals
        return forecasts[:, 0], (confidence_lower[:, 0], confidence_upper[:, 0])
    
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

    def plot_state_characteristics(self):
        """
        Visualize the characteristics of each hidden state.
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
        
        # Get state info
        state_info = self.get_state_characteristics()
        
        # Prepare data for plotting
        state_names = list(state_info.keys())
        n_states = len(state_names)
        n_emissions = self.n_emissions
        
        # Only plot if we have 1 or 2 emissions
        if n_emissions > 2:
            print("Cannot visualize state characteristics for more than 2 emission variables.")
            return None
        
        if n_emissions == 1:
            # For 1D emissions, plot means as bar chart
            plt.figure(figsize=(10, 6))
            emission_means = [state_info[state]['mean_emissions'][0] for state in state_names]
            
            plt.bar(state_names, emission_means)
            plt.title('Average Radiation by State')
            plt.xlabel('State')
            plt.ylabel('Radiation (W/m²)')
            plt.grid(True, axis='y', alpha=0.3)
            
        else:  # 2D emissions
            # For 2D emissions, create scatter plot
            plt.figure(figsize=(10, 8))
            
            x_vals = [state_info[state]['mean_emissions'][0] for state in state_names]
            y_vals = [state_info[state]['mean_emissions'][1] for state in state_names]
            
            plt.scatter(x_vals, y_vals, s=200, c=range(n_states), cmap='viridis')
            
            # Add state labels
            for i, state in enumerate(state_names):
                plt.annotate(
                    state, 
                    (x_vals[i], y_vals[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=12
                )
            
            plt.title('State Characteristics')
            plt.xlabel('Radiation (W/m²)')
            plt.ylabel('Temperature (°C)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return plt.gcf()

    def plot_forecast_horizon_accuracy(self, X_test, y_test, timestamps_test, horizons=[1, 3, 6, 12, 24]):
        """
        Plot prediction accuracy across different forecast horizons.
        
        Parameters:
        -----------
        X_test : array-like
            Test feature data
        y_test : array-like
            Test target values
        timestamps_test : array-like
            Test timestamps
        horizons : list of int
            Forecast horizons to evaluate (in hours)
        """
        rmse_values = []
        
        for horizon in horizons:
            predictions = []
            actuals = []
            
            # Make predictions at each horizon
            for i in range(len(X_test) - horizon):
                # Get current features and timestamp
                current_X = X_test[i]
                current_time = timestamps_test[i]
                
                # Forecast
                forecast, _ = self.forecast(
                    current_X, 
                    current_time, 
                    forecast_horizon=horizon
                )
                
                # Get the forecast at exactly the specified horizon
                predictions.append(forecast[horizon-1][0])  # First feature (radiation)
                
                # Get actual value at that horizon
                actuals.append(y_test[i + horizon])
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((np.array(actuals) - np.array(predictions))**2))
            rmse_values.append(rmse)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(horizons, rmse_values, 'bo-', linewidth=2)
        plt.title('Forecast Accuracy vs. Horizon')
        plt.xlabel('Forecast Horizon (hours)')
        plt.ylabel('RMSE')
        plt.grid(True, alpha=0.3)
        plt.xticks(horizons)
        
        return plt.gcf()

    def plot_transition_heatmaps(self, hours=None):
        """
        Create heatmaps of state transition probabilities for specified hours.
        
        Parameters:
        -----------
        hours : list of int, optional
            Specific hours to visualize (e.g., [8, 12, 16, 20])
            If None, uses [6, 12, 18, 0] (morning, noon, evening, midnight)
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
            
        if hours is None:
            hours = [6, 12, 18, 0]  # Default key times of day
        
        n_hours = len(hours)
        fig, axes = plt.subplots(1, n_hours, figsize=(5*n_hours, 4))
        
        for i, hour in enumerate(hours):
            ax = axes[i] if n_hours > 1 else axes
            
            # Create heatmap
            im = ax.imshow(self.transitions[hour], cmap='Blues', vmin=0, vmax=1)
            
            # Add probability values as text
            for s1 in range(self.n_states):
                for s2 in range(self.n_states):
                    text = ax.text(s2, s1, f"{self.transitions[hour, s1, s2]:.2f}",
                                  ha="center", va="center",
                                  color="white" if self.transitions[hour, s1, s2] > 0.5 else "black")
            
            # Labels
            ax.set_title(f"Hour {hour}")
            ax.set_xticks(range(self.n_states))
            ax.set_yticks(range(self.n_states))
            ax.set_xticklabels(self.state_names)
            ax.set_yticklabels(self.state_names)
            
            if i == 0:
                ax.set_ylabel("From State")
            if i == n_hours // 2:
                ax.set_xlabel("To State")
        
        # Add colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        fig.suptitle("State Transition Probabilities by Hour", fontsize=16, y=1.05)
        plt.tight_layout()
        
        return fig

    def plot_hidden_states(self, X_sample, timestamps=None):
        """
        Visualize the hidden states predicted by the TDMC model.
        
        Parameters:
        -----------
        X_sample : array-like
            Sample feature data
        timestamps : array-like, optional
            Corresponding timestamps
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
            
        # Predict hidden states
        states = self.predict_states(X_sample, timestamps)
        
        # Extract radiation values (assuming first column is radiation)
        radiation = X_sample[:, 0]  
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        
        # Plot radiation data
        if timestamps is not None:
            ax1.plot(timestamps, radiation, 'b-')
            ax2.scatter(timestamps, states, c=states, cmap='viridis', s=30)
        else:
            ax1.plot(radiation, 'b-')
            ax2.scatter(range(len(states)), states, c=states, cmap='viridis', s=30)
        
        # Labels and formatting
        ax1.set_title('Solar Irradiance')
        ax1.set_ylabel('Radiation (W/m²)')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Predicted Hidden States')
        ax2.set_ylabel('State')
        ax2.set_yticks(range(self.n_states))
        ax2.set_yticklabels(self.state_names)
        ax2.grid(True, alpha=0.3)
        
        if timestamps is not None:
            fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig

    def plot_prediction_vs_actual(self, y_true, y_pred, timestamps=None, confidence_intervals=None, title="Solar Irradiance Prediction"):
        """
        Plot predicted vs actual solar irradiance values.
        
        Parameters:
        -----------
        y_true : array-like
            Actual observed irradiance values
        y_pred : array-like
            Predicted irradiance values from the model
        timestamps : array-like, optional
            Timestamps for x-axis (if available)
        confidence_intervals : tuple of (lower, upper), optional
            Lower and upper bounds of prediction confidence intervals
        title : str
            Plot title
        """
        plt.figure(figsize=(15, 6))
        
        # Plot with timestamps if available
        if timestamps is not None:
            plt.plot(timestamps, y_true, 'b-', label='Actual Irradiance', alpha=0.7)
            plt.plot(timestamps, y_pred, 'r-', label='Predicted Irradiance', alpha=0.7)
            
            # Add confidence intervals if available
            if confidence_intervals is not None:
                lower_bound, upper_bound = confidence_intervals
                plt.fill_between(timestamps, lower_bound, upper_bound, 
                                color='r', alpha=0.2, label='95% Confidence Interval')
            
            plt.gcf().autofmt_xdate()  # Rotate date labels
        else:
            # Plot with simple indices
            plt.plot(y_true, 'b-', label='Actual Irradiance', alpha=0.7)
            plt.plot(y_pred, 'r-', label='Predicted Irradiance', alpha=0.7)
            
            # Add confidence intervals if available
            if confidence_intervals is not None:
                lower_bound, upper_bound = confidence_intervals
                plt.fill_between(range(len(y_true)), lower_bound, upper_bound, 
                                color='r', alpha=0.2, label='95% Confidence Interval')
        
        plt.title(title)
        plt.xlabel('Time' if timestamps is not None else 'Sample Index')
        plt.ylabel('Solar Irradiance (W/m²)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Calculate error metrics
        mse = np.mean((y_true - y_pred)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Calculate normalized metrics for non-zero actual values
        non_zero_mask = y_true > 0
        if np.sum(non_zero_mask) > 0:
            nrmse = rmse / np.mean(y_true[non_zero_mask])
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / 
                                  y_true[non_zero_mask])) * 100
        else:
            nrmse = np.nan
            mape = np.nan
        
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"NRMSE: {nrmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        return plt.gcf()

    def plot_transition_matrix(self, time_idx=None, figsize=(10, 8)):
        """
        Plot the transition matrix for a specific time slice or all time slices.
        
        Parameters:
        -----------
        time_idx : int or None
            If None, plot all transition matrices in a grid.
            If int, plot the transition matrix for that specific time slice.
        figsize : tuple
            Figure size for the plot.
        """
        if not self.trained:
            raise ValueError("Model not trained yet")
            
        if time_idx is not None:
            if not 0 <= time_idx < self.time_slices:
                raise ValueError(f"time_idx must be between 0 and {self.time_slices-1}")
            
            plt.figure(figsize=figsize)
            plt.imshow(self.transitions[time_idx], cmap='viridis', aspect='auto')
            plt.colorbar(label='Transition Probability')
            plt.title(f'Transition Matrix for Time Slice {time_idx}')
            plt.xlabel('Next State')
            plt.ylabel('Current State')
            plt.show()
        else:
            # Plot all transition matrices in a grid
            n_cols = min(4, self.time_slices)
            n_rows = (self.time_slices + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows / n_cols))
            axes = axes.flatten()
            
            for i in range(self.time_slices):
                im = axes[i].imshow(self.transitions[i], cmap='viridis', aspect='auto')
                axes[i].set_title(f'Time Slice {i}')
                axes[i].set_xlabel('Next State')
                axes[i].set_ylabel('Current State')
            
            # Remove empty subplots
            for i in range(self.time_slices, len(axes)):
                fig.delaxes(axes[i])
            
            # Add colorbar
            fig.colorbar(im, ax=axes, label='Transition Probability')
            plt.tight_layout()
            plt.show()

    def inverse_transform(self, X_transformed):
        """
        Transform PCA-reduced data back to original feature space.
        
        Parameters:
        -----------
        X_transformed : array-like
            Data in PCA space
            
        Returns:
        --------
        X_original : array-like
            Data in original feature space
        """
        if not hasattr(self.pca, 'components_'):
            raise ValueError("PCA has not been fitted yet")
            
        # Reshape if necessary
        if len(X_transformed.shape) == 3:
            n_samples, sequence_length, n_components = X_transformed.shape
            X_reshaped = X_transformed.reshape(-1, n_components)
            X_original = self.pca.inverse_transform(X_reshaped)
            return X_original.reshape(n_samples, sequence_length, -1)
        else:
            return self.pca.inverse_transform(X_transformed)