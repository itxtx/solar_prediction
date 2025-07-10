import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from typing import List, Tuple, Optional, Dict, Any, Union

# Import centralized configuration
from .config import get_config


# =============================================================================
# Numerical Stability and Logging Helper Functions
# =============================================================================

def safe_normalise(arr: np.ndarray, axis: int = -1, min_val: float = 1e-300) -> np.ndarray:
    """
    Safely normalize an array to ensure probabilities sum to 1.
    
    Parameters:
    -----------
    arr : np.ndarray
        Array to normalize
    axis : int
        Axis along which to normalize
    min_val : float
        Minimum value to use for numerical stability
        
    Returns:
    --------
    np.ndarray
        Normalized array
    """
    # Ensure minimum values for numerical stability
    arr_safe = np.maximum(arr, min_val)
    
    # Calculate sum along specified axis
    arr_sum = np.sum(arr_safe, axis=axis, keepdims=True)
    
    # Avoid division by zero
    arr_sum = np.maximum(arr_sum, min_val)
    
    # Normalize
    normalized = arr_safe / arr_sum
    
    # Final safety check
    normalized = np.maximum(normalized, min_val)
    
    # Re-normalize to ensure exact sum to 1 after flooring
    if axis == -1 or axis == arr.ndim - 1:
        final_sum = np.sum(normalized, axis=axis, keepdims=True)
        normalized = normalized / np.maximum(final_sum, min_val)
    
    return normalized


def floor_prob(prob: Union[float, np.ndarray], min_prob: float = 1e-300) -> Union[float, np.ndarray]:
    """
    Apply a floor to probabilities to avoid numerical underflow.
    
    Parameters:
    -----------
    prob : Union[float, np.ndarray]
        Probability value(s) to floor
    min_prob : float
        Minimum probability value
        
    Returns:
    --------
    Union[float, np.ndarray]
        Floored probability value(s)
    """
    return np.maximum(prob, min_prob)


def _get_logger(verbose: bool = False) -> logging.Logger:
    """
    Get configured logger for TDMC operations.
    
    Parameters:
    -----------
    verbose : bool
        Whether to enable verbose logging
        
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger('TDMC')
    
    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Set level based on verbose flag
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    return logger


def _fix_eigenvalues(matrix: np.ndarray, min_eigenvalue: float = 1e-9, 
                     regularization: float = 1e-6, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """
    Fix eigenvalues of a matrix to ensure positive definiteness.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Matrix to fix (typically covariance matrix)
    min_eigenvalue : float
        Minimum eigenvalue threshold
    regularization : float
        Regularization to add to diagonal
    logger : Optional[logging.Logger]
        Logger for reporting fixes
        
    Returns:
    --------
    np.ndarray
        Fixed matrix with positive eigenvalues
    """
    try:
        eigenvals = np.linalg.eigvals(matrix)
        min_eig = np.min(np.real(eigenvals))
        
        if min_eig <= min_eigenvalue:
            # Add regularization to make positive definite
            offset = -min_eig + min_eigenvalue + regularization
            fixed_matrix = matrix + offset * np.eye(matrix.shape[0])
            
            if logger:
                logger.debug(f"Fixed eigenvalues: min_eig={min_eig:.2e}, added offset={offset:.2e}")
            
            return fixed_matrix
        else:
            # Just add standard regularization
            return matrix + regularization * np.eye(matrix.shape[0])
            
    except Exception as e:
        if logger:
            logger.warning(f"Eigenvalue fix failed: {e}, using identity matrix")
        return np.eye(matrix.shape[0]) * regularization

class SolarTDMC:
    """
    Time-dynamic Markov Chain (TDMC) implementation for solar irradiance prediction.
    This is a specialized Hidden Markov Model with time-dependent transition probabilities.
    """

    def __init__(self, n_states: Optional[int] = None, n_emissions: Optional[int] = None, time_slices: Optional[int] = None):
        """
        Initialize the TDMC model.

        Parameters:
        -----------
        n_states : int
            Number of hidden states in the model.
        n_emissions : int
            Number of observable emission variables (e.g., irradiance, temperature).
        time_slices : int
            Number of time slices for the day (e.g., 24 for hourly).
        """
        # Use centralized configuration for defaults
        config = get_config()
        tdmc_config = config.models.tdmc
        
        self.n_states = n_states if n_states is not None else tdmc_config.n_states
        self.n_emissions = n_emissions if n_emissions is not None else tdmc_config.n_emissions
        self.time_slices = time_slices if time_slices is not None else tdmc_config.time_slices

        self.transitions = np.zeros((self.time_slices, self.n_states, self.n_states))
        for t in range(self.time_slices):
            self.transitions[t] = np.ones((self.n_states, self.n_states)) / self.n_states

        self.emission_means = np.zeros((self.n_states, self.n_emissions))
        self.emission_covars = np.zeros((self.n_states, self.n_emissions, self.n_emissions))
        for s in range(self.n_states):
            self.emission_covars[s] = np.eye(self.n_emissions)

        self.initial_probs = np.ones(self.n_states) / self.n_states
        self.scaler = StandardScaler()
        self.trained = False
        self.state_names: List[str] = [f"State_{i}" for i in range(self.n_states)]

    def _preprocess_data(self, X: np.ndarray, 
                         timestamps: Optional[Union[np.ndarray, pd.Series]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess input data for the model.
        """
        if not self.trained:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        if timestamps is not None:
            if isinstance(timestamps.iloc[0] if isinstance(timestamps, pd.Series) else timestamps[0], (pd.Timestamp, np.datetime64)):
                time_indices = pd.DatetimeIndex(timestamps).hour.to_numpy()
            else: # Assume timestamps are already hour indicators (0-23)
                time_indices = np.array(timestamps)
            # Ensure time_indices are within [0, time_slices-1]
            time_indices = time_indices % self.time_slices
        else:
            time_indices = np.zeros(len(X), dtype=int)
            
        return X_scaled, time_indices

    def fit(self, X: np.ndarray, timestamps: Optional[Union[np.ndarray, pd.Series]] = None, 
            max_iter: Optional[int] = None, tol: Optional[float] = None, state_names: Optional[List[str]] = None):
        """
        Fit the TDMC model to data using the Baum-Welch algorithm.
        """
        config = get_config()
        tdmc_config = config.models.tdmc
        
        # Input validation
        if timestamps is not None and len(X) != len(timestamps):
            raise ValueError(f"X and timestamps must have the same length. Got X: {len(X)}, timestamps: {len(timestamps)}")
        if X.shape[1] != self.n_emissions:
            raise ValueError(f"X has {X.shape[1]} features, but model expects {self.n_emissions} emissions.")
        
        # Initialize logger based on config
        logger = _get_logger(tdmc_config.verbose_logging) if tdmc_config.verbose_logging else None
        
        # Use centralized config for defaults
        max_iter = max_iter if max_iter is not None else tdmc_config.max_iter
        tol = tol if tol is not None else tdmc_config.tolerance
        
        if logger:
            logger.info(f"Starting TDMC fit with {self.n_states} states, {self.n_emissions} emissions")
            logger.info(f"Training parameters: max_iter={max_iter}, tolerance={tol}")
        
        X_scaled, time_indices = self._preprocess_data(X, timestamps)
        
        if logger:
            logger.info(f"Data preprocessed: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
            logger.debug(f"Time indices range: [{np.min(time_indices)}, {np.max(time_indices)}]")
        
        self._initialize_parameters(X_scaled, time_indices)
        
        if logger:
            logger.info("Parameters initialized using K-means clustering")
        
        final_log_likelihood = self._baum_welch_update(X_scaled, time_indices, max_iter, tol)
        
        if logger:
            logger.info(f"Training completed with final log-likelihood: {final_log_likelihood:.4f}")

        if state_names is not None and len(state_names) == self.n_states:
            self.state_names = state_names
            if logger:
                logger.info(f"Updated state names: {self.state_names}")
        
        self.trained = True
        return self

    def _initialize_parameters(self, X_scaled: np.ndarray, time_indices: np.ndarray):
        """Initialize model parameters using K-means clustering."""
        config = get_config()
        tdmc_config = config.models.tdmc
        
        kmeans = KMeans(n_clusters=self.n_states, random_state=tdmc_config.random_state, n_init=tdmc_config.kmeans_n_init)
        state_assignments = kmeans.fit_predict(X_scaled)

        for s in range(self.n_states):
            state_data = X_scaled[state_assignments == s]
            if len(state_data) > 1: # Need at least 2 samples for covariance
                self.emission_means[s] = np.mean(state_data, axis=0)
                cov = np.cov(state_data.T)
                min_eig = np.min(np.real(np.linalg.eigvals(cov)))
                config = get_config()
                tdmc_config = config.models.tdmc
                if min_eig <= 0: # Ensure positive definiteness
                    cov += (-min_eig + tdmc_config.min_eigenvalue_threshold) * np.eye(self.n_emissions) # Add offset to make eigenvalues positive
                self.emission_covars[s] = cov + tdmc_config.covariance_regularization * np.eye(self.n_emissions) # Regularization
            elif len(state_data) == 1:
                 self.emission_means[s] = state_data[0]
                 self.emission_covars[s] = np.eye(self.n_emissions) * 1e-4 # Default small covariance
            else: # If no data for this state, use global statistics (or default)
                self.emission_means[s] = np.mean(X_scaled, axis=0)
                self.emission_covars[s] = np.cov(X_scaled.T) + 1e-4 * np.eye(self.n_emissions)
                if np.any(np.isnan(self.emission_means[s])): # Fallback if X_scaled is empty
                    self.emission_means[s] = np.zeros(self.n_emissions)
                if np.any(np.isnan(self.emission_covars[s])) or self.emission_covars[s].shape != (self.n_emissions, self.n_emissions):
                     self.emission_covars[s] = np.eye(self.n_emissions) * 1e-4


        self.transitions = np.ones((self.time_slices, self.n_states, self.n_states)) * 1e-6 # Smoothing prior
        for t in range(self.time_slices):
            time_slice_mask = (time_indices == t)
            if np.sum(time_slice_mask) > 1:
                states_t = state_assignments[time_slice_mask]
                for i in range(len(states_t) - 1):
                    s1, s2 = states_t[i], states_t[i+1]
                    self.transitions[t, s1, s2] += 1
            
            row_sums = self.transitions[t].sum(axis=1, keepdims=True)
            self.transitions[t] = self.transitions[t] / np.maximum(row_sums, 1e-300) # Normalize with floor

        initial_states_counts = np.zeros(self.n_states) + 1e-6 # Smoothing prior
        # Count first state of sequences or first state in each time slice if data is continuous
        # For simplicity, using K-means assignments for initial distribution estimate
        unique_first_time_indices_mask = np.concatenate(([True], time_indices[1:] != time_indices[:-1]))
        first_states_in_sequences = state_assignments[unique_first_time_indices_mask]

        if len(first_states_in_sequences) > 0:
            for state in first_states_in_sequences:
                initial_states_counts[state] +=1
        
        self.initial_probs = initial_states_counts / np.sum(initial_states_counts)


    def _forward_backward(self, X_scaled: np.ndarray, 
                          time_indices: np.ndarray, logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform forward-backward algorithm for the TDMC."""
        config = get_config()
        tdmc_config = config.models.tdmc
        
        n_samples = len(X_scaled)
        emission_probs = np.zeros((n_samples, self.n_states))

        # Calculate emission probabilities with numerical stability
        for s in range(self.n_states):
            try:
                # Fix eigenvalues for numerical stability
                cov = _fix_eigenvalues(
                    self.emission_covars[s], 
                    tdmc_config.min_eigenvalue_threshold,
                    tdmc_config.covariance_regularization,
                    logger
                )
                
                mvn = multivariate_normal(mean=self.emission_means[s], cov=cov, allow_singular=False)
                emission_probs[:, s] = mvn.pdf(X_scaled)
            except Exception as e: # Fallback if mvn fails
                if logger:
                    logger.warning(f"MVN PDF failed for state {s}: {e}. Using fallback.")
                # Fallback: simple Gaussian-like calculation
                diff_sq = (X_scaled - self.emission_means[s])**2
                precision = np.diag(1.0 / np.maximum(np.diag(self.emission_covars[s]), 1e-9)) 
                emission_probs[:, s] = np.exp(-0.5 * np.sum(diff_sq @ precision, axis=1))

        # Apply probability floor for numerical stability
        emission_probs = floor_prob(emission_probs, tdmc_config.probability_floor)
        
        if logger:
            logger.debug(f"Emission probabilities - min: {np.min(emission_probs):.2e}, max: {np.max(emission_probs):.2e}")

        alpha = np.zeros((n_samples, self.n_states))
        beta = np.zeros((n_samples, self.n_states))
        scale = np.zeros(n_samples)

        # Forward pass with safe normalization
        alpha[0] = floor_prob(self.initial_probs * emission_probs[0], tdmc_config.min_probability)
        scale[0] = np.sum(alpha[0])
        alpha[0] = safe_normalise(alpha[0].reshape(1, -1), min_val=tdmc_config.min_probability).flatten()

        for t in range(1, n_samples):
            current_time_slice_transitions = self.transitions[time_indices[t-1]] # Transitions from t-1 to t
            alpha[t] = (alpha[t-1] @ current_time_slice_transitions) * emission_probs[t]
            alpha[t] = floor_prob(alpha[t], tdmc_config.min_probability)
            scale[t] = np.sum(alpha[t])
            alpha[t] = safe_normalise(alpha[t].reshape(1, -1), min_val=tdmc_config.min_probability).flatten()

        # Backward pass with safe normalization
        beta[-1] = 1.0
        for t in range(n_samples - 2, -1, -1):
            next_time_slice_transitions = self.transitions[time_indices[t]] # Transitions from t to t+1
            beta[t] = (next_time_slice_transitions @ (emission_probs[t+1] * beta[t+1]).T).T
            beta[t] = floor_prob(beta[t], tdmc_config.min_probability)
            if scale[t+1] > 0: 
                beta[t] /= scale[t+1]
            beta[t] = floor_prob(beta[t], tdmc_config.min_probability)
            
        if logger:
            logger.debug(f"Forward-backward complete - scale range: [{np.min(scale):.2e}, {np.max(scale):.2e}]")
            
        return alpha, beta, scale, emission_probs

    def _baum_welch_update(self, X_scaled: np.ndarray, time_indices: np.ndarray, 
                           max_iter: int, tol: float) -> float:
        """Perform Baum-Welch algorithm to update model parameters."""
        config = get_config()
        tdmc_config = config.models.tdmc
        
        # Initialize logger based on config
        logger = _get_logger(tdmc_config.verbose_logging) if tdmc_config.verbose_logging else None
        
        prev_log_likelihood = -np.inf
        log_likelihood = 0.0
        
        if logger:
            logger.info(f"Starting Baum-Welch algorithm - max_iter: {max_iter}, tolerance: {tol}")

        for iteration in range(max_iter):
            alpha, beta, scale, emission_probs = self._forward_backward(X_scaled, time_indices, logger)
            
            # Safe log-likelihood calculation with floor_prob
            safe_scale = floor_prob(scale, tdmc_config.min_probability)
            current_log_likelihood = np.sum(np.log(safe_scale))
            
            # Check convergence
            if abs(current_log_likelihood - prev_log_likelihood) < tol and iteration > 0:
                log_likelihood = current_log_likelihood
                if logger:
                    logger.info(f"Converged at iteration {iteration+1}, Log-Likelihood: {log_likelihood:.4f}")
                break
                
            prev_log_likelihood = current_log_likelihood
            log_likelihood = current_log_likelihood
            
            # Log progress based on configuration
            if logger and (iteration % tdmc_config.log_likelihood_every_n_iter == 0):
                logger.info(f"Iteration {iteration+1}/{max_iter}, Log-Likelihood: {log_likelihood:.4f}")

            n_samples = len(X_scaled)
            gamma = alpha * beta 
            gamma = gamma / np.maximum(np.sum(gamma, axis=1, keepdims=True), 1e-300) # Normalize gamma

            # Update initial state distribution with safe normalization
            self.initial_probs = safe_normalise(gamma[0].reshape(1, -1), min_val=tdmc_config.min_probability).flatten()

            # Update transition matrices - Full ξ/γ accumulation per time slice
            xi_sum = np.zeros((self.time_slices, self.n_states, self.n_states))
            gamma_sum = np.zeros((self.time_slices, self.n_states))
            
            # Compute ξ_t(i,j) using scaled α/β and accumulate per time slice
            for t_idx in range(n_samples - 1):
                ti = time_indices[t_idx]  # Time slice for transition from t_idx to t_idx+1
                trans_prob_ti = self.transitions[ti]  # A_k for this time slice
                
                # Calculate ξ_t(i,j) = P(q_t=i, q_{t+1}=j | O, λ)
                # Using scaled alpha and beta:
                # ξ_t(i,j) = α_t(i) * A(i,j) * B_j(O_{t+1}) * β_{t+1}(j)
                xi_t = np.zeros((self.n_states, self.n_states))
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi_t[i, j] = (alpha[t_idx, i] * 
                                     trans_prob_ti[i, j] * 
                                     emission_probs[t_idx + 1, j] * 
                                     beta[t_idx + 1, j])
                
                # Normalize ξ_t to ensure proper probabilities
                xi_t_sum = np.sum(xi_t)
                if xi_t_sum > 0:
                    xi_t /= xi_t_sum
                
                # Accumulate ξ_t(i,j) for this time slice
                xi_sum[ti] += xi_t
                
                # Accumulate γ_t(i) for this time slice
                gamma_sum[ti] += gamma[t_idx]
            
            # Update transition matrices using accumulated ξ and γ
            for ts in range(self.time_slices):
                # Get smoothing prior from config
                config = get_config()
                tdmc_config = config.models.tdmc
                prior = tdmc_config.transition_smoothing_prior
                min_prob = tdmc_config.min_probability
                
                for i in range(self.n_states):
                    # Calculate denominator: sum of γ_t(i) for this time slice + prior
                    denominator = np.maximum(gamma_sum[ts, i], min_prob) + self.n_states * prior
                    
                    # Update each transition probability A[ts, i, j]
                    for j in range(self.n_states):
                        # Numerator: sum of ξ_t(i,j) for this time slice + prior
                        numerator = xi_sum[ts, i, j] + prior
                        self.transitions[ts, i, j] = numerator / denominator
                    
                    # Safe row normalization to ensure probabilities sum to 1
                    self.transitions[ts, i, :] = safe_normalise(
                        self.transitions[ts, i, :].reshape(1, -1), 
                        min_val=min_prob
                    ).flatten()

            # Update emission parameters with numerical stability
            for s in range(self.n_states):
                gamma_s = gamma[:, s]
                gamma_s_sum = np.sum(gamma_s)
                gamma_s_sum = np.maximum(gamma_s_sum, tdmc_config.min_probability) # Avoid division by zero

                # Update mean
                current_mean = np.sum(gamma_s[:, np.newaxis] * X_scaled, axis=0) / gamma_s_sum
                if not np.any(np.isnan(current_mean)): 
                    self.emission_means[s] = current_mean
                    if logger:
                        logger.debug(f"Updated mean for state {s}: {current_mean}")
                
                # Update covariance with numerical stability
                diff = X_scaled - self.emission_means[s]
                cov_s = np.dot((gamma_s[:, np.newaxis] * diff).T, diff) / gamma_s_sum
                
                # Fix eigenvalues for numerical stability
                cov_s = _fix_eigenvalues(
                    cov_s, 
                    tdmc_config.min_eigenvalue_threshold,
                    tdmc_config.covariance_regularization,
                    logger
                )
                
                if not np.any(np.isnan(cov_s)): 
                    self.emission_covars[s] = cov_s
                    if logger:
                        min_eig = np.min(np.linalg.eigvals(cov_s))
                        logger.debug(f"Updated covariance for state {s}, min eigenvalue: {min_eig:.2e}")
        
        if logger:
            logger.info(f"Finished training. Final Log-Likelihood: {log_likelihood:.4f}")
        return log_likelihood

    def predict_states(self, X: np.ndarray, 
                       timestamps: Optional[Union[np.ndarray, pd.Series]] = None) -> np.ndarray:
        """
        Predict the most likely hidden state sequence using Viterbi algorithm.
        """
        if not self.trained:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        config = get_config()
        tdmc_config = config.models.tdmc
        
        # Initialize logger if verbose logging is enabled
        logger = _get_logger(tdmc_config.verbose_logging) if tdmc_config.verbose_logging else None
        
        X_scaled, time_indices = self._preprocess_data(X, timestamps)
        n_samples = len(X_scaled)
        
        if logger:
            logger.debug(f"Predicting states for {n_samples} samples")
        
        emission_probs = np.zeros((n_samples, self.n_states))
        for s in range(self.n_states):
            try:
                # Use eigenvalue fixing for numerical stability
                cov = _fix_eigenvalues(
                    self.emission_covars[s], 
                    tdmc_config.min_eigenvalue_threshold,
                    tdmc_config.covariance_regularization,
                    logger
                )
                mvn = multivariate_normal(mean=self.emission_means[s], cov=cov, allow_singular=False)
                emission_probs[:, s] = mvn.pdf(X_scaled)
            except Exception: # Fallback
                if logger:
                    logger.warning(f"MVN PDF failed for state {s} in prediction. Using fallback.")
                precision = np.diag(1.0 / np.maximum(np.diag(self.emission_covars[s]), 1e-9))
                emission_probs[:, s] = np.exp(-0.5 * np.sum((X_scaled - self.emission_means[s])**2 @ precision, axis=1))

        # Apply probability floor for numerical stability
        emission_probs = floor_prob(emission_probs, tdmc_config.min_probability) # Avoid log(0) later
        
        if logger:
            logger.debug(f"Emission probabilities computed - min: {np.min(emission_probs):.2e}, max: {np.max(emission_probs):.2e}")
        
        viterbi_probs = np.zeros((n_samples, self.n_states))
        backpointers = np.zeros((n_samples, self.n_states), dtype=int)
        
        # Initialization step
        viterbi_probs[0] = np.log(self.initial_probs) + np.log(emission_probs[0])
        
        # Recursion step
        for t in range(1, n_samples):
            current_time_slice = time_indices[t-1] # Transition from t-1 to t uses A[time_indices[t-1]]
            for s_curr in range(self.n_states):
                trans_probs_to_scurr = np.log(self.transitions[current_time_slice, :, s_curr])
                max_prev_prob = viterbi_probs[t-1] + trans_probs_to_scurr
                
                viterbi_probs[t, s_curr] = np.max(max_prev_prob) + np.log(emission_probs[t, s_curr])
                backpointers[t, s_curr] = np.argmax(max_prev_prob)
                
        # Termination & Path backtracking
        states = np.zeros(n_samples, dtype=int)
        states[-1] = np.argmax(viterbi_probs[-1])
        for t in range(n_samples - 2, -1, -1):
            states[t] = backpointers[t + 1, states[t + 1]]
        
        if logger:
            state_counts = np.bincount(states, minlength=self.n_states)
            logger.debug(f"Predicted state distribution: {dict(zip(self.state_names, state_counts))}")
            
        return states

    def forecast(self, X_last: np.ndarray, timestamp_last_hour: int, 
                 forecast_horizon: int, 
                 weather_forecasts: Optional[np.ndarray] = None,
                 external_forecast_weight: float = 0.3) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Forecast future emissions.
        X_last: Last observed emission values (1D array, original scale).
        timestamp_last_hour: The hour (0-23) of the last observation.
        forecast_horizon: Number of steps to forecast ahead.
        weather_forecasts: Optional external forecasts (e.g., temperature) for the horizon (original scale).
        external_forecast_weight: Weight for blending model forecast with external forecasts (0 to 1).
        """
        if not self.trained:
            raise ValueError("Model not trained yet. Call fit() first.")

        X_last_scaled = self.scaler.transform(X_last.reshape(1, -1))

        # Determine current state probabilities based on last observation
        current_emission_probs = np.zeros(self.n_states)
        for s in range(self.n_states):
            try:
                cov = self.emission_covars[s]
                if np.min(np.linalg.eigvalsh(cov)) <= 1e-9: cov += 1e-6 * np.eye(self.n_emissions)
                mvn = multivariate_normal(mean=self.emission_means[s], cov=cov, allow_singular=False)
                current_emission_probs[s] = mvn.pdf(X_last_scaled.reshape(-1)) # Ensure 1D for pdf
            except Exception:
                precision = np.diag(1.0 / np.maximum(np.diag(self.emission_covars[s]), 1e-9))
                current_emission_probs[s] = np.exp(-0.5 * np.sum((X_last_scaled.reshape(-1) - self.emission_means[s])**2 @ precision))

        current_emission_probs = np.maximum(current_emission_probs, 1e-300)
        state_distribution = current_emission_probs / np.sum(current_emission_probs) # P(q_T | O_T)

        forecasts_scaled = np.zeros((forecast_horizon, self.n_emissions))
        confidence_lower_scaled = np.zeros((forecast_horizon, self.n_emissions))
        confidence_upper_scaled = np.zeros((forecast_horizon, self.n_emissions))

        future_time_indices = [(timestamp_last_hour + i + 1) % self.time_slices for i in range(forecast_horizon)]

        for step in range(forecast_horizon):
            time_idx_for_transition = future_time_indices[step-1] if step > 0 else timestamp_last_hour
            
            # Update state distribution using transition matrix for the current time slice
            # P(q_{T+h+1} | O_T) = P(q_{T+h} | O_T) @ A[time_slice_of_{T+h}]
            state_distribution = state_distribution @ self.transitions[time_idx_for_transition]
            state_distribution = np.maximum(state_distribution, 1e-300)
            state_distribution /= np.sum(state_distribution)


            # Calculate expected emission (mean of the mixture)
            step_forecast_mean_scaled = np.zeros(self.n_emissions)
            for s in range(self.n_states):
                step_forecast_mean_scaled += state_distribution[s] * self.emission_means[s]
            
            # Calculate variance of the mixture (Law of Total Variance)
            # Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
            # Var(Emission | O_T) = E[Var(Emission | q_{T+h+1}, O_T)] + Var(E[Emission | q_{T+h+1}, O_T])
            expected_var_within_states = np.zeros(self.n_emissions)
            var_of_expected_values = np.zeros(self.n_emissions)
            
            for s in range(self.n_states):
                expected_var_within_states += state_distribution[s] * np.diag(self.emission_covars[s])
                var_of_expected_values += state_distribution[s] * (self.emission_means[s] - step_forecast_mean_scaled)**2
            
            step_forecast_var_scaled = expected_var_within_states + var_of_expected_values
            step_forecast_std_scaled = np.sqrt(np.maximum(step_forecast_var_scaled, 1e-9)) # Avoid sqrt of negative/small

            forecasts_scaled[step] = step_forecast_mean_scaled

            # Incorporate external weather forecasts if available (on scaled values if necessary, or scale them)
            if weather_forecasts is not None and step < len(weather_forecasts):
                # Assuming weather_forecasts are on original scale
                # For simplicity, blend after inverse transform, or scale external forecast here
                # Here, we will blend the scaled model forecast with a scaled external forecast
                # This example assumes weather_forecasts has same number of features as n_emissions
                # And that the order matches. This part needs careful alignment.
                # For now, a simpler approach: blend the final original scale forecasts.
                pass # Blending will be done after inverse transform for simplicity

            confidence_lower_scaled[step] = forecasts_scaled[step] - 1.96 * step_forecast_std_scaled
            confidence_upper_scaled[step] = forecasts_scaled[step] + 1.96 * step_forecast_std_scaled

        # Inverse transform to original scale
        forecasts_orig = self.scaler.inverse_transform(forecasts_scaled)
        confidence_lower_orig = self.scaler.inverse_transform(confidence_lower_scaled)
        confidence_upper_orig = self.scaler.inverse_transform(confidence_upper_scaled)

        # Blend with external forecasts (if provided, on original scale)
        if weather_forecasts is not None:
            for step in range(min(forecast_horizon, len(weather_forecasts))):
                # Ensure weather_forecasts[step] is a 1D array of size n_emissions
                external_pred = np.array(weather_forecasts[step]).flatten()
                if len(external_pred) == self.n_emissions:
                     forecasts_orig[step] = (1 - external_forecast_weight) * forecasts_orig[step] + \
                                           external_forecast_weight * external_pred
                # Note: CI blending is more complex and not done here. CIs reflect model uncertainty.

        return forecasts_orig, (confidence_lower_orig, confidence_upper_orig)

    def get_state_characteristics(self) -> Dict[str, Dict[str, Any]]:
        if not self.trained:
            raise ValueError("Model not trained yet.")
        
        state_info = {}
        orig_means = self.scaler.inverse_transform(self.emission_means)
        
        for s_idx in range(self.n_states):
            # Calculate original scale covariances
            # Cov_orig = Cov_scaled * outer(scale, scale)
            scaled_covar = self.emission_covars[s_idx]
            scaler_scales = self.scaler.scale_ if self.scaler.scale_ is not None else np.ones(self.n_emissions)
            orig_covar_diag = np.diag(scaled_covar) * (scaler_scales**2) # for diagonal elements
            # For full matrix: orig_covar = scaled_covar * np.outer(scaler_scales, scaler_scales)
            # This simple diag is an approximation if only diagonal matters for display or if scaler is feature-wise
            
            state_info[self.state_names[s_idx]] = {
                'mean_emissions_original_scale': orig_means[s_idx],
                'variance_emissions_original_scale_diag': orig_covar_diag, # Simplified
                'most_likely_next_states': {
                    f'hour_{t}': sorted([(self.state_names[next_s], self.transitions[t, s_idx, next_s]) 
                                         for next_s in range(self.n_states)], key=lambda x: x[1], reverse=True)[:3]
                    for t in range(self.time_slices)
                }
            }
        return state_info

    def plot_state_transitions(self, time_slice: int = 12, ax: Optional[plt.Axes] = None) -> plt.Figure:
        if not self.trained:
            raise ValueError("Model not trained yet.")
        if not (0 <= time_slice < self.time_slices):
            raise ValueError(f"time_slice must be between 0 and {self.time_slices-1}")

        if ax is None:
            fig, ax_plot = plt.subplots(figsize=(10, 8))
        else:
            ax_plot = ax
            fig = ax_plot.figure

        im = ax_plot.imshow(self.transitions[time_slice], cmap='Blues', vmin=0, vmax=1)
        ax_plot.set_xticks(np.arange(self.n_states))
        ax_plot.set_yticks(np.arange(self.n_states))
        ax_plot.set_xticklabels(self.state_names)
        ax_plot.set_yticklabels(self.state_names)
        plt.setp(ax_plot.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        if fig.axes[-1].get_label() != '<colorbar>': # Avoid multiple colorbars if reusing ax
             fig.colorbar(im, ax=ax_plot, label='Transition Probability')

        for i in range(self.n_states):
            for j in range(self.n_states):
                prob = self.transitions[time_slice, i, j]
                ax_plot.text(j, i, f"{prob:.2f}",
                             ha="center", va="center", 
                             color="white" if prob > 0.5 else "black")
        
        ax_plot.set_title(f"Transition Probabilities at Time Slice {time_slice} (Hour {time_slice})")
        ax_plot.set_xlabel("To State")
        ax_plot.set_ylabel("From State")
        if ax is None: fig.tight_layout()
        return fig

    def plot_emissions_by_state(self, standardized: bool = False, ax: Optional[plt.Axes] = None) -> plt.Figure:
        if not self.trained:
            raise ValueError("Model not trained yet.")
        if self.n_emissions > 2:
            print("Plotting only supported for 1 or 2 emission dimensions.")
            return plt.figure() # Return empty figure

        if ax is None:
            fig, ax_plot = plt.subplots(figsize=(10, 6 if self.n_emissions == 1 else 8))
        else:
            ax_plot = ax
            fig = ax_plot.figure
            
        means_to_plot = self.emission_means
        covars_to_plot = self.emission_covars
        x_label = "Emission Value"
        y_label = "Probability Density"
        title_suffix = "(Standardized)"

        if not standardized:
            title_suffix = "(Original Scale)"
            means_to_plot = self.scaler.inverse_transform(self.emission_means)
            covars_to_plot = np.zeros_like(self.emission_covars)
            scaler_scales = self.scaler.scale_ if self.scaler.scale_ is not None else np.ones(self.n_emissions)
            for s_idx in range(self.n_states):
                covars_to_plot[s_idx] = self.emission_covars[s_idx] * np.outer(scaler_scales, scaler_scales)

        if self.n_emissions == 1:
            min_val = np.min(means_to_plot[:, 0] - 3 * np.sqrt(np.abs(covars_to_plot[:, 0, 0])))
            max_val = np.max(means_to_plot[:, 0] + 3 * np.sqrt(np.abs(covars_to_plot[:, 0, 0])))
            x_grid = np.linspace(min_val, max_val, 500)
            for s in range(self.n_states):
                try:
                    cov = covars_to_plot[s,0,0]
                    if cov <= 1e-9: cov = 1e-9 # ensure positive for pdf
                    y_pdf = multivariate_normal.pdf(x_grid, mean=means_to_plot[s, 0], cov=cov)
                    ax_plot.plot(x_grid, y_pdf, label=self.state_names[s])
                except Exception as e:
                    print(f"Could not plot 1D emission for state {s} ({self.state_names[s]}): {e}")
            ax_plot.set_xlabel(f"{x_label} {title_suffix}")
            ax_plot.set_ylabel(y_label)

        elif self.n_emissions == 2:
            x_min = np.min(means_to_plot[:, 0] - 3 * np.sqrt(np.abs(covars_to_plot[:, 0, 0])))
            x_max = np.max(means_to_plot[:, 0] + 3 * np.sqrt(np.abs(covars_to_plot[:, 0, 0])))
            y_min = np.min(means_to_plot[:, 1] - 3 * np.sqrt(np.abs(covars_to_plot[:, 1, 1])))
            y_max = np.max(means_to_plot[:, 1] + 3 * np.sqrt(np.abs(covars_to_plot[:, 1, 1])))
            
            x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            pos = np.dstack((x_grid, y_grid))
            
            for s in range(self.n_states):
                try:
                    cov = covars_to_plot[s]
                    if np.min(np.linalg.eigvalsh(cov)) <= 1e-9 : cov += 1e-6 * np.eye(self.n_emissions) # Ensure positive definite
                    rv = multivariate_normal(mean=means_to_plot[s], cov=cov, allow_singular=False)
                    ax_plot.contour(x_grid, y_grid, rv.pdf(pos), levels=3, alpha=0.7, 
                                   colors=[plt.cm.viridis(s / max(1, self.n_states-1))]) # Use distinct colors
                    ax_plot.scatter(means_to_plot[s, 0], means_to_plot[s, 1], marker='x', s=100, 
                                   label=self.state_names[s] if not ax_plot.legend_ else None, # Avoid duplicate labels
                                   color=plt.cm.viridis(s / max(1, self.n_states-1)))
                except Exception as e:
                     print(f"Could not plot 2D emission for state {s} ({self.state_names[s]}): {e}")
            ax_plot.set_xlabel(f"Emission 1 {title_suffix} (e.g., Radiation)")
            ax_plot.set_ylabel(f"Emission 2 {title_suffix} (e.g., Temperature)")
        
        ax_plot.set_title(f"Emission Distributions by State {title_suffix}")
        ax_plot.legend()
        if ax is None: fig.tight_layout()
        return fig

    def save_model(self, filepath: str, hp=None, train_cfg=None, history=None, metrics=None, use_enhanced=True):
        """Save model parameters using enhanced checkpointing or legacy format.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        hp : dict, optional
            Hyperparameters dictionary (for enhanced checkpointing)
        train_cfg : dict, optional
            Training configuration dictionary (for enhanced checkpointing)
        history : dict, optional
            Training history dictionary (for enhanced checkpointing)
        metrics : dict, optional
            Final metrics dictionary (for enhanced checkpointing)
        use_enhanced : bool
            Whether to use enhanced checkpointing format (default True)
        """
        if not self.trained:
            print("Warning: Model is not trained. Saving current (possibly initial) parameters.")
        
        if use_enhanced and hp is not None:
            # Use enhanced checkpointing
            from .checkpointing import save_checkpoint
            
            # Create default values if not provided
            if train_cfg is None:
                train_cfg = {}
            if history is None:
                history = {}
            if metrics is None:
                metrics = {}
            
            save_checkpoint(
                model=self,
                path=filepath,
                hp=hp,
                train_cfg=train_cfg,
                history=history,
                metrics=metrics
            )
        else:
            # Use legacy .npz format
            model_data = {
                'n_states': self.n_states,
                'n_emissions': self.n_emissions,
                'time_slices': self.time_slices,
                'transitions': self.transitions,
                'emission_means': self.emission_means,
                'emission_covars': self.emission_covars,
                'initial_probs': self.initial_probs,
                'state_names': self.state_names,
                'scaler_mean_': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
                'scaler_scale_': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
                'trained': self.trained
            }
            np.savez(filepath, **model_data)

    @classmethod
    def load_model(cls, filepath: str, map_location: Optional[str] = None, strict: bool = False) -> 'SolarTDMC':
        """Load model from file using enhanced checkpointing or legacy format.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model file
        map_location : str, optional
            Device location for loading (unused for TDMC but kept for API consistency)
        strict : bool
            Whether to enforce strict checkpoint format compatibility
            
        Returns:
        --------
        SolarTDMC
            Loaded model instance
        """
        from pathlib import Path
        
        # Determine file format by extension
        file_path = Path(filepath)
        
        if file_path.suffix == '.pt':
            # Try enhanced checkpointing first
            try:
                from .checkpointing import load_checkpoint
                checkpoint, metadata = load_checkpoint(filepath, map_location, strict)
                
                # Verify it's a TDMC model
                if metadata.get('model_type') != 'TDMC':
                    raise ValueError(f"Expected TDMC model, got {metadata.get('model_type')}")
                
                # Create model from checkpoint
                model = cls(
                    n_states=checkpoint['n_states'],
                    n_emissions=checkpoint['n_emissions'],
                    time_slices=checkpoint['time_slices']
                )
                
                # Load state
                state_dict = checkpoint['state_dict']
                model.transitions = state_dict['transitions']
                model.emission_means = state_dict['emission_means']
                model.emission_covars = state_dict['emission_covars']
                model.initial_probs = state_dict['initial_probs']
                model.trained = state_dict.get('trained', False)
                
                # Load scaler state
                model.scaler = StandardScaler()
                if state_dict.get('scaler_mean_') is not None:
                    model.scaler.mean_ = state_dict['scaler_mean_']
                    model.scaler.scale_ = state_dict['scaler_scale_']
                    model.scaler.n_features_in_ = model.n_emissions
                    model.scaler.n_samples_seen_ = 1
                
                # Load metadata
                if 'state_names' in checkpoint:
                    model.state_names = checkpoint['state_names']
                
                print(f"Loaded TDMC model from enhanced checkpoint (version {metadata.get('version')})")
                return model
                
            except Exception as e:
                if strict:
                    raise ValueError(f"Failed to load enhanced checkpoint: {e}")
                else:
                    print(f"Enhanced checkpointing failed ({e}), falling back to legacy format")
        
        # Legacy .npz format loading
        try:
            data = np.load(filepath, allow_pickle=True) # allow_pickle for state_names if they are objects
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {filepath}")

        model = cls(
            n_states=int(data['n_states']), # Ensure correct type
            n_emissions=int(data['n_emissions']),
            time_slices=int(data['time_slices'])
        )
        
        model.transitions = data['transitions']
        model.emission_means = data['emission_means']
        model.emission_covars = data['emission_covars']
        model.initial_probs = data['initial_probs']
        # Handle state_names carefully if they were saved as object array
        model.state_names = [str(sn) for sn in data['state_names']] if 'state_names' in data else [f"State_{i}" for i in range(model.n_states)]

        model.scaler = StandardScaler()
        if data['scaler_mean_'] is not None and data['scaler_scale_'] is not None:
            model.scaler.mean_ = data['scaler_mean_']
            model.scaler.scale_ = data['scaler_scale_']
            # After loading mean_ and scale_, the scaler is "fitted"
            model.scaler.n_features_in_ = model.n_emissions 
            if model.scaler.mean_ is not None : model.scaler.n_samples_seen_ = 1 # Dummy value to indicate it's fitted

        model.trained = bool(data['trained']) # Ensure correct type
        
        if not model.trained:
            print("Warning: Loaded model was not marked as trained.")
            # It might be necessary to call fit if scaler attributes were not saved/loaded properly
            # or if the model was indeed saved before training.

        return model
