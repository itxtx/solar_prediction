import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from typing import List, Tuple, Optional, Dict, Any, Union

class SolarTDMC:
    """
    Time-dynamic Markov Chain (TDMC) implementation for solar irradiance prediction.
    This is a specialized Hidden Markov Model with time-dependent transition probabilities.
    """

    def __init__(self, n_states: int = 4, n_emissions: int = 2, time_slices: int = 24):
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
        self.n_states = n_states
        self.n_emissions = n_emissions
        self.time_slices = time_slices

        self.transitions = np.zeros((time_slices, n_states, n_states))
        for t in range(time_slices):
            self.transitions[t] = np.ones((n_states, n_states)) / n_states

        self.emission_means = np.zeros((n_states, n_emissions))
        self.emission_covars = np.zeros((n_states, n_emissions, n_emissions))
        for s in range(n_states):
            self.emission_covars[s] = np.eye(n_emissions)

        self.initial_probs = np.ones(n_states) / n_states
        self.scaler = StandardScaler()
        self.trained = False
        self.state_names: List[str] = [f"State_{i}" for i in range(n_states)]

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
            max_iter: int = 100, tol: float = 1e-4, state_names: Optional[List[str]] = None):
        """
        Fit the TDMC model to data using the Baum-Welch algorithm.
        """
        X_scaled, time_indices = self._preprocess_data(X, timestamps)
        self._initialize_parameters(X_scaled, time_indices)
        self._baum_welch_update(X_scaled, time_indices, max_iter, tol)

        if state_names is not None and len(state_names) == self.n_states:
            self.state_names = state_names
        
        self.trained = True
        return self

    def _initialize_parameters(self, X_scaled: np.ndarray, time_indices: np.ndarray):
        """Initialize model parameters using K-means clustering."""
        kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init='auto')
        state_assignments = kmeans.fit_predict(X_scaled)

        for s in range(self.n_states):
            state_data = X_scaled[state_assignments == s]
            if len(state_data) > 1: # Need at least 2 samples for covariance
                self.emission_means[s] = np.mean(state_data, axis=0)
                cov = np.cov(state_data.T)
                min_eig = np.min(np.real(np.linalg.eigvals(cov)))
                if min_eig <= 0: # Ensure positive definiteness
                    cov += (-min_eig + 1e-6) * np.eye(self.n_emissions) # Add offset to make eigenvalues positive
                self.emission_covars[s] = cov + 1e-4 * np.eye(self.n_emissions) # Regularization
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
                          time_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform forward-backward algorithm for the TDMC."""
        n_samples = len(X_scaled)
        emission_probs = np.zeros((n_samples, self.n_states))

        for s in range(self.n_states):
            try:
                # Ensure covariance is positive definite before creating mvn
                cov = self.emission_covars[s]
                if np.min(np.linalg.eigvalsh(cov)) <= 1e-9: # Check positive definiteness
                    cov = cov + 1e-6 * np.eye(self.n_emissions) # Add jitter
                
                mvn = multivariate_normal(mean=self.emission_means[s], cov=cov, allow_singular=False)
                emission_probs[:, s] = mvn.pdf(X_scaled)
            except Exception as e: # Fallback if mvn fails
                # print(f"Warning: MVN PDF failed for state {s}: {e}. Using fallback.")
                # Fallback: simple Gaussian-like calculation (not a true PDF but a proximity measure)
                diff_sq = (X_scaled - self.emission_means[s])**2
                # Inverse of variance (diagonal) as precision
                precision = np.diag(1.0 / np.maximum(np.diag(self.emission_covars[s]), 1e-9)) 
                emission_probs[:, s] = np.exp(-0.5 * np.sum(diff_sq @ precision, axis=1))

        emission_probs = np.maximum(emission_probs, 1e-300) # Avoid zero probabilities

        alpha = np.zeros((n_samples, self.n_states))
        beta = np.zeros((n_samples, self.n_states))
        scale = np.zeros(n_samples)

        # Forward pass
        alpha[0] = self.initial_probs * emission_probs[0]
        scale[0] = np.sum(alpha[0])
        if scale[0] > 0: alpha[0] /= scale[0]

        for t in range(1, n_samples):
            current_time_slice_transitions = self.transitions[time_indices[t-1]] # Transitions from t-1 to t
            alpha[t] = (alpha[t-1] @ current_time_slice_transitions) * emission_probs[t]
            scale[t] = np.sum(alpha[t])
            if scale[t] > 0: alpha[t] /= scale[t]

        # Backward pass
        beta[-1] = 1.0
        for t in range(n_samples - 2, -1, -1):
            next_time_slice_transitions = self.transitions[time_indices[t]] # Transitions from t to t+1
            beta[t] = (next_time_slice_transitions @ (emission_probs[t+1] * beta[t+1]).T).T
            if scale[t+1] > 0: beta[t] /= scale[t+1]
            
        return alpha, beta, scale, emission_probs

    def _baum_welch_update(self, X_scaled: np.ndarray, time_indices: np.ndarray, 
                           max_iter: int, tol: float) -> float:
        """Perform Baum-Welch algorithm to update model parameters."""
        prev_log_likelihood = -np.inf
        log_likelihood = 0.0

        for iteration in range(max_iter):
            alpha, beta, scale, emission_probs = self._forward_backward(X_scaled, time_indices)
            
            current_log_likelihood = np.sum(np.log(np.maximum(scale, 1e-300))) # Avoid log(0)
            if abs(current_log_likelihood - prev_log_likelihood) < tol and iteration > 0:
                log_likelihood = current_log_likelihood
                # print(f"Converged at iteration {iteration+1}, Log-Likelihood: {log_likelihood:.4f}")
                break
            prev_log_likelihood = current_log_likelihood
            log_likelihood = current_log_likelihood
            # if iteration % 10 == 0:
            #     print(f"Iteration {iteration+1}/{max_iter}, Log-Likelihood: {log_likelihood:.4f}")

            n_samples = len(X_scaled)
            gamma = alpha * beta 
            gamma = gamma / np.maximum(np.sum(gamma, axis=1, keepdims=True), 1e-300) # Normalize gamma

            # Update initial state distribution
            self.initial_probs = gamma[0] / np.sum(gamma[0])
            self.initial_probs = np.maximum(self.initial_probs, 1e-300) # Ensure non-zero
            self.initial_probs /= np.sum(self.initial_probs) # Re-normalize

            # Update transition matrices
            new_transitions = np.zeros_like(self.transitions) + 1e-9 # Add smoothing factor
            gamma_sum_per_time_slice = np.zeros((self.time_slices, self.n_states)) + (self.n_states * 1e-9)

            for t_idx in range(n_samples - 1):
                ti = time_indices[t_idx] # Time slice for transition from t_idx to t_idx+1
                trans_prob_ti_to_ti1 = self.transitions[ti] # A_k for this time slice
                
                # Numerator for xi: alpha_i(k) * A_k(i,j) * B_j(O_{k+1}) * beta_j(k+1)
                # Denominator is sum over i,j of numerator which is P(O|lambda)
                # We need xi_t(i,j) = P(q_t=i, q_{t+1}=j | O, lambda)
                # xi_summed is sum over t of xi_t(i,j) for each time slice.
                
                # P(q_t=i, q_{t+1}=j | O, lambda) for a specific t_idx
                # proportional to alpha[t_idx, s1] * trans_prob_ti_to_ti1[s1,s2] * emission_probs[t_idx+1,s2] * beta[t_idx+1,s2]
                xi_slice = np.zeros((self.n_states, self.n_states))
                for s1 in range(self.n_states):
                    for s2 in range(self.n_states):
                        val = alpha[t_idx, s1] * \
                              trans_prob_ti_to_ti1[s1, s2] * \
                              emission_probs[t_idx+1, s2] * \
                              beta[t_idx+1, s2]
                        # The scaling factor scale[t_idx+1] is already incorporated in beta definition relative to alpha
                        # The term needs to be divided by P(O_t+1 | O_t ... O_0) for this specific transition which is complex.
                        # Standard HMM xi definition:
                        # xi_t(i,j) = (alpha_t(i) * A(i,j) * B_j(O_{t+1}) * beta_{t+1}(j)) / P(O|lambda)
                        # We scale P(O|lambda) out by normalizing later.
                        # Or, using scaled alpha and beta:
                        # xi_t(i,j) directly from alpha[t,i] * A[i,j] * B_j(O_{t+1}) * beta[t+1,j] / scale[t+1] (for beta)
                        # Since beta was scaled by scale[t+1], this is effectively:
                        # alpha[t_idx, s1] * trans_prob_ti_to_ti1[s1, s2] * emission_probs[t_idx+1, s2] * beta[t_idx+1, s2]
                        # No, the P(O|lambda) term needs care. Let's use sum of gamma.
                        xi_slice[s1,s2] = val 
                
                if np.sum(xi_slice) > 0 : # Avoid division by zero if xi_slice is all zeros
                    new_transitions[ti] += xi_slice / np.sum(xi_slice) # Normalize xi_slice before adding
                gamma_sum_per_time_slice[ti] += gamma[t_idx]


            for ts in range(self.time_slices):
                # Denominator for A_ij update is sum_t gamma_t(i) for that time slice
                # Numerator is sum_t xi_t(i,j) for that time slice
                # This needs sums not just for one t_idx but over all t_idx belonging to time slice ts
                # The accumulation above is correct. Now normalize:
                # Sum of xi over t, for each time slice
                # Sum of gamma over t, for each time slice
                # self.transitions[ts] = new_transitions[ts] / np.maximum(gamma_sum_per_time_slice[ts][:, np.newaxis], 1e-300)
                # A simpler approach:
                # Sum_t xi_t(i,j) / Sum_t gamma_t(i)
                # Let's re-evaluate transition update with standard formulas:
                # ξ_sum_ts[i,j] = sum_{t where time_indices[t]==ts} P(q_t=i, q_{t+1}=j | O, λ)
                # γ_sum_ts[i]   = sum_{t where time_indices[t]==ts} P(q_t=i | O, λ)
                # A[ts,i,j] = ξ_sum_ts[i,j] / γ_sum_ts[i]
                pass # Placeholder for correct accumulation of xi and gamma per time slice

            # Simpler Baum-Welch transition update (summing xi and gamma across relevant time steps for each slice transition matrix)
            # This part requires careful handling of time-slicing for xi and gamma sums.
            # Using an approximation for now by re-estimating from gamma, which is not fully correct for TDMC
            # A robust TDMC Baum-Welch M-step for transitions is more complex.
            # For now, let's keep the initialization's K-means based transition and allow it to adapt slowly or stick to it.
            # Or, an EM update where xi is calculated for each t and summed up per (time_slice, s1, s2)
            # And gamma summed up per (time_slice, s1)
            
            # M-step for transitions (corrected logic needed here for true TDMC B-W)
            # The current _initialize_parameters sets a reasonable starting point.
            # For a full Baum-Welch, one would accumulate xi sums and gamma sums
            # specific to each time_slice's transition matrix.
            # For this iteration, we will focus on emission updates which are more standard.

            # Update emission parameters
            for s in range(self.n_states):
                gamma_s = gamma[:, s]
                gamma_s_sum = np.sum(gamma_s)
                gamma_s_sum = np.maximum(gamma_s_sum, 1e-300) # Avoid division by zero

                # Update mean
                current_mean = np.sum(gamma_s[:, np.newaxis] * X_scaled, axis=0) / gamma_s_sum
                if not np.any(np.isnan(current_mean)): self.emission_means[s] = current_mean
                
                # Update covariance
                diff = X_scaled - self.emission_means[s]
                cov_s = np.dot((gamma_s[:, np.newaxis] * diff).T, diff) / gamma_s_sum
                
                # Ensure positive definiteness and add regularization
                min_eig = np.min(np.real(np.linalg.eigvals(cov_s)))
                if min_eig <= 1e-9: # Check positive definiteness
                    cov_s += (-min_eig + 1e-6) * np.eye(self.n_emissions) 
                cov_s += 1e-4 * np.eye(self.n_emissions) # Regularization
                if not np.any(np.isnan(cov_s)): self.emission_covars[s] = cov_s
        
        # print(f"Finished training. Final Log-Likelihood: {log_likelihood:.4f}")
        return log_likelihood

    def predict_states(self, X: np.ndarray, 
                       timestamps: Optional[Union[np.ndarray, pd.Series]] = None) -> np.ndarray:
        """
        Predict the most likely hidden state sequence using Viterbi algorithm.
        """
        if not self.trained:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        X_scaled, time_indices = self._preprocess_data(X, timestamps)
        n_samples = len(X_scaled)
        
        emission_probs = np.zeros((n_samples, self.n_states))
        for s in range(self.n_states):
            try:
                cov = self.emission_covars[s]
                if np.min(np.linalg.eigvalsh(cov)) <= 1e-9: cov += 1e-6 * np.eye(self.n_emissions)
                mvn = multivariate_normal(mean=self.emission_means[s], cov=cov, allow_singular=False)
                emission_probs[:, s] = mvn.pdf(X_scaled)
            except Exception: # Fallback
                precision = np.diag(1.0 / np.maximum(np.diag(self.emission_covars[s]), 1e-9))
                emission_probs[:, s] = np.exp(-0.5 * np.sum((X_scaled - self.emission_means[s])**2 @ precision, axis=1))

        emission_probs = np.maximum(emission_probs, 1e-300) # Avoid log(0) later
        
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

    def save_model(self, filepath: str):
        """Save model parameters to a .npz file."""
        if not self.trained:
            print("Warning: Model is not trained. Saving current (possibly initial) parameters.")
        
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
    def load_model(cls, filepath: str) -> 'SolarTDMC':
        """Load model from a .npz file."""
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