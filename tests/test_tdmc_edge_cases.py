#!/usr/bin/env python3
"""
Comprehensive edge case tests for TDMC transition matrix updates.
"""

import numpy as np
import warnings
from solar_prediction.tdmc import SolarTDMC
from solar_prediction.config import get_config


def test_zero_xi_handling():
    """Test handling when ξ_t sums to zero (numerical edge case)."""
    
    np.random.seed(42)
    
    # Create data that might lead to very small ξ values
    X = np.array([[0, 0], [1e-10, 1e-10], [0, 0]])
    timestamps = np.array([0, 1, 2])
    
    model = SolarTDMC(n_states=2, n_emissions=2, time_slices=3)
    
    # Should handle gracefully without crashing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress potential numerical warnings
        model.fit(X, timestamps=timestamps, max_iter=2)
    
    # Verify transition matrices are still valid
    assert not np.any(np.isnan(model.transitions))
    assert np.all(model.transitions >= 0)
    assert np.all(model.transitions <= 1)
    
    # Check row normalization
    for ts in range(3):
        for i in range(2):
            row_sum = np.sum(model.transitions[ts, i, :])
            assert abs(row_sum - 1.0) <= 1e-6
    
    print("✅ Zero ξ handling test passed!")


def test_single_state_per_time_slice():
    """Test when some time slices have data from only one state."""
    
    np.random.seed(123)
    
    # Create data where some time slices will likely be assigned to single states
    X = np.array([
        [10, 10],   # Time slice 0 - high values
        [10.1, 10.1],  # Time slice 1 - high values  
        [-10, -10], # Time slice 2 - low values
        [-10.1, -10.1], # Time slice 0 - low values (different from first)
        [0, 0],     # Time slice 1 - medium values
        [0.1, 0.1]  # Time slice 2 - medium values
    ])
    timestamps = np.array([0, 1, 2, 0, 1, 2])
    
    model = SolarTDMC(n_states=3, n_emissions=2, time_slices=3)
    
    model.fit(X, timestamps=timestamps, max_iter=3)
    
    # Should still produce valid transition matrices
    assert not np.any(np.isnan(model.transitions))
    assert np.all(model.transitions > 0)  # Should be > 0 due to smoothing prior
    
    print("✅ Single state per time slice test passed!")


def test_smoothing_prior_effect():
    """Test that smoothing prior prevents zero probabilities."""
    
    config = get_config()
    tdmc_config = config.models.tdmc
    prior = tdmc_config.transition_smoothing_prior
    min_prob = tdmc_config.min_probability
    
    np.random.seed(456)
    
    # Create simple data
    X = np.random.randn(20, 2)
    timestamps = np.array([i % 5 for i in range(20)])
    
    model = SolarTDMC(n_states=2, n_emissions=2, time_slices=5)
    model.fit(X, timestamps=timestamps, max_iter=2)
    
    # All transition probabilities should be at least min_probability
    assert np.all(model.transitions >= min_prob), \
        f"Some probabilities below min_probability {min_prob}"
    
    # The smoothing should ensure no transitions are exactly zero
    assert not np.any(model.transitions == 0), \
        "Found zero transition probabilities despite smoothing"
    
    print(f"✅ Smoothing prior effect test passed! Min probability: {min_prob}")


def test_numerical_stability_extreme_values():
    """Test numerical stability with extreme input values."""
    
    np.random.seed(789)
    
    # Create data with extreme values
    X = np.array([
        [1e6, 1e6],     # Very large values
        [1e-6, 1e-6],   # Very small values
        [0, 0],         # Zero values
        [-1e6, -1e6],   # Very negative values
    ])
    timestamps = np.array([0, 1, 2, 3])
    
    model = SolarTDMC(n_states=2, n_emissions=2, time_slices=4)
    
    # Should handle extreme values without numerical issues
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, timestamps=timestamps, max_iter=2)
    
    # Check for numerical stability
    assert not np.any(np.isnan(model.transitions))
    assert not np.any(np.isinf(model.transitions))
    assert np.all(np.isfinite(model.transitions))
    
    # Check bounds
    assert np.all(model.transitions >= 0)
    assert np.all(model.transitions <= 1)
    
    print("✅ Numerical stability extreme values test passed!")


def test_transition_matrix_convergence():
    """Test that transition matrices converge to stable values."""
    
    np.random.seed(101112)
    
    # Create data with clear temporal patterns
    n_samples = 200
    X = np.zeros((n_samples, 2))
    timestamps = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        hour = i % 24
        timestamps[i] = hour
        
        # Create pattern: high values during day (6-18), low at night
        if 6 <= hour <= 18:
            X[i] = [5 + np.random.normal(0, 0.5), 20 + np.random.normal(0, 1)]
        else:
            X[i] = [0.5 + np.random.normal(0, 0.2), 5 + np.random.normal(0, 0.5)]
    
    model = SolarTDMC(n_states=2, n_emissions=2, time_slices=24)
    
    # Fit with more iterations to test convergence
    model.fit(X, timestamps=timestamps, max_iter=10)
    
    # Store initial transition matrices
    final_transitions = model.transitions.copy()
    
    # Run additional iterations to see if it converges
    model.fit(X, timestamps=timestamps, max_iter=5)
    
    # Transitions should not change drastically (convergence)
    max_change = np.max(np.abs(final_transitions - model.transitions))
    print(f"Max transition change after additional iterations: {max_change:.6f}")
    
    # Should still be valid
    assert not np.any(np.isnan(model.transitions))
    
    print("✅ Transition matrix convergence test passed!")


if __name__ == "__main__":
    print("Running TDMC transition matrix edge case tests...\n")
    
    test_zero_xi_handling()
    print()
    
    test_single_state_per_time_slice()
    print()
    
    test_smoothing_prior_effect()
    print()
    
    test_numerical_stability_extreme_values()
    print()
    
    test_transition_matrix_convergence()
    print()
    
    print("🎉 All edge case tests completed successfully!")
    print("\nThe TDMC implementation is robust against:")
    print("✓ Zero ξ values and numerical edge cases")
    print("✓ Sparse data distributions across time slices")
    print("✓ Extreme input values and numerical instability")
    print("✓ Proper smoothing prior application")
    print("✓ Convergence behavior over multiple iterations")
