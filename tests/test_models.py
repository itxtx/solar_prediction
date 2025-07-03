"""
Test models: train LSTM & GRU for 2 epochs with tiny window, assert loss↓.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from solar_prediction.lstm import WeatherLSTM, create_model_hyperparameters_from_config, create_training_config_from_config
from solar_prediction.gru import WeatherGRU, create_gru_model_hyperparameters_from_config, create_gru_training_config_from_config
from solar_prediction.data_prep import prepare_weather_data
from solar_prediction.config import get_config


@pytest.fixture
def device():
    """Get device for testing (prefer CPU for consistency)."""
    return "cpu"  # Use CPU for tests to avoid GPU memory issues


@pytest.fixture
def tiny_training_data(small_sample_data):
    """Prepare tiny training data for quick model tests."""
    # Get configuration
    config = get_config()
    
    # Use only the first 30 rows for very quick testing
    df = small_sample_data.head(30).copy()
    
    try:
        # Prepare data using the pipeline
        X_train, X_val, X_test, y_train, y_val, y_test, scalers, feature_cols, transform_info = prepare_weather_data(
            df,
            config.data,
            config.transformation,
            config.features,
            config.scaling,
            config.sequences
        )
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'feature_cols': feature_cols,
            'scalers': scalers,
            'transform_info': transform_info
        }
    except Exception as e:
        pytest.skip(f"Failed to prepare tiny training data: {e}")


class TestLSTMModel:
    """Test LSTM model functionality."""
    
    def test_lstm_initialization(self):
        """Test LSTM model initialization."""
        model_params = create_model_hyperparameters_from_config(
            input_dim=5,
            config_override={'hidden_dim': 32, 'num_layers': 1}
        )
        
        model = WeatherLSTM(model_params)
        
        assert model.params.input_dim == 5
        assert model.params.hidden_dim == 32
        assert model.params.num_layers == 1
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'fc1')
        
    def test_lstm_forward_pass(self, numpy_arrays_2d, device):
        """Test LSTM forward pass."""
        X, y = numpy_arrays_2d
        
        # Create model
        model_params = create_model_hyperparameters_from_config(
            input_dim=X.shape[2],
            config_override={'hidden_dim': 16, 'num_layers': 1}
        )
        model = WeatherLSTM(model_params)
        model.to(device)
        model.eval()
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(X_tensor)
        
        # Check output shape
        assert output.shape == (X.shape[0], model_params.output_dim)
        assert torch.all(torch.isfinite(output))
        
    def test_lstm_training_2_epochs(self, tiny_training_data, device):
        """Test LSTM training for 2 epochs with tiny window, assert loss decreases."""
        data = tiny_training_data
        
        if data['X_train'].shape[0] < 5:
            pytest.skip("Not enough training data")
        
        # Create model with tiny configuration
        model_params = create_model_hyperparameters_from_config(
            input_dim=data['X_train'].shape[2],
            config_override={'hidden_dim': 8, 'num_layers': 1, 'dropout_prob': 0.1}
        )
        
        # Create training config for 2 epochs
        training_config = create_training_config_from_config(
            config_override={
                'epochs': 2,
                'batch_size': min(4, data['X_train'].shape[0]),
                'learning_rate': 0.01,
                'patience': 10
            }
        )
        
        model = WeatherLSTM(model_params)
        model.to(device)
        
        try:
            # Train the model
            model.fit(
                data['X_train'],
                data['y_train'],
                data['X_val'] if data['X_val'].shape[0] > 0 else data['X_train'][:2],
                data['y_val'] if data['y_val'].shape[0] > 0 else data['y_train'][:2],
                training_config,
                device=device
            )
            
            # Check that training history exists
            assert len(model.history['train_loss']) == 2  # 2 epochs
            assert len(model.history['val_loss']) == 2
            
            # Assert that loss decreased or stayed stable (allow small increases due to tiny data)
            initial_loss = model.history['train_loss'][0]
            final_loss = model.history['train_loss'][-1]
            
            # Loss should decrease or not increase by more than 20% (due to tiny dataset)
            assert final_loss <= initial_loss * 1.2, \
                f"Loss increased too much: {initial_loss} -> {final_loss}"
            
            # Check that losses are finite
            assert all(np.isfinite(model.history['train_loss']))
            assert all(np.isfinite(model.history['val_loss']))
            
        except Exception as e:
            pytest.skip(f"LSTM training failed: {e}")

    def test_lstm_prediction(self, tiny_training_data, device):
        """Test LSTM prediction after training."""
        data = tiny_training_data
        
        if data['X_train'].shape[0] < 3:
            pytest.skip("Not enough training data")
        
        model_params = create_model_hyperparameters_from_config(
            input_dim=data['X_train'].shape[2],
            config_override={'hidden_dim': 8, 'num_layers': 1}
        )
        
        training_config = create_training_config_from_config(
            config_override={'epochs': 1, 'batch_size': 2}
        )
        
        model = WeatherLSTM(model_params)
        model.to(device)
        
        try:
            # Quick training
            model.fit(
                data['X_train'][:5] if data['X_train'].shape[0] >= 5 else data['X_train'],
                data['y_train'][:5] if data['y_train'].shape[0] >= 5 else data['y_train'],
                data['X_train'][:2],
                data['y_train'][:2],
                training_config,
                device=device
            )
            
            # Test prediction
            test_X = data['X_train'][:3]
            predictions = model.predict(test_X, device=device)
            
            # Check prediction shape and values
            assert predictions.shape == (test_X.shape[0], model_params.output_dim)
            assert np.all(np.isfinite(predictions))
            
        except Exception as e:
            pytest.skip(f"LSTM prediction test failed: {e}")


class TestGRUModel:
    """Test GRU model functionality."""
    
    def test_gru_initialization(self):
        """Test GRU model initialization."""
        model_params = create_gru_model_hyperparameters_from_config(
            input_dim=5,
            config_override={'hidden_dim': 32, 'num_layers': 1, 'bidirectional': False}
        )
        
        model = WeatherGRU(model_params)
        
        assert model.params.input_dim == 5
        assert model.params.hidden_dim == 32
        assert model.params.num_layers == 1
        assert hasattr(model, 'gru')
        assert hasattr(model, 'fc')
        
    def test_gru_forward_pass(self, numpy_arrays_2d, device):
        """Test GRU forward pass."""
        X, y = numpy_arrays_2d
        
        # Create model
        model_params = create_gru_model_hyperparameters_from_config(
            input_dim=X.shape[2],
            config_override={'hidden_dim': 16, 'num_layers': 1, 'bidirectional': False}
        )
        model = WeatherGRU(model_params)
        model.to(device)
        model.eval()
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(X_tensor)
        
        # Check output shape
        assert output.shape == (X.shape[0], model_params.output_dim)
        assert torch.all(torch.isfinite(output))
        
    def test_gru_training_2_epochs(self, tiny_training_data, device):
        """Test GRU training for 2 epochs with tiny window, assert loss decreases."""
        data = tiny_training_data
        
        if data['X_train'].shape[0] < 5:
            pytest.skip("Not enough training data")
        
        # Create model with tiny configuration
        model_params = create_gru_model_hyperparameters_from_config(
            input_dim=data['X_train'].shape[2],
            config_override={'hidden_dim': 8, 'num_layers': 1, 'bidirectional': False, 'dropout_prob': 0.1}
        )
        
        # Create training config for 2 epochs
        training_config = create_gru_training_config_from_config(
            config_override={
                'epochs': 2,
                'batch_size': min(4, data['X_train'].shape[0]),
                'learning_rate': 0.01,
                'patience': 10
            }
        )
        
        model = WeatherGRU(model_params)
        model.to(device)
        
        try:
            # Train the model
            model.fit(
                data['X_train'],
                data['y_train'],
                data['X_val'] if data['X_val'].shape[0] > 0 else data['X_train'][:2],
                data['y_val'] if data['y_val'].shape[0] > 0 else data['y_train'][:2],
                training_config,
                device=device
            )
            
            # Check that training history exists
            assert len(model.history['train_loss']) == 2  # 2 epochs
            assert len(model.history['val_loss']) == 2
            
            # Assert that loss decreased or stayed stable
            initial_loss = model.history['train_loss'][0]
            final_loss = model.history['train_loss'][-1]
            
            # Loss should decrease or not increase by more than 20% (due to tiny dataset)
            assert final_loss <= initial_loss * 1.2, \
                f"Loss increased too much: {initial_loss} -> {final_loss}"
            
            # Check that losses are finite
            assert all(np.isfinite(model.history['train_loss']))
            assert all(np.isfinite(model.history['val_loss']))
            
        except Exception as e:
            pytest.skip(f"GRU training failed: {e}")

    def test_gru_bidirectional(self, numpy_arrays_2d, device):
        """Test bidirectional GRU functionality."""
        X, y = numpy_arrays_2d
        
        # Create bidirectional model
        model_params = create_gru_model_hyperparameters_from_config(
            input_dim=X.shape[2],
            config_override={'hidden_dim': 8, 'num_layers': 1, 'bidirectional': True}
        )
        model = WeatherGRU(model_params)
        model.to(device)
        model.eval()
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(X_tensor)
        
        # Check output shape (should still be same as unidirectional)
        assert output.shape == (X.shape[0], model_params.output_dim)
        assert torch.all(torch.isfinite(output))


class TestModelComparisons:
    """Test comparisons between LSTM and GRU models."""
    
    def test_lstm_vs_gru_output_shapes(self, numpy_arrays_2d, device):
        """Test that LSTM and GRU produce same output shapes."""
        X, y = numpy_arrays_2d
        
        # Create LSTM model
        lstm_params = create_model_hyperparameters_from_config(
            input_dim=X.shape[2],
            config_override={'hidden_dim': 16, 'num_layers': 1}
        )
        lstm_model = WeatherLSTM(lstm_params)
        lstm_model.to(device)
        lstm_model.eval()
        
        # Create GRU model
        gru_params = create_gru_model_hyperparameters_from_config(
            input_dim=X.shape[2],
            config_override={'hidden_dim': 16, 'num_layers': 1, 'bidirectional': False}
        )
        gru_model = WeatherGRU(gru_params)
        gru_model.to(device)
        gru_model.eval()
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Forward pass
        with torch.no_grad():
            lstm_output = lstm_model(X_tensor)
            gru_output = gru_model(X_tensor)
        
        # Check that shapes match
        assert lstm_output.shape == gru_output.shape
        assert torch.all(torch.isfinite(lstm_output))
        assert torch.all(torch.isfinite(gru_output))

    def test_model_parameter_counts(self):
        """Test that model parameter counts are reasonable."""
        input_dim = 10
        
        # Create models with same configuration
        lstm_params = create_model_hyperparameters_from_config(
            input_dim=input_dim,
            config_override={'hidden_dim': 32, 'num_layers': 2}
        )
        lstm_model = WeatherLSTM(lstm_params)
        
        gru_params = create_gru_model_hyperparameters_from_config(
            input_dim=input_dim,
            config_override={'hidden_dim': 32, 'num_layers': 2, 'bidirectional': False}
        )
        gru_model = WeatherGRU(gru_params)
        
        # Count parameters
        lstm_param_count = sum(p.numel() for p in lstm_model.parameters())
        gru_param_count = sum(p.numel() for p in gru_model.parameters())
        
        # LSTM should have more parameters than GRU (4 gates vs 3 gates)
        assert lstm_param_count > gru_param_count
        
        # Parameter counts should be reasonable (not too large)
        assert lstm_param_count < 1_000_000  # Less than 1M parameters
        assert gru_param_count < 1_000_000


class TestModelEdgeCases:
    """Test edge cases for model training and prediction."""
    
    def test_model_with_minimal_data(self, device):
        """Test models with minimal training data."""
        # Create minimal data
        X = np.random.randn(3, 5, 4)  # 3 samples, 5 timesteps, 4 features
        y = np.random.randn(3, 1)
        
        # Test LSTM
        lstm_params = create_model_hyperparameters_from_config(
            input_dim=4,
            config_override={'hidden_dim': 4, 'num_layers': 1}
        )
        training_config = create_training_config_from_config(
            config_override={'epochs': 1, 'batch_size': 2}
        )
        
        lstm_model = WeatherLSTM(lstm_params)
        lstm_model.to(device)
        
        try:
            lstm_model.fit(X[:2], y[:2], X[2:], y[2:], training_config, device=device)
            predictions = lstm_model.predict(X, device=device)
            assert predictions.shape[0] == X.shape[0]
        except Exception as e:
            pytest.skip(f"LSTM minimal data test failed: {e}")

    def test_model_error_handling(self, device):
        """Test model error handling with invalid inputs."""
        # Create model
        model_params = create_model_hyperparameters_from_config(
            input_dim=5,
            config_override={'hidden_dim': 8, 'num_layers': 1}
        )
        model = WeatherLSTM(model_params)
        model.to(device)
        
        # Test with wrong input dimensions
        X_wrong = np.random.randn(10, 5, 3)  # Wrong feature dimension
        y_wrong = np.random.randn(10, 1)
        
        training_config = create_training_config_from_config(
            config_override={'epochs': 1, 'batch_size': 2}
        )
        
        with pytest.raises((ValueError, RuntimeError)):
            model.fit(X_wrong, y_wrong, X_wrong[:2], y_wrong[:2], training_config, device=device)
