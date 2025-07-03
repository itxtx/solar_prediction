#!/usr/bin/env python3
"""
Test script for enhanced checkpointing functionality.

This script tests the new enhanced checkpointing utility to ensure it works correctly
with all model types (TDMC, LSTM, GRU) and maintains backward compatibility.
"""

import numpy as np
import tempfile
import os
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import model classes and checkpointing utilities
from solar_prediction.tdmc import SolarTDMC
from solar_prediction.lstm import WeatherLSTM, ModelHyperparameters, create_model_hyperparameters_from_config
from solar_prediction.gru import WeatherGRU, GRUModelHyperparameters, create_gru_model_hyperparameters_from_config
from solar_prediction.checkpointing import save_checkpoint, load_checkpoint, create_model_from_checkpoint


def create_test_data(n_samples=100, n_features=10, sequence_length=24):
    """Create synthetic test data for model testing."""
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.randn(n_samples, 1)
    return X, y


def test_tdmc_enhanced_checkpointing():
    """Test enhanced checkpointing with TDMC model."""
    print("\n=== Testing TDMC Enhanced Checkpointing ===")
    
    # Create TDMC model
    model = SolarTDMC(n_states=3, n_emissions=5, time_slices=24)
    
    # Create some test data
    X_test = np.random.randn(50, 5)
    timestamps = np.arange(50)
    
    # Fit model (simulate training)
    model.fit(X_test, timestamps)
    
    # Prepare test metadata
    hp = {
        'n_states': 3,
        'n_emissions': 5,
        'time_slices': 24
    }
    train_cfg = {
        'max_iterations': 100,
        'tolerance': 1e-4
    }
    history = {
        'log_likelihood': [-1000, -950, -920, -900]
    }
    metrics = {
        'final_log_likelihood': -900,
        'convergence_iterations': 4
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test enhanced saving
        enhanced_path = Path(temp_dir) / "tdmc_enhanced.pt"
        model.save_model(
            str(enhanced_path),
            hp=hp,
            train_cfg=train_cfg,
            history=history,
            metrics=metrics,
            use_enhanced=True
        )
        
        # Test enhanced loading
        loaded_model = SolarTDMC.load_model(str(enhanced_path))
        
        # Verify model state
        assert loaded_model.n_states == model.n_states
        assert loaded_model.n_emissions == model.n_emissions
        assert loaded_model.time_slices == model.time_slices
        assert loaded_model.trained == model.trained
        np.testing.assert_array_almost_equal(loaded_model.transitions, model.transitions)
        
        # Test legacy format compatibility
        legacy_path = Path(temp_dir) / "tdmc_legacy.npz"
        model.save_model(str(legacy_path), use_enhanced=False)
        loaded_legacy = SolarTDMC.load_model(str(legacy_path))
        assert loaded_legacy.n_states == model.n_states
        
        print("✓ TDMC enhanced checkpointing tests passed")


def test_lstm_enhanced_checkpointing():
    """Test enhanced checkpointing with LSTM model."""
    print("\n=== Testing LSTM Enhanced Checkpointing ===")
    
    # Create LSTM model
    params = create_model_hyperparameters_from_config(input_dim=10)
    model = WeatherLSTM(params)
    
    # Simulate some training history
    model.history = {
        'epochs': [1, 2, 3],
        'train_loss': [1.0, 0.8, 0.6],
        'val_loss': [1.1, 0.9, 0.7],
        'val_rmse': [0.8, 0.7, 0.6],
        'val_r2': [0.5, 0.6, 0.7],
        'val_mape': [15.0, 12.0, 10.0],
        'val_mae': [0.5, 0.4, 0.3],
        'lr': [0.001, 0.001, 0.0005]
    }
    
    # Prepare test metadata
    train_cfg = {
        'epochs': 3,
        'batch_size': 32,
        'learning_rate': 0.001
    }
    metrics = {
        'final_val_loss': 0.7,
        'best_val_r2': 0.7
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test enhanced saving
        enhanced_path = Path(temp_dir) / "lstm_enhanced.pt"
        model.save(
            str(enhanced_path),
            train_cfg=train_cfg,
            metrics=metrics,
            use_enhanced=True
        )
        
        # Test enhanced loading
        loaded_model = WeatherLSTM.load(str(enhanced_path))
        
        # Verify model parameters
        assert loaded_model.params.input_dim == model.params.input_dim
        assert loaded_model.params.hidden_dim == model.params.hidden_dim
        assert loaded_model.history['epochs'] == model.history['epochs']
        
        # Test legacy format compatibility
        legacy_path = Path(temp_dir) / "lstm_legacy.pt"
        model.save(str(legacy_path), use_enhanced=False)
        loaded_legacy = WeatherLSTM.load(str(legacy_path))
        assert loaded_legacy.params.input_dim == model.params.input_dim
        
        print("✓ LSTM enhanced checkpointing tests passed")


def test_gru_enhanced_checkpointing():
    """Test enhanced checkpointing with GRU model."""
    print("\n=== Testing GRU Enhanced Checkpointing ===")
    
    # Create GRU model
    params = create_gru_model_hyperparameters_from_config(input_dim=10)
    model = WeatherGRU(params)
    
    # Simulate some training history
    model.history = {
        'epochs': [1, 2, 3],
        'train_loss': [1.2, 0.9, 0.7],
        'val_loss': [1.3, 1.0, 0.8],
        'val_rmse': [0.9, 0.8, 0.7],
        'val_r2': [0.4, 0.5, 0.6],
        'val_mape': [18.0, 15.0, 12.0],
        'val_mae': [0.6, 0.5, 0.4],
        'lr': [0.001, 0.001, 0.0005]
    }
    
    # Prepare test metadata
    train_cfg = {
        'epochs': 3,
        'batch_size': 32,
        'learning_rate': 0.001,
        'bidirectional': False
    }
    metrics = {
        'final_val_loss': 0.8,
        'best_val_r2': 0.6
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test enhanced saving
        enhanced_path = Path(temp_dir) / "gru_enhanced.pt"
        model.save(
            str(enhanced_path),
            train_cfg=train_cfg,
            metrics=metrics,
            use_enhanced=True
        )
        
        # Test enhanced loading
        loaded_model = WeatherGRU.load(str(enhanced_path))
        
        # Verify model parameters
        assert loaded_model.params.input_dim == model.params.input_dim
        assert loaded_model.params.hidden_dim == model.params.hidden_dim
        assert loaded_model.params.bidirectional == model.params.bidirectional
        assert loaded_model.history['epochs'] == model.history['epochs']
        
        # Test legacy format compatibility
        legacy_path = Path(temp_dir) / "gru_legacy.pt"
        model.save(str(legacy_path), use_enhanced=False)
        loaded_legacy = WeatherGRU.load(str(legacy_path))
        assert loaded_legacy.params.input_dim == model.params.input_dim
        
        print("✓ GRU enhanced checkpointing tests passed")


def test_cross_compatibility():
    """Test version compatibility and error handling."""
    print("\n=== Testing Cross-Compatibility ===")
    
    # Test loading with strict mode
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a TDMC model and save it
        model = SolarTDMC(n_states=2, n_emissions=3, time_slices=12)
        model.fit(np.random.randn(30, 3), np.arange(30))
        
        enhanced_path = Path(temp_dir) / "test_model.pt"
        model.save_model(
            str(enhanced_path),
            hp={'n_states': 2, 'n_emissions': 3, 'time_slices': 12},
            train_cfg={},
            history={},
            metrics={}
        )
        
        # Test loading with different strict modes
        checkpoint, metadata = load_checkpoint(str(enhanced_path), strict=False)
        assert metadata['model_type'] == 'TDMC'
        assert 'version' in metadata
        
        # Test factory function
        created_model = create_model_from_checkpoint(str(enhanced_path))
        assert isinstance(created_model, SolarTDMC)
        assert created_model.n_states == 2
        
        print("✓ Cross-compatibility tests passed")


def test_metadata_completeness():
    """Test that all required metadata is saved and loaded correctly."""
    print("\n=== Testing Metadata Completeness ===")
    
    # Create an LSTM model with comprehensive metadata
    params = create_model_hyperparameters_from_config(input_dim=8)
    model = WeatherLSTM(params)
    
    comprehensive_hp = {
        'input_dim': 8,
        'hidden_dim': 64,
        'num_layers': 2,
        'output_dim': 1,
        'dropout_prob': 0.3
    }
    
    comprehensive_train_cfg = {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'patience': 10,
        'weight_decay': 1e-5,
        'scheduler_type': 'plateau'
    }
    
    comprehensive_history = {
        'epochs': list(range(1, 11)),
        'train_loss': [1.0 - i*0.08 for i in range(10)],
        'val_loss': [1.1 - i*0.08 for i in range(10)],
        'val_rmse': [0.8 - i*0.05 for i in range(10)],
        'val_r2': [0.3 + i*0.05 for i in range(10)],
        'val_mape': [20.0 - i*1.5 for i in range(10)],
        'val_mae': [0.6 - i*0.03 for i in range(10)],
        'lr': [0.001] * 10
    }
    
    comprehensive_metrics = {
        'final_train_loss': 0.28,
        'final_val_loss': 0.38,
        'best_val_r2': 0.78,
        'best_val_rmse': 0.35,
        'training_time_seconds': 1245.6,
        'total_parameters': sum(p.numel() for p in model.parameters())
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        enhanced_path = Path(temp_dir) / "comprehensive_test.pt"
        
        # Save with comprehensive metadata
        save_checkpoint(
            model=model,
            path=str(enhanced_path),
            hp=comprehensive_hp,
            train_cfg=comprehensive_train_cfg,
            history=comprehensive_history,
            metrics=comprehensive_metrics,
            version="1.1"
        )
        
        # Load and verify all metadata is present
        checkpoint, metadata = load_checkpoint(str(enhanced_path))
        
        # Check core checkpoint data
        assert 'state_dict' in checkpoint
        assert 'hyperparameters' in checkpoint
        assert 'training_config' in checkpoint
        assert 'history' in checkpoint
        assert 'metrics' in checkpoint
        assert 'version' in checkpoint
        assert 'timestamp' in checkpoint
        assert 'model_type' in checkpoint
        
        # Check metadata
        assert metadata['model_type'] == 'LSTM'
        assert metadata['version'] == '1.1'
        assert 'timestamp' in metadata
        assert 'load_time' in metadata
        
        # Verify data integrity
        assert checkpoint['hyperparameters']['input_dim'] == 8
        assert checkpoint['training_config']['epochs'] == 50
        assert len(checkpoint['history']['epochs']) == 10
        assert checkpoint['metrics']['final_val_loss'] == 0.38
        
        print("✓ Metadata completeness tests passed")


def main():
    """Run all enhanced checkpointing tests."""
    print("Enhanced Checkpointing Test Suite")
    print("=" * 50)
    
    try:
        test_tdmc_enhanced_checkpointing()
        test_lstm_enhanced_checkpointing()
        test_gru_enhanced_checkpointing()
        test_cross_compatibility()
        test_metadata_completeness()
        
        print("\n" + "=" * 50)
        print("✅ All enhanced checkpointing tests passed successfully!")
        print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
