# Solar Radiation Prediction LSTM Model: Technical Summary

## Model Architecture

The model utilizes a Long Short-Term Memory (LSTM) network. The specific structure of the `WeatherLSTM` network, based on 
the final trained model, is as follows:

**WeatherLSTM Network Structure:**<br>
┌─────────────────────────────────────────────────────┐<br>
│ ➤ LSTM Layer (10→256, layers=2, dropout=0.3)       │<br>
│ ➤ Dropout Layer (p=0.3)                             │<br>
│ ➤ Fully Connected Layer (256→64)                    │<br>
│ ➤ ReLU Activation                                   │<br>
│ ➤ Dropout Layer (p=0.3)                             │<br>
│ ➤ Fully Connected Layer (64→32)                     │<br>
│ ➤ ReLU Activation                                   │<br>
│ ➤ Dropout Layer (p=0.3)                             │<br>
│ ➤ Fully Connected Layer (32→1)                      │<br>
└─────────────────────────────────────────────────────┘

## Core Parameters

- **Input Dimension**: 10 features
- **Hidden Dimension**: 256
- **LSTM Layers**: 2
- **Output Dimension**: 1 (predicting a single value, 'Radiation')
- **Dropout Probability**: 0.3 (applied after the LSTM layer and after each ReLU activation in the fully connected block)
- **Fully Connected Layers**: 256→64→32→1 with ReLU activations

## Dataset Structure

- **Training Set**: X_train shape: (118051, 24, 10), y_train shape: (118051, 1)
- **Validation Set**: X_val shape: (49188, 24, 10), y_val shape: (49188, 1)
- **Test Set**: X_test shape: (29513, 24, 10), y_test shape: (29513, 1)
- **Data Format**: Sequences of 24 timesteps, with 10 features per timestep.

## Loss Function

The model was trained using the Mean Squared Error (MSE) loss function:

$\mathcal{L}_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_{true} - y_{pred})^2$

(Note: The Optuna hyperparameter search explored a combined MSE and MAPE loss, but the final model training utilized MSE loss.)

## Regularization Techniques

- **Dropout**: Probability of 0.3 applied after the LSTM layer and after each ReLU activation in the fully connected block.
- **L2 Regularization**: Weight decay of $1 \times 10^{-5}$ applied during training.
- **Gradient Clipping**: Norm constrained to 1.0 to prevent exploding gradients.

## Optimization Strategy

- **Optimizer**: Adam.
- **Learning Rate**: $0.00015$ for the final training.
- **Learning Rate Scheduler**: CosineAnnealingLR with $T_{max}=100$.
- **Patience for Early Stopping**: 30 epochs (though the final training ran for the full 100 epochs as per the log).

## Data Transformation

- **Target Variable ('Radiation') Transformation**:
    - A floor of 0.0 was applied to 'Radiation' values before transformation.
    - Yeo-Johnson Power Transform applied to the 'Radiation' column (lambda = -0.1973). The transformed target is named 'Radiation_yj'.
    - The transformed target ('Radiation_yj') was then scaled using StandardScaler.
- **Feature Scaling**: Input features were scaled using MinMaxScaler.
- **Feature Engineering**:
    - 'SolarElevation': A proxy feature for solar elevation was added.
    - 'Radiation_is_low': A binary feature created with a threshold of 0.0 for the 'Radiation' value.
    - Redundant date/time features were removed if similar ones already existed (e.g., 'DaylightMinutes' removed if 'dayLength' present).

## Input Features

The model uses 10 features for prediction:

- 'Cloudcover'
- 'Humidity'
- 'Pressure'
- 'Radiation' (original)
- 'Radiation_is_low'
- 'Rain'
- 'Temperature'
- 'TimeMinutesCos' (Cyclical encoding of time)
- 'TimeMinutesSin' (Cyclical encoding of time)
- 'WindSpeed'

## Key Performance Metrics (Test Set - Original Scale)

- **RMSE (Root Mean Squared Error)**: 0.1393
- **MAE (Mean Absolute Error)**: 0.0806
- **R² (R-squared)**: 0.9893
- **Correlation Coefficient**: 0.9914
- **MAPE (Mean Absolute Percentage Error)**: Extremely high due to near-zero actual values.
- **MAPE (capped at 100% error per point)**: 52.32%

## Error Distribution Analysis

The residuals (Actuals - Predictions) on the scaled test data show:

- **Mean Residual**: 0.0028
- **Std Dev of Residuals**: 0.0275
- **Min/Max Residuals**: -0.1730 / 0.2808

The high uncapped MAPE on the original scale suggests the model may struggle with predictions when true radiation values 
are very close to zero. However, the R² value close to 1 and high correlation indicate a strong overall fit for the 
majority of the data points. The capped MAPE provides a more stable metric for percentage error in such cases.

## Hyperparameter Optimization

Optuna was used to find the best-performing hyperparameters by maximizing the R² score on the validation set over 20 trials. The best trial achieved an **R² of 0.9906**.

**Best Parameters Found:**
- `hidden_dim`: 256
- `num_layers`: 1
- `dropout_prob`: 0.1
- `learning_rate`: 0.000387
- `scheduler_type`: 'cosine'
- `loss_type`: 'mse'

While the Optuna search provided guidance, the final model was trained with a slightly different, manually-tuned architecture for robust performance.

## Model Capabilities

This architecture, combined with the data preprocessing and regularization techniques, aims to capture the complex 
temporal patterns in weather data for accurate radiation prediction, demonstrating strong performance as indicated by the 
R² and correlation metrics. The model also supports:

- **Monte Carlo Dropout**: For uncertainty quantification
- **Batch Processing**: For memory-efficient evaluation
- **Inverse Transformations**: For converting predictions back to original scale
- **Comprehensive Evaluation**: Including residual analysis and prediction plots

The model achieves excellent performance with an R² of 0.9893 on the test set, indicating it captures 98.93% of the variance in solar radiation data. 