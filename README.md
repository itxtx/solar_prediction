
# Solar Radiation Forecasting using an LSTM Network

This project demonstrates a complete workflow for time-series forecasting using a Long Short-Term Memory (LSTM) neural network. The model is built with PyTorch and predicts solar Global Horizontal Irradiance (GHI) based on historical weather data. The process includes comprehensive data preparation, hyperparameter tuning with Optuna, final model training, and in-depth evaluation, including uncertainty estimation.

## Table of Contents

  - [Project Overview]
  - [Methodology]
      - [1. Data Preparation]
      - [2. Hyperparameter Optimization]
      - [3. Model Architecture]
      - [4. Training]
      - [5. Evaluation & Uncertainty]
  - [Performance]
      - [Test Set Metrics]
      - [Visualizations]
  - [How to Use]
      - [Prerequisites]
      - [File Structure]
      - [Running the Project]
  - [Dependencies]

## Project Overview

The primary goal is to forecast the 'GHI' (a measure of solar radiation) using a sequence of past weather conditions. This is a time-series regression task. The project leverages an LSTM network to capture temporal dependencies in the data.

The workflow consists of:

1.  **Data Loading and Preprocessing**: Ingesting and cleaning the `solar_weather.csv` dataset.
2.  **Feature Engineering**: Creating new features from existing data to improve model performance (e.g., cyclical time features).
3.  **Data Transformation**: Applying a `Yeo-Johnson` power transform to the target variable to stabilize variance and handling non-normality. Features are scaled to a [0, 1] range.
4.  **Hyperparameter Tuning**: Using `Optuna` to systematically find the optimal model architecture and training parameters.
5.  **Model Training**: Training the final LSTM model on a dedicated GPU (`mps` or `cuda`) with a cosine annealing learning rate scheduler and early stopping.
6.  **Model Evaluation**: Assessing the model's performance on a held-out test set using metrics like R², RMSE, and MAE.
7.  **Uncertainty Estimation**: Using Monte Carlo (MC) Dropout during inference to quantify the model's prediction uncertainty.

-----

## Methodology

### 1\. Data Preparation

Data preparation is handled by the `data_prep.py` module, which performs the following steps:

  - **Renaming & Sorting**: Columns are renamed for clarity, and the data is sorted by timestamp.
  - **Feature Engineering**: `SolarElevation` is engineered as a proxy for the sun's position.
  - **Target Transformation**: The target variable `GHI` is transformed using a **Yeo-Johnson** power transform (`lambda = -0.1973`) to make its distribution more Gaussian.
  - **Feature Selection**: The model uses the following 10 features for prediction:
    ```
    ['Cloudcover', 'Humidity', 'Pressure', 'Radiation', 'Radiation_is_low', 'Rain', 'Temperature', 'TimeMinutesCos', 'TimeMinutesSin', 'WindSpeed']
    ```
  - **Scaling**:
      - Features are scaled using `MinMaxScaler`.
      - The transformed target is scaled using `StandardScaler`.
  - **Sequencing**: The data is converted into sequences with a window size of **24 time steps**.
  - **Data Splitting**: The dataset is split into training (63.75%), validation (21.25%), and test (15%) sets.

### 2\. Hyperparameter Optimization

`Optuna` was used to find the best-performing hyperparameters by maximizing the R² score on the validation set over 20 trials. The best trial achieved an **R² of 0.9906**.

  - **Best Parameters Found:**
      - `hidden_dim`: 256
      - `num_layers`: 1
      - `dropout_prob`: 0.1
      - `learning_rate`: 0.000387
      - `scheduler_type`: 'cosine'
      - `loss_type`: 'mse'

### 3\. Model Architecture

While the Optuna search provided guidance, the final model was trained with a slightly different, manually-tuned architecture for robust performance. The final model is a `WeatherLSTM` with the following structure:

```
WeatherLSTM(
  (lstm): LSTM(10, 256, num_layers=2, batch_first=True, dropout=0.3)
  (dropout1): Dropout(p=0.3, inplace=False)
  (fc1): Linear(in_features=256, out_features=64, bias=True)
  (relu): ReLU()
  (dropout2): Dropout(p=0.3, inplace=False)
  (fc2): Linear(in_features=64, out_features=128, bias=True)
  (relu2): ReLU()
  (dropout3): Dropout(p=0.3, inplace=False)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
)
```

### 4\. Training

The final model was trained with the following configuration:

  - **Optimizer**: Adam
  - **Epochs**: 100 (with early stopping patience of 30, stopped at epoch 89)
  - **Batch Size**: 32
  - **Loss Function**: Mean Squared Error (MSE)
  - **Learning Rate**: $1.5 \\times 10^{-4}$
  - **LR Scheduler**: Cosine Annealing
  - **Regularization**:
      - **Dropout**: `p=0.3` in both LSTM and fully-connected layers.
      - **Gradient Clipping**: Norm clipped to a max value of 1.0.

### 5\. Evaluation & Uncertainty

The model's performance was rigorously tested on the held-out test set. To quantify uncertainty, MC Dropout was employed. This involves running inference multiple times (`mc_samples=30`) with dropout layers enabled to generate a distribution of possible outputs for each input sequence. The mean of this distribution serves as the final prediction, and the standard deviation serves as the uncertainty measure.

-----

## Performance

The model demonstrates excellent predictive power on the unseen test data.

### Test Set Metrics

| Metric | Scaled Value | Original Value |
| :--- | :--- | :--- |
| **R-squared (R²)** | **0.9893** | **0.9893** |
| **RMSE** | 0.1041 | 0.1393 |
| **MAE** | 0.0602 | 0.0806 |
| **Capped MAPE** | 9.18% | 52.32% |

*Note: The high Capped MAPE on the original scale is expected, as percentage-based errors become very large when the true radiation values are close to zero.*

### Visualizations

#### Training History

The training and validation loss decreased consistently, with the model stopping early at epoch 89 as validation performance plateaued, preventing overfitting.

*(Image placeholder for `model.plot_training_history()`)*

#### Predictions vs. Actuals

The model's predictions align almost perfectly with the actual values on the test set, as shown by the tight clustering around the ideal y=x line.

*(Image placeholder for the Predictions vs. Actuals scatter plot)*

#### Time Series Forecast

A plot of the predictions overlaid on the actuals for a sample of the test set shows the model's ability to capture the dynamic patterns of solar radiation.

*(Image placeholder for the time series forecast plot)*

#### Prediction with Uncertainty

The MC Dropout method provides a 95% confidence interval, giving insight into the model's certainty for each prediction.

*(Image placeholder for `plot_prediction_with_uncertainty()`)*

-----

## How to Use

### Prerequisites

  - Python 3.8+
  - PyTorch
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib
  - Seaborn
  - Optuna

### File Structure

```
.
├── solar_weather.csv       # Dataset
├── data_prep.py            # Module for data preparation pipeline
├── lstm.py                 # Contains the WeatherLSTM model class and helper dataclasses
├── main_notebook.ipynb     # Main Jupyter Notebook to run the project steps
└── weather_lstm_model.pt   # Saved trained model weights
```


-----

## Dependencies

  - **`numpy`**: For numerical operations.
  - **`pandas`**: For data manipulation and loading the CSV.
  - **`torch`**: For building and training the LSTM model.
  - **`matplotlib` & `seaborn`**: For data visualization.
  - **`scikit-learn`**: For data scaling, transformations, and metrics.
  - **`optuna`**: For automated hyperparameter tuning.
  - **`logging`**: For tracking the pipeline's progress.