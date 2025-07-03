
# Solar Radiation Forecasting using an LSTM Network

This project demonstrates a complete workflow for time-series forecasting using a Long Short-Term Memory (LSTM) neural network. The model is built with PyTorch and predicts solar Global Horizontal Irradiance (GHI) based on historical weather data. The process includes comprehensive data preparation, hyperparameter tuning with Optuna, final model training, and in-depth evaluation, including uncertainty estimation.

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
  (fc2): Linear(in_features=64, out_features=32, bias=True)
  (relu2): ReLU()
  (dropout3): Dropout(p=0.3, inplace=False)
  (fc3): Linear(in_features=32, out_features=1, bias=True)
)
```

### 4\. Training

The final model was trained with the following configuration:

  - **Optimizer**: Adam
  - **Epochs**: 100 (with early stopping patience of 30, ran for full 100 epochs)
  - **Batch Size**: 32
  - **Loss Function**: Mean Squared Error (MSE)
  - **Learning Rate**: $1.5 \times 10^{-4}$
  - **LR Scheduler**: Cosine Annealing with $T_{max}=100$
  - **Regularization**:
      - **Dropout**: `p=0.3` in both LSTM and fully-connected layers.
      - **Gradient Clipping**: Norm clipped to a max value of 1.0.
      - **Weight Decay**: $1 \times 10^{-5}$ (L2 regularization)

### 5\. Evaluation & Uncertainty

The model's performance was rigorously tested on the held-out test set. To quantify uncertainty, MC Dropout was employed. This involves running inference multiple times (`mc_samples=30`) with dropout layers enabled to generate a distribution of possible outputs for each input sequence. The mean of this distribution serves as the final prediction, and the standard deviation serves as the uncertainty measure.

-----

## Performance

The LSTM model demonstrates exceptional predictive performance on the unseen test data, achieving state-of-the-art results for solar radiation forecasting.

### Test Set Metrics

| Metric | Scaled Value | Original Scale |
|--------|-------------|----------------|
| **R-squared (R²)** | **0.9893** | **0.9893** |
| **RMSE** | 0.1041 | 0.1393 |
| **MAE** | 0.0602 | 0.0806 |
| **Capped MAPE** | 9.18% | 52.32% |
| **Correlation Coefficient** | 0.9914 | 0.9914 |

### Performance Analysis

- **Excellent Fit**: The model captures 98.93% of the variance in solar radiation data (R² = 0.9893)
- **High Correlation**: Near-perfect correlation (0.9914) between predicted and actual values
- **Low Error Rates**: RMSE of 0.1393 and MAE of 0.0806 on the original scale
- **Robust Predictions**: The model maintains consistent performance across both scaled and original scales

*Note: The high Capped MAPE (52.32%) on the original scale is expected and not concerning, as percentage-based errors become very large when the true radiation values are close to zero. This is a common phenomenon in solar radiation prediction where many values are near zero during nighttime or cloudy conditions.*


-----

## How to Use

### Dataset

The project includes two dataset options:

1. **Full Dataset** (`data/SolarPrediction.csv`): Complete dataset with ~32,600 records spanning 4 months (Sep-Dec 2016) with ~5-minute intervals.

2. **Sample Dataset** (`data/sample/SolarPrediction_sample.csv`): A 1-week subset (Sep 1-8, 2016) with ~1,850 records, ideal for:
   - Quick testing and development
   - CI/CD pipelines
   - Learning and experimentation
   - Resource-constrained environments

The notebooks automatically detect which dataset is available and load accordingly. If the full dataset is missing, the pipeline will seamlessly fall back to the sample dataset.

**Dataset Schema** (both files have identical structure):
- `UNIXTime`: Unix timestamp
- `Data`: Date string
- `Time`: Time string  
- `Radiation`: Solar radiation (GHI) - target variable
- `Temperature`: Air temperature
- `Pressure`: Atmospheric pressure
- `Humidity`: Relative humidity
- `WindDirection(Degrees)`: Wind direction
- `Speed`: Wind speed
- `TimeSunRise`: Sunrise time
- `TimeSunSet`: Sunset time

### Prerequisites

  - Python 3.8+
  - PyTorch
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib
  - Seaborn
  - Optuna

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Navigate to the notebooks directory: `cd notebooks`
4. Run the LSTM notebook: `jupyter notebook solar_data_notebook_lstm.ipynb`

### File Structure

```
.
├── data/
│   ├── SolarPrediction.csv              # Full dataset (32k+ records)
│   └── sample/
│       └── SolarPrediction_sample.csv   # Sample dataset (1-week, ~1.8k records)
├── solar_prediction/
│   ├── data_prep.py                     # Module for data preparation pipeline
│   ├── data_loader.py                   # Data loading utility with fallback logic
│   ├── lstm.py                          # Contains the WeatherLSTM model class and helper dataclasses
│   └── ...
├── notebooks/
│   ├── solar_data_notebook_lstm.ipynb   # Main Jupyter Notebook to run the project steps
│   ├── solar_data_gru.ipynb             # GRU model implementation
│   ├── solar_data_sarima.ipynb          # SARIMA model implementation
│   └── solar_data_tdmc.ipynb            # TDMC model implementation
├── models/
│   └── weather_lstm_model.pt            # Saved trained model weights
├── README.md                            # Main project documentation
└── README_LSTM.md                       # Detailed technical documentation for the LSTM model
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
