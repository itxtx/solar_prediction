# Solar Prediction System: Data Flow and Model Lifecycle Analysis

## System Overview

This document provides a comprehensive high-level analysis of the solar prediction system's architecture, data flow, and model lifecycles based on the existing codebase examination.

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SOLAR PREDICTION SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐          │
│  │   Configuration  │    │   Data Pipeline  │    │   Model Engines  │          │
│  │     Module       │◄───┤      Module      ├───►│     Module       │          │
│  │   (config.py)    │    │   (data_prep.py) │    │ (lstm/gru/tdmc)  │          │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 2. Data Flow Pipeline

### 2.1 Raw Data Input
```
RAW WEATHER DATA
├── Solar Irradiance (GHI) - Target Variable
├── Temperature
├── Pressure  
├── Humidity
├── Wind Speed
├── Cloud Cover
├── Rain
├── Snow
├── Weather Type
└── Temporal Data
    ├── Timestamps
    ├── Sunrise/Sunset Times
    ├── Day Length
    └── Solar Position
```

### 2.2 Data Preprocessing Pipeline
```
┌─────────────────┐
│  Raw CSV Data   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Initial Setup   │◄── Column Standardization (STD_* constants)
│ & Validation    │◄── Time-based Sorting
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Time Feature    │◄── Cyclical Time Encoding (sin/cos)
│ Engineering     │◄── Solar Position Features
│                 │◄── Daylight Features
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Target Variable │◄── Piecewise Transform (GHI-specific)
│ Transformations │◄── Log Transform  
│                 │◄── Yeo-Johnson Power Transform
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Domain Feature  │◄── Low Target Indicators
│ Engineering     │◄── Domain-specific Features
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Feature         │◄── Minimal/Basic/All Feature Sets
│ Selection       │◄── Configuration-driven Selection
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Data Scaling    │◄── StandardScaler / MinMaxScaler
│                 │◄── Separate Feature/Target Scaling
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Sequence        │◄── Sliding Window Creation
│ Generation      │◄── Train/Validation/Test Split
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Model-Ready     │
│ Tensor Data     │
└─────────────────┘
```

### 2.3 Configuration-Driven Processing
The entire pipeline is controlled by centralized configuration classes:
- `DataInputConfig`: Column mappings and input specifications
- `DataTransformationConfig`: Transformation parameters
- `FeatureEngineeringConfig`: Feature creation and selection
- `ScalingConfig`: Scaling method selection
- `SequenceConfig`: Sequence generation parameters

## 3. Model Architectures and Lifecycles

### 3.1 LSTM Model (lstm.py)
```
LSTM MODEL LIFECYCLE
├── Initialization
│   ├── WeatherLSTM class instantiation
│   ├── Multi-layer LSTM with dropout
│   ├── Fully connected layers (64→16→8→1)
│   └── History tracking initialization
│
├── Training Phase
│   ├── Input Validation (3D tensors)
│   ├── Custom Loss Functions
│   │   ├── MSE Loss
│   │   ├── Combined Loss (MSE + MAPE)
│   │   └── Value-Aware Loss
│   ├── Optimizer: Adam with weight decay
│   ├── Learning Rate Scheduling
│   │   ├── ReduceLROnPlateau
│   │   └── CosineAnnealingLR
│   ├── Gradient Clipping
│   ├── Early Stopping
│   └── Best Model State Preservation
│
├── Evaluation Phase
│   ├── Memory-efficient batch processing
│   ├── Scaled metrics computation
│   ├── Inverse transformation pipeline
│   │   ├── Inverse scaling
│   │   ├── Inverse structural transforms
│   │   └── Domain-specific clipping
│   ├── Original scale metrics
│   └── Optional plotting
│
├── Uncertainty Quantification
│   ├── Monte Carlo Dropout
│   ├── Multiple forward passes
│   ├── Statistical aggregation
│   └── Confidence interval calculation
│
└── Model Persistence
    ├── State dictionary serialization
    ├── Hyperparameter preservation
    ├── Training history storage
    └── Transform info retention
```

### 3.2 GRU Model (gru.py)
```
GRU MODEL LIFECYCLE
├── Initialization
│   ├── WeatherGRU class instantiation
│   ├── Bidirectional GRU option
│   ├── Simpler FC architecture
│   └── History tracking
│
├── Training Phase
│   ├── Simplified loss functions (MSE/MAE)
│   ├── Similar optimization strategy
│   ├── Early stopping
│   └── Model state management
│
├── Evaluation & Uncertainty
│   ├── Batch-wise evaluation
│   ├── MC Dropout for uncertainty
│   └── Inverse transformation
│
└── Persistence
    ├── Similar to LSTM
    └── GRU-specific parameters
```

### 3.3 TDMC Model (tdmc.py)
```
TDMC MODEL LIFECYCLE
├── Initialization
│   ├── SolarTDMC class instantiation
│   ├── Time-dependent transition matrices
│   ├── Gaussian emission models
│   └── State initialization
│
├── Training Phase (EM Algorithm)
│   ├── Data preprocessing with time indices
│   ├── K-means initialization
│   │   ├── State assignments
│   │   ├── Emission parameter estimation
│   │   └── Transition matrix initialization
│   ├── Baum-Welch Algorithm
│   │   ├── Forward-backward algorithm
│   │   ├── E-step: State probabilities
│   │   ├── M-step: Parameter updates
│   │   └── Convergence checking
│   └── Regularization for numerical stability
│
├── Inference Phase
│   ├── Viterbi Algorithm (state prediction)
│   ├── Forward algorithm (forecasting)
│   ├── Mixture model predictions
│   └── Uncertainty quantification
│
├── State Analysis
│   ├── State characterization
│   ├── Transition visualization
│   ├── Emission distribution plotting
│   └── Time-dependent behavior analysis
│
└── Model Persistence
    ├── NumPy array serialization
    ├── Parameter preservation
    └── Scaler information storage
```

## 4. Data Flow Through Models

### 4.1 Common Input Pipeline
```
Raw Weather Data (CSV)
          ↓
Column Standardization (STD_* names)
          ↓
Time Feature Engineering
          ↓
Target Transformations (Log/Power/Piecewise)
          ↓
Feature Selection & Scaling
          ↓
Sequence Generation (Window-based)
          ↓
┌─────────────────────────────────────┐
│           MODEL INPUTS              │
│ X: (batch, sequence_length, features) │
│ y: (batch, 1)                        │
└─────────────────────────────────────┘
```

### 4.2 Model-Specific Processing

#### LSTM/GRU Flow:
```
3D Tensor Input → RNN Layers → Dropout → FC Layers → Output
     ↓
Training: Loss Computation → Backprop → Optimizer Step
     ↓
Evaluation: Prediction → Inverse Transform → Metrics
```

#### TDMC Flow:
```
2D Array Input → Time Index Extraction → State Modeling
     ↓
Training: EM Algorithm → Parameter Learning → Convergence
     ↓
Inference: State Sequence → Emission Prediction → Forecasting
```

## 5. Memory and Performance Considerations

### 5.1 Memory Management
- **Batch Processing**: All models use configurable batch sizes
- **Memory-Efficient Evaluation**: Streaming evaluation for large datasets
- **Gradient Accumulation**: Supports large effective batch sizes
- **MC Dropout**: Memory-aware uncertainty quantification

### 5.2 Computational Optimization
- **Early Stopping**: Prevents overfitting and reduces training time
- **Learning Rate Scheduling**: Adaptive learning rates
- **Gradient Clipping**: Numerical stability
- **Vectorized Operations**: NumPy/PyTorch optimizations

## 6. Configuration Management

### 6.1 Centralized Configuration (config.py)
```
SolarPredictionConfig
├── DataProcessingConfig
├── DataTransformationConfig  
├── FeatureEngineeringConfig
├── ScalingConfig
├── SequenceConfig
├── ModelsConfig
│   ├── LSTMConfig
│   ├── GRUConfig
│   └── TDMCConfig
├── PathsConfig
├── LoggingConfig
├── PlottingConfig
└── EvaluationConfig
```

### 6.2 Configuration Benefits
- **Type Safety**: Pydantic validation
- **Environment-Specific**: Development/Production/Testing configs
- **Centralized Control**: Single source of truth
- **Validation**: Built-in consistency checks

## 7. Transformation Pipeline and Inverse Operations

### 7.1 Forward Transformation Chain
```
Original Data
    ↓
Structural Transforms (Log/Power/Piecewise)
    ↓
Scaling (StandardScaler/MinMaxScaler)
    ↓
Model-Ready Data
```

### 7.2 Inverse Transformation Chain
```
Model Predictions (Scaled)
    ↓
Inverse Scaling
    ↓
Inverse Structural Transforms
    ↓
Domain Clipping (GHI bounds)
    ↓
Original Scale Predictions
```

## 8. Key Integration Points

### 8.1 Cross-Module Dependencies
- **data_prep.py** ↔ **config.py**: Configuration-driven processing
- **models** ↔ **data_prep.py**: Transform info preservation
- **models** ↔ **config.py**: Hyperparameter management
- **All modules** ↔ **config.py**: Centralized constants

### 8.2 State Management
- **Training State**: Model checkpoints, optimizer states
- **Transform State**: Scaler objects, transformation parameters
- **History State**: Training metrics, loss curves
- **Configuration State**: Runtime parameter management

## 9. Future Refactoring Considerations

### 9.1 Memory Optimization Opportunities
1. **Streaming Data Pipeline**: Process data in chunks
2. **Model Compression**: Quantization and pruning
3. **Efficient Caching**: Intelligent intermediate result caching
4. **Garbage Collection**: Explicit memory management

### 9.2 Bandwidth Optimization
1. **Data Compression**: Compressed data formats
2. **Feature Selection**: Automated relevance filtering
3. **Model Distillation**: Lightweight student models
4. **Incremental Learning**: Online learning capabilities

### 9.3 Architecture Improvements
1. **Plugin Architecture**: Modular model components
2. **Pipeline Parallelization**: Concurrent processing stages
3. **Distributed Training**: Multi-GPU/multi-node support
4. **Auto-ML Integration**: Automated hyperparameter tuning

## 10. Testing and Validation Framework

### 10.1 Current Testing Approach
- Input validation at each stage
- Configuration validation
- Numerical stability checks
- Memory leak prevention

### 10.2 Recommended Enhancements
- Unit tests for each transformation
- Integration tests for full pipeline
- Performance benchmarking
- Memory profiling automation

---

This analysis provides a comprehensive view of the current system architecture and serves as a foundation for future memory/bandwidth optimization efforts and system refactoring.
