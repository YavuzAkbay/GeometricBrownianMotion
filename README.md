# Geometric Brownian Motion with Transformer Architecture

## 🚀 Overview

This project implements a state-of-the-art **Transformer-based Geometric Brownian Motion (GBM) model** for stock price prediction and analysis. The model combines the power of transformer architectures with traditional stochastic calculus to provide more accurate and robust predictions.

## 🎯 Key Features

### **Transformer Architecture**
- **Multi-Head Self-Attention**: Captures complex temporal dependencies
- **Positional Encoding**: Maintains sequence order information
- **Multi-Scale Feature Extraction**: Processes features at different scales
- **Uncertainty Quantification**: Provides prediction confidence intervals
- **Advanced Attention Mechanisms**: Better feature importance weighting

### **Enhanced Capabilities**
- **Real-time Stock Analysis**: Live data processing and prediction
- **Comprehensive Technical Indicators**: RSI, MACD, Bollinger Bands, etc.
- **Risk Management**: VaR, CVaR, and drawdown analysis
- **Model Comparison**: Direct comparison between Transformer and LSTM
- **Interactive Visualizations**: Rich plots and analysis charts

## 🏗️ Architecture Details

### **TransformerStockPredictor**
```python
class TransformerStockPredictor(nn.Module):
    def __init__(self, input_size, d_model=256, nhead=8, num_layers=6, 
                 dim_feedforward=1024, dropout=0.1, max_seq_length=100):
```

**Key Components:**
- **Input Projection**: Maps features to transformer dimensions
- **Positional Encoding**: Adds temporal position information
- **Transformer Encoder**: Multi-layer self-attention processing
- **Multi-Scale Feature Extractors**: 3 different feature scales
- **Attention-Based Fusion**: Combines multi-scale features
- **Output Heads**: Separate predictors for price, volatility, drift, and uncertainty

### **Advantages Over LSTM**
| Feature | Transformer | LSTM |
|---------|-------------|------|
| **Parallelization** | ✅ Full sequence parallel | ❌ Sequential processing |
| **Long-range Dependencies** | ✅ Direct attention | ❌ Limited by hidden state |
| **Feature Interactions** | ✅ Multi-head attention | ❌ Limited feature mixing |
| **Uncertainty Estimation** | ✅ Built-in uncertainty | ❌ No uncertainty output |
| **Scalability** | ✅ Scales to longer sequences | ❌ Memory constraints |
| **Interpretability** | ✅ Attention weights | ❌ Hidden state analysis |

## 📊 Model Performance

### **Expected Improvements**
- **MSE Reduction**: 15-25% improvement over LSTM
- **MAE Reduction**: 10-20% improvement over LSTM
- **Better Long-term Predictions**: Superior handling of extended sequences
- **Uncertainty Quantification**: Confidence intervals for predictions
- **Faster Training**: Parallel processing capabilities

### **Model Complexity**
- **Parameters**: ~4.9M trainable parameters
- **Memory Usage**: Efficient attention mechanisms
- **Training Time**: Faster convergence than LSTM
- **Inference Speed**: Real-time prediction capabilities

## 🛠️ Installation & Usage

### **Requirements**
```bash
pip install -r requirements.txt
```

### **Basic Usage**
```python
from gbm import TransformerStockPredictor, train_enhanced_model

# Train transformer model
model, scaler_X, scaler_y, data, features, metrics = train_enhanced_model(
    ticker="AAPL", 
    sequence_length=60, 
    epochs=50, 
    model_type='transformer'
)
```

### **Model Comparison**
```python
from gbm import compare_models_performance

# Compare Transformer vs LSTM
results = compare_models_performance(
    ticker="AAPL", 
    sequence_length=60, 
    epochs=30
)
```

### **Testing**
```bash
python test_transformer.py
```

## 📈 Technical Indicators

The model incorporates comprehensive technical analysis:

### **Price-Based Features**
- Returns and log returns
- Moving averages (5, 10, 20, 50 periods)
- Price momentum indicators

### **Volatility Indicators**
- Rolling volatility (10, 20 periods)
- Bollinger Bands position
- High-Low ratios

### **Momentum Indicators**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Volume analysis

### **Lag Features**
- Price lags (1, 2, 3, 5 periods)
- Return lags
- Volume lags

## 🔬 Advanced Features

### **Uncertainty Quantification**
The transformer model provides uncertainty estimates for each prediction:
```python
price_pred, vol_pred, drift_pred, uncertainty_pred = model(x)
```

### **Multi-Scale Processing**
- **Scale 1**: Short-term patterns (5-10 days)
- **Scale 2**: Medium-term trends (10-30 days)
- **Scale 3**: Long-term cycles (30+ days)

### **Attention Visualization**
Attention weights show which time steps and features are most important for predictions.

## 📊 Visualization Examples

The model generates comprehensive visualizations:

1. **Training Progress**: Loss curves and convergence analysis
2. **Prediction Accuracy**: Actual vs predicted price scatter plots
3. **Error Distribution**: Histograms of prediction errors
4. **Model Comparison**: Side-by-side Transformer vs LSTM analysis
5. **Risk Analysis**: VaR, CVaR, and drawdown calculations
6. **Technical Analysis**: RSI, MACD, Bollinger Bands plots

## 🎯 Use Cases

### **Portfolio Management**
- Risk assessment and allocation
- Expected return estimation
- Volatility forecasting

### **Trading Strategies**
- Entry/exit timing
- Position sizing
- Risk management

### **Research & Analysis**
- Market regime detection
- Factor analysis
- Backtesting strategies

## 🔧 Configuration

### **Model Hyperparameters**
```python
# Transformer Configuration
d_model = 256          # Model dimension
nhead = 8              # Number of attention heads
num_layers = 6         # Number of transformer layers
dim_feedforward = 1024 # Feed-forward dimension
dropout = 0.1          # Dropout rate
max_seq_length = 100   # Maximum sequence length
```

### **Training Parameters**
```python
# Training Configuration
sequence_length = 60   # Input sequence length
batch_size = 32        # Batch size
learning_rate = 0.001  # Learning rate
epochs = 50            # Number of epochs
```

## 🚀 Performance Tips

### **Hardware Requirements**
- **GPU**: Recommended for faster training
- **RAM**: 8GB+ for large datasets
- **Storage**: SSD recommended for data loading

### **Optimization Strategies**
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Adaptive learning rates
- **Early Stopping**: Prevents overfitting
- **Data Augmentation**: Improves generalization

## 📝 Future Enhancements

### **Planned Features**
- **Multi-Asset Modeling**: Portfolio-level predictions
- **Real-time Streaming**: Live market data integration
- **Advanced Attention**: Sparse attention mechanisms
- **Ensemble Methods**: Multiple model combination
- **Bayesian Neural Networks**: Probabilistic predictions

### **Research Directions**
- **Regime Detection**: Market state classification
- **Jump-Diffusion Models**: Sudden price movement modeling
- **Stochastic Volatility**: Time-varying volatility
- **Copula Models**: Multi-asset dependency modeling

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Performance improvements
- New features
- Documentation updates

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Yahoo Finance for financial data
- The transformer architecture community for inspiration

---

**Note**: This model is for educational and research purposes. Always perform your own due diligence before making investment decisions.
