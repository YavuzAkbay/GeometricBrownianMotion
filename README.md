# Enhanced Stock Price Prediction using Ito's Lemma and Machine Learning

A sophisticated hybrid model that combines **stochastic calculus** (Ito's Lemma), **Geometric Brownian Motion (GBM)**, and **deep learning** to predict stock prices with enhanced accuracy. This project leverages LSTM networks with attention mechanisms to learn market dynamics and improve traditional financial modeling approaches.

## 🚀 Features

- **Hybrid LSTM-GBM Architecture**: Combines deep learning with stochastic differential equations
- **Enhanced Technical Analysis**: 20+ technical indicators including RSI, MACD, Bollinger Bands
- **Multi-head Attention Mechanism**: Captures complex temporal dependencies in market data
- **Stochastic Parameter Prediction**: ML-powered drift and volatility estimation
- **Comprehensive Visualization**: Advanced plotting for model performance and market analysis
- **Risk Analysis**: Monte Carlo simulations with confidence intervals
- **Real-time Predictions**: 6-month forecasting with uncertainty quantification

## 📊 Model Architecture

The `EnhancedStockPredictor` class implements a sophisticated neural network that predicts three key parameters:

- **Price Prediction**: Direct stock price forecasting
- **Volatility Estimation**: Dynamic volatility modeling using sigmoid activation
- **Drift Calculation**: Market trend estimation with tanh activation

### Key Components

1. **LSTM Layers**: Capture sequential patterns in financial time series
2. **Multi-head Attention**: Focus on relevant historical periods
3. **Feature Extraction**: Dense layers for complex pattern recognition
4. **Parameter Prediction**: Separate heads for price, volatility, and drift

## 🛠 Installation

1. Clone the repository

```bash
git clone https://github.com/YavuzAkbay/GeometricBrownianMotion
cd GeometricBrownianMotion
```

2. Install required packages

```bash
pip install -r requirements.txt
```

3. For GPU acceleration (optional):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```


## 📈 Usage

### Basic Usage

```python
from stock_predictor import train_enhanced_model, enhanced_analysis_and_visualization
```

Train the model for a specific ticker

```python
ticker = "AAPL" # or any stock symbol
model, scaler_X, scaler_y, enhanced_data, feature_columns, metrics = train_enhanced_model(
ticker,
sequence_length=60,
epochs=100
)
```

Generate predictions and visualizations

```python
results = enhanced_analysis_and_visualization(
ticker, model, scaler_X, scaler_y,
enhanced_data, feature_columns,
forecast_months=6
)
```


### Advanced Configuration

Customize model parameters

```python
model = EnhancedStockPredictor(
input_size=22, # Number of features
hidden_size=128, # LSTM hidden units
num_layers=2, # LSTM layers
dropout=0.2 # Dropout rate
)
```

Adjust training parameters

```python
train_enhanced_model(
ticker="TSLA",
sequence_length=90, # Longer sequences for more context
epochs=200 # More training epochs
)
```


## 🔬 Technical Indicators

The model incorporates comprehensive technical analysis:

### Price-based Indicators
- Moving Averages (5, 10, 20, 50 periods)
- Moving Average Ratios
- Bollinger Bands Position
- Price lag features (1, 2, 3, 5 periods)

### Momentum Indicators
- RSI (Relative Strength Index)
- MACD and MACD Signal
- Price change and log returns

### Volatility Measures
- Rolling volatility (10, 20 periods)
- High-Low ratio
- Open-Close ratio

### Volume Analysis
- Volume moving average
- Volume ratio analysis
- Volume lag features

## 📊 Model Output

The system provides comprehensive analysis including:

### Visualizations
1. **Training Performance**: Loss curves and prediction accuracy
2. **Price Forecasting**: ML vs Traditional GBM comparison
3. **Risk Analysis**: Probability distributions and percentile analysis
4. **Technical Analysis**: Price charts with indicators
5. **Volatility Analysis**: Historical vs predicted volatility

### Metrics
- **Expected Returns**: ML-enhanced vs traditional estimates
- **Volatility Predictions**: Dynamic volatility modeling
- **Profit Probability**: Statistical likelihood of positive returns
- **Risk Percentiles**: 5th, 25th, 50th, 75th, 95th percentile analysis

## 🧮 Mathematical Foundation

The model is based on the **Geometric Brownian Motion** equation derived from Ito's Lemma:

dS = μS dt + σS dW

Where:
- `S`: Stock price
- `μ`: Drift parameter (predicted by ML)
- `σ`: Volatility parameter (predicted by ML)
- `dW`: Wiener process (random walk)

The ML model learns to predict `μ` and `σ` dynamically based on market conditions, improving upon static parameter assumptions in traditional models.

## 🎯 Performance Features

- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Gradient Clipping**: Prevents exploding gradients during training
- **Learning Rate Scheduling**: Adaptive learning rate reduction
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Batch Processing**: Efficient mini-batch training


## 🔧 Configuration Options

### Model Parameters
- `sequence_length`: Historical data window (default: 60 days)
- `hidden_size`: LSTM hidden units (default: 128)
- `num_layers`: LSTM depth (default: 2)
- `dropout`: Regularization rate (default: 0.2)

### Training Parameters
- `epochs`: Training iterations (default: 100)
- `batch_size`: Mini-batch size (default: 32)
- `learning_rate`: Initial learning rate (default: 0.001)

### Prediction Parameters
- `forecast_months`: Prediction horizon (default: 6 months)
- `num_simulations`: Monte Carlo runs (default: 1000)

## 📊 Example Results

For XLU (Utilities Select Sector SPDR Fund):
- **ML Expected Return**: +5.23%
- **Traditional Expected Return**: +3.87%
- **Volatility Improvement**: 2.15% reduction in prediction uncertainty
- **Profit Probability**: 68.3% (ML) vs 61.2% (Traditional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the GPL v3 - see the (https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. **Do not use this for actual trading decisions without proper risk management and professional financial advice.** Past performance does not guarantee future results. Trading stocks involves substantial risk of loss.

## 🙏 Acknowledgments

- **Ito's Lemma**: Foundation of stochastic calculus in finance
- **PyTorch**: Deep learning framework
- **yfinance**: Financial data API
- **scikit-learn**: Machine learning utilities

## 📧 Contact

Yavuz Akbay - akbay.yavuz@gmail.com

---

⭐️ If this project helped with your financial analysis, please consider giving it a star!

**Built with ❤️ for the intersection of mathematics, machine learning, and finance**
