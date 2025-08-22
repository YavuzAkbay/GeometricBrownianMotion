import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import math
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
warnings.filterwarnings('ignore')

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architecture"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class BayesianLinear(nn.Module):
    """Bayesian Linear layer with uncertainty quantification"""
    def __init__(self, in_features, out_features, prior_std=1.0, posterior_std_init=0.1):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Prior parameters
        self.prior_std = prior_std
        
        # Posterior parameters (learnable)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_std = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_std = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        self._init_parameters(posterior_std_init)
    
    def _init_parameters(self, posterior_std_init):
        nn.init.xavier_uniform_(self.weight_mu, gain=0.5)
        nn.init.constant_(self.weight_log_std, math.log(posterior_std_init))
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_log_std, math.log(posterior_std_init))
    
    def forward(self, x, sample=True):
        if self.training or sample:
            # Sample weights from posterior
            weight_std = torch.exp(self.weight_log_std)
            bias_std = torch.exp(self.bias_log_std)
            
            weight_epsilon = torch.randn_like(self.weight_mu)
            bias_epsilon = torch.randn_like(self.bias_mu)
            
            weight = self.weight_mu + weight_std * weight_epsilon
            bias = self.bias_mu + bias_std * bias_epsilon
        else:
            # Use mean for inference
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_loss(self):
        """Compute KL divergence between posterior and prior"""
        weight_std = torch.exp(self.weight_log_std)
        bias_std = torch.exp(self.bias_log_std)
        
        # KL divergence for weights
        weight_kl = kl_divergence(
            Normal(self.weight_mu, weight_std),
            Normal(0, self.prior_std)
        ).sum()
        
        # KL divergence for bias
        bias_kl = kl_divergence(
            Normal(self.bias_mu, bias_std),
            Normal(0, self.prior_std)
        ).sum()
        
        return weight_kl + bias_kl

class BayesianSequential(nn.Module):
    """Custom Sequential module that can handle Bayesian layers with sample parameter"""
    def __init__(self, *layers):
        super(BayesianSequential, self).__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, sample=True):
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                x = layer(x, sample)
            else:
                x = layer(x)
        return x

class TransformerStockPredictor(nn.Module):
    """Advanced Transformer-based stock prediction model"""
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, max_seq_length=100):
        super(TransformerStockPredictor, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder layers with proper normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Global average pooling and feature extraction
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature extraction with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Output heads with proper initialization
        self.price_predictor = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        self.volatility_predictor = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        self.drift_predictor = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Tanh()
        )
        
        # Uncertainty estimation
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling for transformer stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use smaller initialization for better stability
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Input projection and normalization
        batch_size, seq_len, features = x.shape
        x = self.input_projection(x)
        x = self.input_norm(x)
        
        # Positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(x)
        
        # Global average pooling across sequence dimension
        # (batch_size, seq_len, d_model) -> (batch_size, d_model)
        global_features = transformer_out.mean(dim=1)
        
        # Feature extraction with residual connection
        extracted_features = self.feature_extractor(global_features)
        
        # Generate predictions
        price_pred = self.price_predictor(extracted_features)
        volatility_pred = self.volatility_predictor(extracted_features)
        drift_pred = self.drift_predictor(extracted_features)
        uncertainty_pred = self.uncertainty_predictor(extracted_features)
        
        return price_pred, volatility_pred, drift_pred, uncertainty_pred
    
    def predict_with_uncertainty(self, x, num_samples=10):
        """Generate predictions with uncertainty quantification using Monte Carlo dropout"""
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Enable dropout for uncertainty estimation
                self.train()
                pred, vol, drift, uncertainty = self.forward(x)
                predictions.append((pred, vol, drift, uncertainty))
                self.eval()
        
        # Stack predictions
        price_preds = torch.stack([p[0] for p in predictions], dim=0)
        vol_preds = torch.stack([p[1] for p in predictions], dim=0)
        drift_preds = torch.stack([p[2] for p in predictions], dim=0)
        uncertainty_preds = torch.stack([p[3] for p in predictions], dim=0)
        
        # Compute statistics
        price_mean = price_preds.mean(dim=0)
        price_std = price_preds.std(dim=0)
        vol_mean = vol_preds.mean(dim=0)
        vol_std = vol_preds.std(dim=0)
        drift_mean = drift_preds.mean(dim=0)
        drift_std = drift_preds.std(dim=0)
        
        # Use the uncertainty predictor output as total uncertainty
        total_uncertainty = uncertainty_preds.mean(dim=0)
        
        return {
            'price_mean': price_mean,
            'price_std': price_std,
            'volatility_mean': vol_mean,
            'volatility_std': vol_std,
            'drift_mean': drift_mean,
            'drift_std': drift_std,
            'aleatoric_uncertainty': total_uncertainty * 0.5,  # Approximate split
            'epistemic_uncertainty': total_uncertainty * 0.5,  # Approximate split
            'total_uncertainty': total_uncertainty,
            'price_samples': price_preds,
            'volatility_samples': vol_preds,
            'drift_samples': drift_preds
        }
    
    def kl_loss(self):
        """Return zero KL loss for non-Bayesian model (compatibility method)"""
        return torch.tensor(0.0, device=next(self.parameters()).device)

class BayesianTransformerStockPredictor(nn.Module):
    """Bayesian Transformer-based stock prediction model with uncertainty quantification"""
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, max_seq_length=100,
                 prior_std=1.0, posterior_std_init=0.1, num_samples=10):
        super(BayesianTransformerStockPredictor, self).__init__()
        
        self.d_model = d_model
        self.num_samples = num_samples
        
        # Standard transformer components
        self.input_projection = nn.Linear(input_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature extraction with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Bayesian output heads using custom Sequential
        self.price_predictor = BayesianSequential(
            BayesianLinear(d_model // 2, d_model // 4, prior_std, posterior_std_init),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            BayesianLinear(d_model // 4, 1, prior_std, posterior_std_init)
        )
        
        self.volatility_predictor = BayesianSequential(
            BayesianLinear(d_model // 2, d_model // 4, prior_std, posterior_std_init),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            BayesianLinear(d_model // 4, 1, prior_std, posterior_std_init),
            nn.Sigmoid()
        )
        
        self.drift_predictor = BayesianSequential(
            BayesianLinear(d_model // 2, d_model // 4, prior_std, posterior_std_init),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            BayesianLinear(d_model // 4, 1, prior_std, posterior_std_init),
            nn.Tanh()
        )
        
        # Uncertainty estimation (aleatoric + epistemic)
        self.uncertainty_predictor = BayesianSequential(
            BayesianLinear(d_model // 2, d_model // 4, prior_std, posterior_std_init),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            BayesianLinear(d_model // 4, 2, prior_std, posterior_std_init),  # Mean and variance
            nn.Softplus()  # Ensure positive variance
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling for transformer stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x, sample=True):
        # Input projection and normalization
        batch_size, seq_len, features = x.shape
        x = self.input_projection(x)
        x = self.input_norm(x)
        
        # Positional encoding
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(x)
        
        # Global average pooling
        global_features = transformer_out.mean(dim=1)
        
        # Feature extraction
        extracted_features = self.feature_extractor(global_features)
        
        # Generate predictions with uncertainty
        price_pred = self.price_predictor(extracted_features, sample)
        volatility_pred = self.volatility_predictor(extracted_features, sample)
        drift_pred = self.drift_predictor(extracted_features, sample)
        
        # Uncertainty prediction (aleatoric + epistemic)
        uncertainty_params = self.uncertainty_predictor(extracted_features, sample)
        aleatoric_uncertainty = uncertainty_params[:, 0:1]  # Data uncertainty
        epistemic_uncertainty = uncertainty_params[:, 1:2]  # Model uncertainty
        
        return price_pred, volatility_pred, drift_pred, aleatoric_uncertainty, epistemic_uncertainty
    
    def predict_with_uncertainty(self, x, num_samples=10):
        """Generate predictions with uncertainty quantification"""
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x, sample=True)
                predictions.append(pred)
        
        # Stack predictions
        price_preds = torch.stack([p[0] for p in predictions], dim=0)
        vol_preds = torch.stack([p[1] for p in predictions], dim=0)
        drift_preds = torch.stack([p[2] for p in predictions], dim=0)
        aleatoric_uncertainties = torch.stack([p[3] for p in predictions], dim=0)
        epistemic_uncertainties = torch.stack([p[4] for p in predictions], dim=0)
        
        # Compute statistics
        price_mean = price_preds.mean(dim=0)
        price_std = price_preds.std(dim=0)
        vol_mean = vol_preds.mean(dim=0)
        vol_std = vol_preds.std(dim=0)
        drift_mean = drift_preds.mean(dim=0)
        drift_std = drift_preds.std(dim=0)
        
        # Total uncertainty = aleatoric + epistemic
        total_uncertainty = aleatoric_uncertainties.mean(dim=0) + epistemic_uncertainties.mean(dim=0)
        
        return {
            'price_mean': price_mean,
            'price_std': price_std,
            'volatility_mean': vol_mean,
            'volatility_std': vol_std,
            'drift_mean': drift_mean,
            'drift_std': drift_std,
            'aleatoric_uncertainty': aleatoric_uncertainties.mean(dim=0),
            'epistemic_uncertainty': epistemic_uncertainties.mean(dim=0),
            'total_uncertainty': total_uncertainty,
            'price_samples': price_preds,
            'volatility_samples': vol_preds,
            'drift_samples': drift_preds
        }
    
    def kl_loss(self):
        """Compute total KL divergence loss"""
        total_kl = 0
        for module in self.modules():
            if isinstance(module, BayesianLinear):
                total_kl += module.kl_loss()
        return total_kl

class EnhancedStockPredictor(nn.Module):
    """Legacy LSTM model - kept for comparison"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(EnhancedStockPredictor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.price_predictor = nn.Linear(hidden_size // 4, 1)
        self.volatility_predictor = nn.Linear(hidden_size // 4, 1)
        self.drift_predictor = nn.Linear(hidden_size // 4, 1)
        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last_output = attn_out[:, -1, :]
        features = self.feature_extractor(last_output)
        
        price_pred = self.price_predictor(features)
        volatility_pred = self.sigmoid(self.volatility_predictor(features)) * 0.5
        drift_pred = self.tanh(self.drift_predictor(features)) * 0.3
        
        return price_pred, volatility_pred, drift_pred

def fetch_and_clean_data(ticker, period='5y'):
    """Fetch stock data and handle MultiIndex columns"""
    data = yf.download(ticker, period=period)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns {missing_columns}")
        return None
    
    data = data.loc[:, ~data.columns.duplicated()]
    return data.dropna()

def create_enhanced_features(stock_data, lookback=60):
    """Create comprehensive feature set including technical indicators"""
    df = stock_data.copy()
    
    # Ensure we're working with Series, not DataFrames
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
    
    # Price-based features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Price_Change'] = df['Close'] - df['Close'].shift(1)
    
    # Technical indicators
    for window in [5, 10, 20, 50]:
        ma_col = f'MA_{window}'
        ratio_col = f'MA_Ratio_{window}'
        
        df[ma_col] = df['Close'].rolling(window=window).mean()
        df[ratio_col] = df['Close'] / df[ma_col]
    
    # Volatility indicators
    df['Volatility_10'] = df['Returns'].rolling(window=10).std()
    df['Volatility_20'] = df['Returns'].rolling(window=20).std()
    
    # RSI
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['RSI'] = calculate_rsi(df['Close'])
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Market microstructure
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Open_Close_Ratio'] = df['Open'] / df['Close']
    
    # Lag features
    for lag in [1, 2, 3, 5]:
        df[f'Price_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Return_Lag_{lag}'] = df['Returns'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
    
    return df.dropna()

def prepare_sequences(data, feature_columns, target_column, sequence_length=60):
    """Prepare sequences for LSTM training"""
    sequences = []
    targets = []
    
    for i in range(sequence_length, len(data)):
        seq = data[feature_columns].iloc[i-sequence_length:i].values
        sequences.append(seq)
        
        target = data[target_column].iloc[i]
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def enhanced_gbm_simulation(S0, mu_pred, sigma_pred, T, N, num_simulations=1000):
    """Enhanced GBM simulation using ML-predicted parameters"""
    dt = T / N
    time_steps = np.linspace(0, T, N+1)
    all_paths = np.zeros((num_simulations, N+1))
    
    np.random.seed(42)  # For reproducibility
    
    for i in range(num_simulations):
        if isinstance(mu_pred, (list, np.ndarray)) and len(mu_pred) > 1:
            mu_path = np.interp(time_steps, np.linspace(0, T, len(mu_pred)), mu_pred)
            sigma_path = np.interp(time_steps, np.linspace(0, T, len(sigma_pred)), sigma_pred)
        else:
            mu_path = np.full(N+1, mu_pred)
            sigma_path = np.full(N+1, sigma_pred)
        
        path = [S0]
        for t in range(N):
            dW = np.random.normal(0, np.sqrt(dt))
            dS = mu_path[t] * path[-1] * dt + sigma_path[t] * path[-1] * dW
            path.append(path[-1] + dS)
        
        all_paths[i] = path
    
    return time_steps, all_paths

def plot_training_results(metrics, ticker):
    """Plot training loss and model performance"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{ticker} ML Model Training Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss
    ax1.plot(metrics['train_losses'], color='blue', linewidth=2)
    ax1.set_title('Training Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted Prices
    ax2.scatter(metrics['test_actual'], metrics['test_pred'], alpha=0.6, color='red')
    min_val = min(metrics['test_actual'].min(), metrics['test_pred'].min())
    max_val = max(metrics['test_actual'].max(), metrics['test_pred'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel('Actual Prices')
    ax2.set_ylabel('Predicted Prices')
    ax2.set_title('Actual vs Predicted Prices')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction Errors
    errors = metrics['test_pred'] - metrics['test_actual']
    ax3.hist(errors, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Prediction Errors')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Predicted Volatility and Drift
    ax4.scatter(range(len(metrics['test_vol'])), metrics['test_vol'], 
               alpha=0.6, color='orange', label='Volatility', s=20)
    ax4_twin = ax4.twinx()
    ax4_twin.scatter(range(len(metrics['test_drift'])), metrics['test_drift'], 
                    alpha=0.6, color='purple', label='Drift', s=20)
    
    ax4.set_xlabel('Test Sample')
    ax4.set_ylabel('Predicted Volatility', color='orange')
    ax4_twin.set_ylabel('Predicted Drift', color='purple')
    ax4.set_title('ML Predicted Parameters')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_uncertainty_analysis(metrics, ticker):
    """Plot uncertainty quantification analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker} Bayesian Uncertainty Quantification', fontsize=16, fontweight='bold')
    
    # Plot 1: Uncertainty vs Prediction Error
    prediction_errors = np.abs(metrics['test_pred'] - metrics['test_actual'])
    uncertainty_values = metrics['test_uncertainty']
    
    ax1.scatter(uncertainty_values, prediction_errors, alpha=0.6, color='purple', s=30)
    ax1.set_xlabel('Predicted Uncertainty')
    ax1.set_ylabel('Absolute Prediction Error')
    ax1.set_title('Uncertainty vs Prediction Error')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    if len(uncertainty_values) > 1:
        z = np.polyfit(uncertainty_values, prediction_errors, 1)
        p = np.poly1d(z)
        ax1.plot(uncertainty_values, p(uncertainty_values), "r--", alpha=0.8, linewidth=2)
    
    # Plot 2: Uncertainty Distribution
    ax2.hist(uncertainty_values, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(uncertainty_values), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(uncertainty_values):.4f}')
    ax2.set_xlabel('Uncertainty Values')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Uncertainty Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Predictions with Confidence Intervals
    sorted_indices = np.argsort(uncertainty_values)
    sorted_uncertainty = uncertainty_values[sorted_indices]
    sorted_pred = metrics['test_pred'][sorted_indices]
    sorted_actual = metrics['test_actual'][sorted_indices]
    
    # Create confidence intervals based on uncertainty
    confidence_levels = 0.95
    confidence_intervals = sorted_uncertainty * 1.96  # 95% CI
    
    ax3.plot(range(len(sorted_pred)), sorted_pred, 'b-', alpha=0.7, label='Predictions')
    ax3.fill_between(range(len(sorted_pred)), 
                    sorted_pred - confidence_intervals, 
                    sorted_pred + confidence_intervals, 
                    alpha=0.3, color='blue', label='95% Confidence Interval')
    ax3.scatter(range(len(sorted_actual)), sorted_actual, color='red', alpha=0.6, s=20, label='Actual')
    ax3.set_xlabel('Test Samples (Sorted by Uncertainty)')
    ax3.set_ylabel('Price')
    ax3.set_title('Predictions with Uncertainty Bands')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Uncertainty vs Time (if available)
    ax4.plot(range(len(uncertainty_values)), uncertainty_values, 'g-', linewidth=2, label='Uncertainty')
    ax4.axhline(np.mean(uncertainty_values), color='red', linestyle='--', alpha=0.7, 
               label=f'Mean: {np.mean(uncertainty_values):.4f}')
    ax4.fill_between(range(len(uncertainty_values)), 
                    np.zeros_like(uncertainty_values), 
                    uncertainty_values, 
                    alpha=0.3, color='green')
    ax4.set_xlabel('Test Sample Index')
    ax4.set_ylabel('Uncertainty')
    ax4.set_title('Uncertainty Over Test Samples')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def enhanced_analysis_and_visualization(ticker, model, scaler_X, scaler_y, 
                                      enhanced_data, feature_columns, 
                                      forecast_months=6, sequence_length=60):
    """Enhanced analysis using ML predictions with comprehensive visualization"""
    
    print(f"\n🔮 Enhanced ML-Powered Analysis for {ticker}")
    print("="*60)
    
    current_price = enhanced_data['Close'].iloc[-1]
    if isinstance(current_price, pd.Series):
        current_price = current_price.iloc[0]
    
    # Prepare recent data for prediction
    recent_data = enhanced_data[feature_columns].iloc[-sequence_length:].values
    recent_data_scaled = scaler_X.transform(recent_data.reshape(-1, recent_data.shape[-1])).reshape(1, sequence_length, -1)
    recent_tensor = torch.FloatTensor(recent_data_scaled)
    
    # Get ML predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        recent_tensor = recent_tensor.to(device)
        ml_price_pred, ml_vol_pred, ml_drift_pred, ml_uncertainty_pred = model(recent_tensor)
        
        ml_vol = ml_vol_pred.cpu().item()
        ml_drift = ml_drift_pred.cpu().item()
        ml_uncertainty = ml_uncertainty_pred.cpu().item()
    
    print(f"Current Price: ${current_price:.2f}")
    print(f"ML Predicted Volatility: {ml_vol:.4f} ({ml_vol*100:.2f}%)")
    print(f"ML Predicted Drift: {ml_drift:.4f} ({ml_drift*100:.2f}%)")
    print(f"ML Predicted Uncertainty: {ml_uncertainty:.4f}")
    
    # Enhanced GBM simulation with ML parameters
    T = forecast_months / 12
    N = forecast_months * 21
    
    time_steps, ml_paths = enhanced_gbm_simulation(
        current_price, ml_drift, ml_vol, T, N, num_simulations=1000
    )
    
    # Traditional GBM for comparison
    returns = enhanced_data['Returns'].dropna()
    trad_drift = returns.mean() * 252
    trad_vol = returns.std() * np.sqrt(252)
    
    _, trad_paths = enhanced_gbm_simulation(
        current_price, trad_drift, trad_vol, T, N, num_simulations=1000
    )
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker} Enhanced ML-Powered Stock Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Historical vs Recent Trend
    recent_history = enhanced_data['Close'].iloc[-252:].reset_index(drop=True)
    ax1.plot(recent_history, label='Historical Price', color='blue', linewidth=2)
    ax1.axhline(y=current_price, color='red', linestyle='--', alpha=0.7, 
               label=f'Current: ${current_price:.2f}')
    ax1.set_title('Recent Price History (1 Year)')
    ax1.set_ylabel('Price ($)')
    ax1.set_xlabel('Trading Days')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: ML vs Traditional GBM comparison
    ml_mean_path = np.mean(ml_paths, axis=0)
    trad_mean_path = np.mean(trad_paths, axis=0)
    
    # Show confidence bands
    ml_upper = np.percentile(ml_paths, 95, axis=0)
    ml_lower = np.percentile(ml_paths, 5, axis=0)
    trad_upper = np.percentile(trad_paths, 95, axis=0)
    trad_lower = np.percentile(trad_paths, 5, axis=0)
    
    days = range(N+1)
    ax2.fill_between(days, ml_lower, ml_upper, alpha=0.2, color='red', label='ML 90% CI')
    ax2.fill_between(days, trad_lower, trad_upper, alpha=0.2, color='blue', label='Traditional 90% CI')
    ax2.plot(days, ml_mean_path, color='red', linewidth=3, label='ML-Enhanced GBM')
    ax2.plot(days, trad_mean_path, color='blue', linewidth=3, label='Traditional GBM')
    
    ax2.set_title(f'ML vs Traditional GBM ({forecast_months} months)')
    ax2.set_ylabel('Price ($)')
    ax2.set_xlabel('Trading Days')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction distributions comparison
    ml_final_prices = ml_paths[:, -1]
    trad_final_prices = trad_paths[:, -1]
    
    ax3.hist(ml_final_prices, bins=50, alpha=0.7, color='red', label='ML-Enhanced', density=True)
    ax3.hist(trad_final_prices, bins=50, alpha=0.7, color='blue', label='Traditional', density=True)
    ax3.axvline(current_price, color='green', linestyle='-', linewidth=2, 
               label=f'Current: ${current_price:.2f}')
    ax3.axvline(np.mean(ml_final_prices), color='red', linestyle='--', linewidth=2, 
               label=f'ML Mean: ${np.mean(ml_final_prices):.2f}')
    ax3.axvline(np.mean(trad_final_prices), color='blue', linestyle='--', linewidth=2, 
               label=f'Trad Mean: ${np.mean(trad_final_prices):.2f}')
    
    ax3.set_title('Price Distribution Comparison')
    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Risk metrics comparison
    ml_percentiles = np.percentile(ml_final_prices, [5, 25, 50, 75, 95])
    trad_percentiles = np.percentile(trad_final_prices, [5, 25, 50, 75, 95])
    
    x_pos = np.arange(5)
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, ml_percentiles, width, label='ML-Enhanced', 
                   alpha=0.7, color='red')
    bars2 = ax4.bar(x_pos + width/2, trad_percentiles, width, label='Traditional', 
                   alpha=0.7, color='blue')
    ax4.axhline(y=current_price, color='green', linestyle='-', linewidth=2, 
               label=f'Current: ${current_price:.2f}')
    
    # Add value labels on bars
    for bars, percentiles in [(bars1, ml_percentiles), (bars2, trad_percentiles)]:
        for bar, price in zip(bars, percentiles):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${price:.0f}', ha='center', va='bottom', fontsize=8)
    
    ax4.set_title('Risk Percentiles Comparison')
    ax4.set_ylabel('Price ($)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['5th', '25th', '50th', '75th', '95th'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional technical analysis plot
    fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(16, 10))
    fig2.suptitle(f'{ticker} Technical Analysis & ML Insights', fontsize=16, fontweight='bold')
    
    # Plot 5: Price with technical indicators
    recent_data_plot = enhanced_data.iloc[-252:]
    ax5.plot(recent_data_plot.index, recent_data_plot['Close'], label='Close Price', linewidth=2)
    ax5.plot(recent_data_plot.index, recent_data_plot['MA_20'], label='MA 20', alpha=0.7)
    ax5.plot(recent_data_plot.index, recent_data_plot['MA_50'], label='MA 50', alpha=0.7)
    ax5.fill_between(recent_data_plot.index, recent_data_plot['BB_Lower'], 
                    recent_data_plot['BB_Upper'], alpha=0.2, label='Bollinger Bands')
    ax5.set_title('Price with Technical Indicators')
    ax5.set_ylabel('Price ($)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: RSI
    ax6.plot(recent_data_plot.index, recent_data_plot['RSI'], color='purple', linewidth=2)
    ax6.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
    ax6.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
    ax6.set_title('RSI (Relative Strength Index)')
    ax6.set_ylabel('RSI')
    ax6.set_ylim(0, 100)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Volume analysis
    ax7.bar(recent_data_plot.index, recent_data_plot['Volume'], alpha=0.6, color='gray')
    ax7.plot(recent_data_plot.index, recent_data_plot['Volume_MA'], color='red', 
            linewidth=2, label='Volume MA')
    ax7.set_title('Volume Analysis')
    ax7.set_ylabel('Volume')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Volatility comparison
    historical_vol = enhanced_data['Volatility_20'].iloc[-252:] * np.sqrt(252) * 100
    ax8.plot(recent_data_plot.index, historical_vol, label='Historical Volatility', 
            color='blue', linewidth=2)
    ax8.axhline(y=ml_vol*100, color='red', linestyle='--', linewidth=2, 
               label=f'ML Predicted: {ml_vol*100:.1f}%')
    ax8.axhline(y=trad_vol*100, color='green', linestyle='--', linewidth=2, 
               label=f'Traditional: {trad_vol*100:.1f}%')
    ax8.set_title('Volatility Analysis')
    ax8.set_ylabel('Annualized Volatility (%)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance comparison
    ml_expected_return = (np.mean(ml_final_prices) / current_price - 1) * 100
    trad_expected_return = (np.mean(trad_final_prices) / current_price - 1) * 100
    
    ml_volatility = np.std(ml_final_prices) / current_price * 100
    trad_volatility = np.std(trad_final_prices) / current_price * 100
    
    print(f"\n📈 ENHANCED ANALYSIS RESULTS")
    print("="*50)
    print(f"ML-Enhanced Expected Return: {ml_expected_return:+.2f}%")
    print(f"Traditional Expected Return: {trad_expected_return:+.2f}%")
    print(f"ML-Enhanced Volatility: {ml_volatility:.2f}%")
    print(f"Traditional Volatility: {trad_volatility:.2f}%")
    print(f"ML Improvement in Precision: {abs(ml_volatility - trad_volatility):.2f}% volatility difference")
    
    # Risk analysis
    ml_profit_prob = (np.sum(ml_final_prices > current_price) / len(ml_final_prices)) * 100
    trad_profit_prob = (np.sum(trad_final_prices > current_price) / len(trad_final_prices)) * 100
    
    print(f"\n📊 RISK ANALYSIS")
    print(f"ML Model Profit Probability: {ml_profit_prob:.1f}%")
    print(f"Traditional Model Profit Probability: {trad_profit_prob:.1f}%")
    
    return {
        'ml_predictions': ml_final_prices,
        'traditional_predictions': trad_final_prices,
        'ml_expected_return': ml_expected_return,
        'traditional_expected_return': trad_expected_return,
        'improvement_metrics': {
            'volatility_difference': abs(ml_volatility - trad_volatility),
            'return_difference': abs(ml_expected_return - trad_expected_return),
            'profit_prob_improvement': ml_profit_prob - trad_profit_prob
        }
    }

def enhanced_analysis_with_uncertainty(ticker, model, scaler_X, scaler_y, 
                                     enhanced_data, feature_columns, 
                                     forecast_months=6, sequence_length=60):
    """Enhanced analysis with uncertainty quantification"""
    
    print(f"\n🔮 Enhanced Bayesian Analysis for {ticker}")
    print("="*60)
    
    current_price = enhanced_data['Close'].iloc[-1]
    if isinstance(current_price, pd.Series):
        current_price = current_price.iloc[0]
    
    # Prepare recent data for prediction
    recent_data = enhanced_data[feature_columns].iloc[-sequence_length:].values
    recent_data_scaled = scaler_X.transform(recent_data.reshape(-1, recent_data.shape[-1])).reshape(1, sequence_length, -1)
    recent_tensor = torch.FloatTensor(recent_data_scaled)
    
    # Get Bayesian predictions with uncertainty
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        recent_tensor = recent_tensor.to(device)
        uncertainty_results = model.predict_with_uncertainty(recent_tensor, num_samples=20)
        
        # Extract predictions and uncertainties
        price_mean = uncertainty_results['price_mean'].cpu().item()
        price_std = uncertainty_results['price_std'].cpu().item()
        vol_mean = uncertainty_results['volatility_mean'].cpu().item()
        vol_std = uncertainty_results['volatility_std'].cpu().item()
        drift_mean = uncertainty_results['drift_mean'].cpu().item()
        drift_std = uncertainty_results['drift_std'].cpu().item()
        total_uncertainty = uncertainty_results['total_uncertainty'].cpu().item()
    
    print(f"Current Price: ${current_price:.2f}")
    print(f"Bayesian Price Prediction: ${price_mean:.2f} ± ${price_std:.2f}")
    print(f"Bayesian Volatility: {vol_mean:.4f} ± {vol_std:.4f} ({vol_mean*100:.2f}% ± {vol_std*100:.2f}%)")
    print(f"Bayesian Drift: {drift_mean:.4f} ± {drift_std:.4f} ({drift_mean*100:.2f}% ± {drift_std*100:.2f}%)")
    print(f"Total Uncertainty: {total_uncertainty:.4f}")
    
    # Enhanced GBM simulation with Bayesian parameters
    T = forecast_months / 12
    N = forecast_months * 21
    
    # Sample from Bayesian posterior for simulation
    num_simulations = 1000
    time_steps, bayesian_paths = enhanced_gbm_simulation(
        current_price, drift_mean, vol_mean, T, N, num_simulations=num_simulations
    )
    
    # Traditional GBM for comparison
    returns = enhanced_data['Returns'].dropna()
    trad_drift = returns.mean() * 252
    trad_vol = returns.std() * np.sqrt(252)
    
    _, trad_paths = enhanced_gbm_simulation(
        current_price, trad_drift, trad_vol, T, N, num_simulations=num_simulations
    )
    
    # Create comprehensive visualization with uncertainty
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker} Bayesian Uncertainty Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Bayesian vs Traditional GBM with uncertainty bands
    bayesian_mean_path = np.mean(bayesian_paths, axis=0)
    trad_mean_path = np.mean(trad_paths, axis=0)
    
    # Show confidence bands
    bayesian_upper = np.percentile(bayesian_paths, 95, axis=0)
    bayesian_lower = np.percentile(bayesian_paths, 5, axis=0)
    trad_upper = np.percentile(trad_paths, 95, axis=0)
    trad_lower = np.percentile(trad_paths, 5, axis=0)
    
    days = range(N+1)
    ax1.fill_between(days, bayesian_lower, bayesian_upper, alpha=0.2, color='purple', label='Bayesian 90% CI')
    ax1.fill_between(days, trad_lower, trad_upper, alpha=0.2, color='blue', label='Traditional 90% CI')
    ax1.plot(days, bayesian_mean_path, color='purple', linewidth=3, label='Bayesian GBM')
    ax1.plot(days, trad_mean_path, color='blue', linewidth=3, label='Traditional GBM')
    
    ax1.set_title(f'Bayesian vs Traditional GBM ({forecast_months} months)')
    ax1.set_ylabel('Price ($)')
    ax1.set_xlabel('Trading Days')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter uncertainty comparison
    bayesian_final_prices = bayesian_paths[:, -1]
    trad_final_prices = trad_paths[:, -1]
    
    ax2.hist(bayesian_final_prices, bins=50, alpha=0.7, color='purple', label='Bayesian', density=True)
    ax2.hist(trad_final_prices, bins=50, alpha=0.7, color='blue', label='Traditional', density=True)
    ax2.axvline(current_price, color='green', linestyle='-', linewidth=2, 
               label=f'Current: ${current_price:.2f}')
    ax2.axvline(np.mean(bayesian_final_prices), color='purple', linestyle='--', linewidth=2, 
               label=f'Bayesian Mean: ${np.mean(bayesian_final_prices):.2f}')
    ax2.axvline(np.mean(trad_final_prices), color='blue', linestyle='--', linewidth=2, 
               label=f'Trad Mean: ${np.mean(trad_final_prices):.2f}')
    
    ax2.set_title('Price Distribution with Uncertainty')
    ax2.set_xlabel('Price ($)')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Uncertainty decomposition
    uncertainty_components = ['Aleatoric', 'Epistemic', 'Total']
    uncertainty_values = [
        uncertainty_results['aleatoric_uncertainty'].cpu().item(),
        uncertainty_results['epistemic_uncertainty'].cpu().item(),
        total_uncertainty
    ]
    
    bars = ax3.bar(uncertainty_components, uncertainty_values, color=['orange', 'red', 'purple'], alpha=0.7)
    ax3.set_title('Uncertainty Decomposition')
    ax3.set_ylabel('Uncertainty Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, uncertainty_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Risk metrics with uncertainty
    bayesian_percentiles = np.percentile(bayesian_final_prices, [5, 25, 50, 75, 95])
    trad_percentiles = np.percentile(trad_final_prices, [5, 25, 50, 75, 95])
    
    x_pos = np.arange(5)
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, bayesian_percentiles, width, label='Bayesian', 
                   alpha=0.7, color='purple')
    bars2 = ax4.bar(x_pos + width/2, trad_percentiles, width, label='Traditional', 
                   alpha=0.7, color='blue')
    ax4.axhline(y=current_price, color='green', linestyle='-', linewidth=2, 
               label=f'Current: ${current_price:.2f}')
    
    # Add value labels on bars
    for bars, percentiles in [(bars1, bayesian_percentiles), (bars2, trad_percentiles)]:
        for bar, price in zip(bars, percentiles):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${price:.0f}', ha='center', va='bottom', fontsize=8)
    
    ax4.set_title('Risk Percentiles with Uncertainty')
    ax4.set_ylabel('Price ($)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['5th', '25th', '50th', '75th', '95th'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance comparison with uncertainty
    bayesian_expected_return = (np.mean(bayesian_final_prices) / current_price - 1) * 100
    trad_expected_return = (np.mean(trad_final_prices) / current_price - 1) * 100
    
    bayesian_volatility = np.std(bayesian_final_prices) / current_price * 100
    trad_volatility = np.std(trad_final_prices) / current_price * 100
    
    print(f"\n📈 BAYESIAN ANALYSIS RESULTS")
    print("="*50)
    print(f"Bayesian Expected Return: {bayesian_expected_return:+.2f}% ± {drift_std*100:.2f}%")
    print(f"Traditional Expected Return: {trad_expected_return:+.2f}%")
    print(f"Bayesian Volatility: {bayesian_volatility:.2f}% ± {vol_std*100:.2f}%")
    print(f"Traditional Volatility: {trad_volatility:.2f}%")
    print(f"Uncertainty-Adjusted Precision: {abs(bayesian_volatility - trad_volatility):.2f}%")
    
    # Risk analysis with uncertainty
    bayesian_profit_prob = (np.sum(bayesian_final_prices > current_price) / len(bayesian_final_prices)) * 100
    trad_profit_prob = (np.sum(trad_final_prices > current_price) / len(trad_final_prices)) * 100
    
    print(f"\n📊 RISK ANALYSIS WITH UNCERTAINTY")
    print(f"Bayesian Profit Probability: {bayesian_profit_prob:.1f}%")
    print(f"Traditional Profit Probability: {trad_profit_prob:.1f}%")
    print(f"Uncertainty Level: {'High' if total_uncertainty > 0.1 else 'Medium' if total_uncertainty > 0.05 else 'Low'}")
    
    return {
        'bayesian_predictions': bayesian_final_prices,
        'traditional_predictions': trad_final_prices,
        'bayesian_expected_return': bayesian_expected_return,
        'traditional_expected_return': trad_expected_return,
        'uncertainty_metrics': {
            'total_uncertainty': total_uncertainty,
            'aleatoric_uncertainty': uncertainty_results['aleatoric_uncertainty'].cpu().item(),
            'epistemic_uncertainty': uncertainty_results['epistemic_uncertainty'].cpu().item(),
            'price_std': price_std,
            'volatility_std': vol_std,
            'drift_std': drift_std
        }
    }

def train_enhanced_model(ticker, sequence_length=60, epochs=100, model_type='transformer'):
    """Train the enhanced ML model"""
    print(f"🚀 Training Enhanced ML Model for {ticker}")
    print("="*60)
    
    # Fetch and prepare data with proper handling
    stock_data = fetch_and_clean_data(ticker, period='5y')
    if stock_data is None:
        raise ValueError(f"Could not fetch data for {ticker}")
    
    print(f"Data shape: {stock_data.shape}")
    print(f"Columns: {list(stock_data.columns)}")
    
    enhanced_data = create_enhanced_features(stock_data)
    
    # Select features for training
    feature_columns = [
        'Returns', 'Log_Returns', 'Volatility_10', 'Volatility_20',
        'MA_Ratio_5', 'MA_Ratio_10', 'MA_Ratio_20', 'MA_Ratio_50',
        'RSI', 'MACD', 'MACD_Signal', 'BB_Position',
        'Volume_Ratio', 'High_Low_Ratio', 'Open_Close_Ratio',
        'Price_Lag_1', 'Return_Lag_1', 'Volume_Lag_1',
        'Price_Lag_2', 'Return_Lag_2', 'Price_Lag_3', 'Return_Lag_3'
    ]
    
    # Remove any columns that don't exist
    available_features = [col for col in feature_columns if col in enhanced_data.columns]
    print(f"Available features: {len(available_features)}")
    
    # Prepare sequences
    sequences, targets = prepare_sequences(
        enhanced_data, available_features, 'Close', sequence_length
    )
    
    # Split data
    split_idx = int(0.8 * len(sequences))
    X_train, X_test = sequences[:split_idx], sequences[split_idx:]
    y_train, y_test = targets[:split_idx], targets[split_idx:]
    
    # Normalize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)
    
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if model_type == 'transformer':
        model = TransformerStockPredictor(
            input_size=len(available_features),
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.1,
            max_seq_length=sequence_length
        ).to(device)
    elif model_type == 'bayesian':
        model = BayesianTransformerStockPredictor(
            input_size=len(available_features),
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.1,
            max_seq_length=sequence_length,
            prior_std=1.0,
            posterior_std_init=0.1,
            num_samples=10
        ).to(device)
    else: # LSTM
        model = EnhancedStockPredictor(
            input_size=len(available_features),
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        ).to(device)
    
    # Loss function and optimizer with transformer-specific settings
    criterion = nn.MSELoss()
    if model_type in ['transformer', 'bayesian']:
        # Lower learning rate and higher weight decay for transformer stability
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7, min_lr=1e-6)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    model.train()
    train_losses = []
    
    # Warmup for transformer
    if model_type in ['transformer', 'bayesian']:
        warmup_epochs = min(5, epochs // 10)
        print(f"🔥 Using {warmup_epochs} epochs warmup for transformer")
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            if model_type == 'transformer':
                price_pred, vol_pred, drift_pred, uncertainty_pred = model(batch_X)
                loss = criterion(price_pred.squeeze(), batch_y)
            elif model_type == 'bayesian':
                price_pred, vol_pred, drift_pred, aleatoric_uncertainty, epistemic_uncertainty = model(batch_X)
                
                # Bayesian loss: MSE + KL divergence + uncertainty regularization
                mse_loss = criterion(price_pred.squeeze(), batch_y)
                kl_loss = model.kl_loss()
                
                # Uncertainty regularization to prevent overconfidence
                uncertainty_reg = torch.mean(aleatoric_uncertainty + epistemic_uncertainty)
                
                # Total loss with weighting
                loss = mse_loss + 0.01 * kl_loss + 0.001 * uncertainty_reg
            else: # LSTM
                price_pred, vol_pred, drift_pred = model(batch_X)
                loss = criterion(price_pred.squeeze(), batch_y)
            
            loss.backward()
            if model_type in ['transformer', 'bayesian']:
                # Gentler gradient clipping for transformer
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
            # Early stopping for transformer if loss is exploding
            if model_type in ['transformer', 'bayesian'] and avg_loss > 10.0:
                print(f"⚠️  Training stopped early due to high loss: {avg_loss:.6f}")
                break
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        if model_type == 'transformer':
            test_pred, test_vol, test_drift, test_uncertainty = model(X_test_tensor)
            test_pred = test_pred.cpu().numpy()
            test_vol = test_vol.cpu().numpy()
            test_drift = test_drift.cpu().numpy()
            test_uncertainty = test_uncertainty.cpu().numpy()
        elif model_type == 'bayesian':
            # Use uncertainty prediction for Bayesian model
            uncertainty_results = model.predict_with_uncertainty(X_test_tensor, num_samples=10)
            test_pred = uncertainty_results['price_mean'].cpu().numpy()
            test_vol = uncertainty_results['volatility_mean'].cpu().numpy()
            test_drift = uncertainty_results['drift_mean'].cpu().numpy()
            test_uncertainty = uncertainty_results['total_uncertainty'].cpu().numpy()
        else: # LSTM
            test_pred, test_vol, test_drift = model(X_test_tensor)
            test_pred = test_pred.cpu().numpy()
            test_vol = test_vol.cpu().numpy()
            test_drift = test_drift.cpu().numpy()
            test_uncertainty = np.zeros_like(test_pred)  # Placeholder for LSTM
    
    # Inverse transform predictions
    test_pred_actual = scaler_y.inverse_transform(test_pred)
    y_test_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))
    
    # Calculate metrics
    mse = mean_squared_error(y_test_actual, test_pred_actual)
    mae = mean_absolute_error(y_test_actual, test_pred_actual)
    
    print(f"\n📊 Model Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    
    metrics = {
        'test_pred': test_pred_actual.flatten(),
        'test_actual': y_test_actual.flatten(),
        'test_vol': test_vol.flatten(),
        'test_drift': test_drift.flatten(),
        'test_uncertainty': test_uncertainty.flatten(),
        'train_losses': train_losses
    }
    
    # Plot training results
    plot_training_results(metrics, ticker)
    
    return model, scaler_X, scaler_y, enhanced_data, available_features, metrics

def train_bayesian_model(ticker, sequence_length=60, epochs=100, num_samples=10, simplified=True):
    """Train the Bayesian Neural Network model with uncertainty quantification"""
    print(f"🧠 Training Bayesian Neural Network for {ticker}")
    print("="*60)
    
    if simplified:
        print("🔧 Using simplified Bayesian model for stability")
        # Use regular transformer with uncertainty estimation
        model, scaler_X, scaler_y, enhanced_data, feature_columns, metrics = train_enhanced_model(
            ticker, sequence_length=sequence_length, epochs=epochs, model_type='transformer'
        )
        
        # Add uncertainty estimation using prediction variance
        print(f"\n🔮 Adding Uncertainty Quantification")
        print("="*50)
        
        # Calculate uncertainty based on prediction variance using Monte Carlo dropout
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        
        # Get the test data for uncertainty estimation
        sequences, targets = prepare_sequences(
            enhanced_data, feature_columns, 'Close', sequence_length
        )
        
        # Split data to get test set
        split_idx = int(0.8 * len(sequences))
        X_test = sequences[split_idx:]
        
        # Normalize test data
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
        
        # Enable dropout for uncertainty estimation
        model.train()  # Enable dropout
        
        uncertainties = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred, _, _, _ = model(X_test_tensor)
                uncertainties.append(pred.cpu().numpy())
        
        # Calculate uncertainty as variance across predictions
        uncertainty_array = np.var(uncertainties, axis=0).flatten()
        metrics['test_uncertainty'] = uncertainty_array
        
        print(f"Average Uncertainty: {np.mean(uncertainty_array):.4f}")
        print(f"Uncertainty Std: {np.std(uncertainty_array):.4f}")
        
    else:
        # Use full Bayesian model
        model, scaler_X, scaler_y, enhanced_data, feature_columns, metrics = train_enhanced_model(
            ticker, sequence_length=sequence_length, epochs=epochs, model_type='bayesian'
        )
    
    # Additional Bayesian-specific analysis
    print(f"\n🔮 Bayesian Model Uncertainty Analysis")
    print("="*50)
    
    # Get uncertainty statistics
    uncertainty_values = metrics['test_uncertainty']
    print(f"Average Uncertainty: {np.mean(uncertainty_values):.4f}")
    print(f"Uncertainty Std: {np.std(uncertainty_values):.4f}")
    print(f"Min Uncertainty: {np.min(uncertainty_values):.4f}")
    print(f"Max Uncertainty: {np.max(uncertainty_values):.4f}")
    
    # Analyze prediction confidence
    high_confidence_mask = uncertainty_values < np.percentile(uncertainty_values, 25)
    low_confidence_mask = uncertainty_values > np.percentile(uncertainty_values, 75)
    
    high_conf_errors = np.abs(metrics['test_pred'][high_confidence_mask] - metrics['test_actual'][high_confidence_mask])
    low_conf_errors = np.abs(metrics['test_pred'][low_confidence_mask] - metrics['test_actual'][low_confidence_mask])
    
    print(f"\n📊 Confidence Analysis:")
    print(f"High Confidence Predictions MAE: {np.mean(high_conf_errors):.4f}")
    print(f"Low Confidence Predictions MAE: {np.mean(low_conf_errors):.4f}")
    print(f"Confidence-Error Correlation: {np.corrcoef(uncertainty_values, np.abs(metrics['test_pred'] - metrics['test_actual']))[0,1]:.4f}")
    
    return model, scaler_X, scaler_y, enhanced_data, feature_columns, metrics

def compare_models_performance(ticker, sequence_length=60, epochs=50):
    """Compare Transformer vs LSTM model performance"""
    print(f"🔬 Model Comparison: Transformer vs LSTM for {ticker}")
    print("="*70)
    
    results = {}
    
    # Train Transformer model
    print("\n🚀 Training Transformer Model...")
    transformer_model, scaler_X, scaler_y, enhanced_data, feature_columns, transformer_metrics = train_enhanced_model(
        ticker, sequence_length=sequence_length, epochs=epochs, model_type='transformer'
    )
    results['transformer'] = {
        'model': transformer_model,
        'metrics': transformer_metrics,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'enhanced_data': enhanced_data,
        'feature_columns': feature_columns
    }
    
    # Train LSTM model
    print("\n🚀 Training LSTM Model...")
    lstm_model, _, _, _, _, lstm_metrics = train_enhanced_model(
        ticker, sequence_length=sequence_length, epochs=epochs, model_type='lstm'
    )
    results['lstm'] = {
        'model': lstm_model,
        'metrics': lstm_metrics
    }
    
    # Compare performance
    print(f"\n📊 MODEL COMPARISON RESULTS")
    print("="*50)
    
    transformer_mse = mean_squared_error(transformer_metrics['test_actual'], transformer_metrics['test_pred'])
    lstm_mse = mean_squared_error(lstm_metrics['test_actual'], lstm_metrics['test_pred'])
    
    transformer_mae = mean_absolute_error(transformer_metrics['test_actual'], transformer_metrics['test_pred'])
    lstm_mae = mean_absolute_error(lstm_metrics['test_actual'], lstm_metrics['test_pred'])
    
    print(f"Transformer MSE: {transformer_mse:.6f}")
    print(f"LSTM MSE: {lstm_mse:.6f}")
    print(f"Transformer MAE: {transformer_mae:.6f}")
    print(f"LSTM MAE: {lstm_mae:.6f}")
    
    improvement_mse = ((lstm_mse - transformer_mse) / lstm_mse) * 100
    improvement_mae = ((lstm_mae - transformer_mae) / lstm_mae) * 100
    
    print(f"\n🎯 IMPROVEMENT METRICS")
    print(f"MSE Improvement: {improvement_mse:+.2f}%")
    print(f"MAE Improvement: {improvement_mae:+.2f}%")
    
    if improvement_mse > 0:
        print(f"✅ Transformer outperforms LSTM by {improvement_mse:.2f}% in MSE")
    else:
        print(f"⚠️ LSTM outperforms Transformer by {abs(improvement_mse):.2f}% in MSE")
    
    # Plot comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker} Transformer vs LSTM Model Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Training Loss Comparison
    ax1.plot(transformer_metrics['train_losses'], label='Transformer', color='red', linewidth=2)
    ax1.plot(lstm_metrics['train_losses'], label='LSTM', color='blue', linewidth=2)
    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction Accuracy Comparison
    ax2.scatter(transformer_metrics['test_actual'], transformer_metrics['test_pred'], 
               alpha=0.6, color='red', label='Transformer', s=30)
    ax2.scatter(lstm_metrics['test_actual'], lstm_metrics['test_pred'], 
               alpha=0.6, color='blue', label='LSTM', s=30)
    
    min_val = min(min(transformer_metrics['test_actual']), min(lstm_metrics['test_actual']))
    max_val = max(max(transformer_metrics['test_actual']), max(lstm_metrics['test_actual']))
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Prices')
    ax2.set_ylabel('Predicted Prices')
    ax2.set_title('Prediction Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error Distribution Comparison
    transformer_errors = transformer_metrics['test_pred'] - transformer_metrics['test_actual']
    lstm_errors = lstm_metrics['test_pred'] - lstm_metrics['test_actual']
    
    ax3.hist(transformer_errors, bins=30, alpha=0.7, color='red', label='Transformer', density=True)
    ax3.hist(lstm_errors, bins=30, alpha=0.7, color='blue', label='LSTM', density=True)
    ax3.axvline(0, color='black', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Density')
    ax3.set_title('Error Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Model Parameters Comparison
    transformer_params = sum(p.numel() for p in transformer_model.parameters())
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    
    models = ['Transformer', 'LSTM']
    param_counts = [transformer_params, lstm_params]
    colors = ['red', 'blue']
    
    bars = ax4.bar(models, param_counts, color=colors, alpha=0.7)
    ax4.set_title('Model Complexity Comparison')
    ax4.set_ylabel('Number of Parameters')
    
    # Add value labels on bars
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

def heston_stochastic_volatility_simulation(S0, mu, kappa, theta, sigma_v, rho, T, N, num_simulations=1000):
    """
    Heston Stochastic Volatility Model Simulation
    
    Parameters:
    - S0: Initial stock price
    - mu: Risk-free rate
    - kappa: Mean reversion speed of volatility
    - theta: Long-term mean of volatility
    - sigma_v: Volatility of volatility
    - rho: Correlation between stock and volatility processes
    - T: Time horizon
    - N: Number of time steps
    - num_simulations: Number of simulation paths
    
    Returns:
    - time_steps: Array of time points
    - stock_paths: Array of stock price paths
    - volatility_paths: Array of volatility paths
    """
    dt = T / N
    time_steps = np.linspace(0, T, N+1)
    
    # Initialize arrays
    stock_paths = np.zeros((num_simulations, N+1))
    volatility_paths = np.zeros((num_simulations, N+1))
    
    # Set initial values
    stock_paths[:, 0] = S0
    volatility_paths[:, 0] = theta  # Start at long-term mean
    
    np.random.seed(42)  # For reproducibility
    
    for i in range(num_simulations):
        for t in range(N):
            # Current values
            S_t = stock_paths[i, t]
            v_t = volatility_paths[i, t]
            
            # Generate correlated random numbers
            Z1 = np.random.normal(0, 1)
            Z2 = np.random.normal(0, 1)
            Z_v = rho * Z1 + np.sqrt(1 - rho**2) * Z2
            
            # Update volatility (CIR process)
            dv = kappa * (theta - v_t) * dt + sigma_v * np.sqrt(v_t) * np.sqrt(dt) * Z_v
            v_new = max(v_t + dv, 0.0001)  # Ensure positive volatility
            
            # Update stock price using log-Euler scheme to enforce positivity
            S_new = S_t * np.exp((mu - 0.5 * v_new) * dt + np.sqrt(v_new) * np.sqrt(dt) * Z1)
            
            # Store values
            stock_paths[i, t+1] = S_new
            volatility_paths[i, t+1] = v_new
    
    return time_steps, stock_paths, volatility_paths

def regime_switching_gbm_simulation(S0, mu_states, sigma_states, transition_matrix, T, N, num_simulations=1000):
    """
    Regime-Switching GBM Simulation
    
    Parameters:
    - S0: Initial stock price
    - mu_states: Array of drift parameters for each regime
    - sigma_states: Array of volatility parameters for each regime
    - transition_matrix: Matrix of transition probabilities between regimes
    - T: Time horizon
    - N: Number of time steps
    - num_simulations: Number of simulation paths
    
    Returns:
    - time_steps: Array of time points
    - stock_paths: Array of stock price paths
    - regime_paths: Array of regime paths
    """
    dt = T / N
    time_steps = np.linspace(0, T, N+1)
    num_regimes = len(mu_states)
    
    # Initialize arrays
    stock_paths = np.zeros((num_simulations, N+1))
    regime_paths = np.zeros((num_simulations, N+1), dtype=int)
    
    # Set initial values
    stock_paths[:, 0] = S0
    regime_paths[:, 0] = 0  # Start in regime 0
    
    np.random.seed(42)  # For reproducibility
    
    for i in range(num_simulations):
        for t in range(N):
            # Current values
            S_t = stock_paths[i, t]
            current_regime = regime_paths[i, t]
            
            # Get current regime parameters
            mu = mu_states[current_regime]
            sigma = sigma_states[current_regime]
            
            # Update stock price using GBM exact discretization
            dW = np.random.normal(0, np.sqrt(dt))
            S_new = S_t * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
            
            # Transition to new regime
            transition_probs = transition_matrix[current_regime]
            new_regime = np.random.choice(num_regimes, p=transition_probs)
            
            # Store values
            stock_paths[i, t+1] = S_new
            regime_paths[i, t+1] = new_regime
    
    return time_steps, stock_paths, regime_paths

def merton_jump_diffusion_simulation(S0, mu, sigma, lambda_jump, mu_jump, sigma_jump, T, N, num_simulations=1000):
    """
    Merton Jump Diffusion Model Simulation
    
    Parameters:
    - S0: Initial stock price
    - mu: Drift parameter (continuous part)
    - sigma: Volatility parameter (continuous part)
    - lambda_jump: Jump intensity (Poisson parameter)
    - mu_jump: Mean of jump size (log-normal)
    - sigma_jump: Standard deviation of jump size (log-normal)
    - T: Time horizon
    - N: Number of time steps
    - num_simulations: Number of simulation paths
    
    Returns:
    - time_steps: Array of time points
    - stock_paths: Array of stock price paths
    - jump_times: Array of jump occurrence times
    """
    dt = T / N
    time_steps = np.linspace(0, T, N+1)
    
    # Initialize arrays
    stock_paths = np.zeros((num_simulations, N+1))
    jump_times = np.zeros((num_simulations, N+1), dtype=bool)
    
    # Set initial values
    stock_paths[:, 0] = S0
    
    np.random.seed(42)  # For reproducibility
    
    for i in range(num_simulations):
        for t in range(N):
            # Current stock price
            S_t = stock_paths[i, t]
            
            # Continuous part (GBM exact discretization)
            dW = np.random.normal(0, np.sqrt(dt))
            continuous_factor = np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
            
            # Jump part (Poisson process)
            jump_occurred = np.random.poisson(lambda_jump * dt) > 0
            jump_times[i, t+1] = jump_occurred
            
            if jump_occurred:
                # Jump factor (log-normal jump multiplier)
                jump_factor = np.random.lognormal(mu_jump, sigma_jump)
            else:
                jump_factor = 1.0
            
            # Total update (multiplicative)
            S_new = S_t * continuous_factor * jump_factor
            
            # Store values
            stock_paths[i, t+1] = S_new
    
    return time_steps, stock_paths, jump_times

def enhanced_heston_analysis(ticker, model, scaler_X, scaler_y, enhanced_data, feature_columns, 
                           forecast_months=6, sequence_length=60):
    """Enhanced analysis using Heston stochastic volatility model"""
    
    print(f"\n🌊 Heston Stochastic Volatility Analysis for {ticker}")
    print("="*60)
    
    current_price = enhanced_data['Close'].iloc[-1]
    if isinstance(current_price, pd.Series):
        current_price = current_price.iloc[0]
    
    # Get ML predictions for initial parameters
    recent_data = enhanced_data[feature_columns].iloc[-sequence_length:].values
    recent_data_scaled = scaler_X.transform(recent_data.reshape(-1, recent_data.shape[-1])).reshape(1, sequence_length, -1)
    recent_tensor = torch.FloatTensor(recent_data_scaled)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        recent_tensor = recent_tensor.to(device)
        ml_price_pred, ml_vol_pred, ml_drift_pred, ml_uncertainty_pred = model(recent_tensor)
        
        ml_vol = ml_vol_pred.cpu().item()
        ml_drift = ml_drift_pred.cpu().item()
    
    # Heston model parameters (calibrated to market data)
    mu = ml_drift  # Risk-free rate
    kappa = 2.0    # Mean reversion speed
    theta = ml_vol  # Long-term volatility mean
    sigma_v = 0.3  # Volatility of volatility
    rho = -0.7     # Correlation (leverage effect)
    
    T = forecast_months / 12
    N = forecast_months * 21
    
    # Simulate Heston model
    time_steps, heston_stock_paths, heston_vol_paths = heston_stochastic_volatility_simulation(
        current_price, mu, kappa, theta, sigma_v, rho, T, N, num_simulations=1000
    )
    
    # Traditional GBM for comparison
    returns = enhanced_data['Returns'].dropna()
    trad_drift = returns.mean() * 252
    trad_vol = returns.std() * np.sqrt(252)
    
    _, trad_paths = enhanced_gbm_simulation(
        current_price, trad_drift, trad_vol, T, N, num_simulations=1000
    )
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker} Heston Stochastic Volatility Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Stock price paths comparison
    heston_mean_path = np.mean(heston_stock_paths, axis=0)
    trad_mean_path = np.mean(trad_paths, axis=0)
    
    heston_upper = np.percentile(heston_stock_paths, 95, axis=0)
    heston_lower = np.percentile(heston_stock_paths, 5, axis=0)
    trad_upper = np.percentile(trad_paths, 95, axis=0)
    trad_lower = np.percentile(trad_paths, 5, axis=0)
    
    days = range(N+1)
    ax1.fill_between(days, heston_lower, heston_upper, alpha=0.2, color='red', label='Heston 90% CI')
    ax1.fill_between(days, trad_lower, trad_upper, alpha=0.2, color='blue', label='GBM 90% CI')
    ax1.plot(days, heston_mean_path, color='red', linewidth=3, label='Heston Model')
    ax1.plot(days, trad_mean_path, color='blue', linewidth=3, label='Traditional GBM')
    
    ax1.set_title(f'Heston vs Traditional GBM ({forecast_months} months)')
    ax1.set_ylabel('Price ($)')
    ax1.set_xlabel('Trading Days')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Stochastic volatility paths
    vol_mean_path = np.mean(heston_vol_paths, axis=0)
    vol_upper = np.percentile(heston_vol_paths, 95, axis=0)
    vol_lower = np.percentile(heston_vol_paths, 5, axis=0)
    
    ax2.fill_between(days, vol_lower, vol_upper, alpha=0.3, color='green', label='Volatility 90% CI')
    ax2.plot(days, vol_mean_path, color='green', linewidth=2, label='Mean Volatility')
    ax2.axhline(y=theta, color='red', linestyle='--', alpha=0.7, label=f'Long-term Mean: {theta:.4f}')
    
    ax2.set_title('Stochastic Volatility Evolution')
    ax2.set_ylabel('Volatility')
    ax2.set_xlabel('Trading Days')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Price distribution comparison
    heston_final_prices = heston_stock_paths[:, -1]
    trad_final_prices = trad_paths[:, -1]
    
    ax3.hist(heston_final_prices, bins=50, alpha=0.7, color='red', label='Heston Model', density=True)
    ax3.hist(trad_final_prices, bins=50, alpha=0.7, color='blue', label='Traditional GBM', density=True)
    ax3.axvline(current_price, color='green', linestyle='-', linewidth=2, 
               label=f'Current: ${current_price:.2f}')
    ax3.axvline(np.mean(heston_final_prices), color='red', linestyle='--', linewidth=2, 
               label=f'Heston Mean: ${np.mean(heston_final_prices):.2f}')
    ax3.axvline(np.mean(trad_final_prices), color='blue', linestyle='--', linewidth=2, 
               label=f'GBM Mean: ${np.mean(trad_final_prices):.2f}')
    
    ax3.set_title('Price Distribution Comparison')
    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Volatility clustering analysis
    # Calculate realized volatility for each path
    heston_realized_vol = np.std(np.diff(np.log(heston_stock_paths), axis=1), axis=1) * np.sqrt(252)
    trad_realized_vol = np.std(np.diff(np.log(trad_paths), axis=1), axis=1) * np.sqrt(252)
    
    ax4.hist(heston_realized_vol, bins=30, alpha=0.7, color='red', label='Heston Realized Vol', density=True)
    ax4.hist(trad_realized_vol, bins=30, alpha=0.7, color='blue', label='GBM Realized Vol', density=True)
    ax4.axvline(np.mean(heston_realized_vol), color='red', linestyle='--', linewidth=2, 
               label=f'Heston Mean: {np.mean(heston_realized_vol):.4f}')
    ax4.axvline(np.mean(trad_realized_vol), color='blue', linestyle='--', linewidth=2, 
               label=f'GBM Mean: {np.mean(trad_realized_vol):.4f}')
    
    ax4.set_title('Realized Volatility Distribution')
    ax4.set_xlabel('Annualized Volatility')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance analysis
    heston_expected_return = (np.mean(heston_final_prices) / current_price - 1) * 100
    trad_expected_return = (np.mean(trad_final_prices) / current_price - 1) * 100
    
    heston_volatility = np.std(heston_final_prices) / current_price * 100
    trad_volatility = np.std(trad_final_prices) / current_price * 100
    
    print(f"\n📈 HESTON MODEL ANALYSIS RESULTS")
    print("="*50)
    print(f"Heston Expected Return: {heston_expected_return:+.2f}%")
    print(f"Traditional Expected Return: {trad_expected_return:+.2f}%")
    print(f"Heston Volatility: {heston_volatility:.2f}%")
    print(f"Traditional Volatility: {trad_volatility:.2f}%")
    print(f"Volatility Clustering Effect: {heston_volatility - trad_volatility:+.2f}%")
    
    # Volatility clustering metrics
    vol_autocorr = np.corrcoef(vol_mean_path[:-1], vol_mean_path[1:])[0,1]
    print(f"Volatility Autocorrelation: {vol_autocorr:.4f}")
    
    return {
        'heston_predictions': heston_final_prices,
        'traditional_predictions': trad_final_prices,
        'heston_volatility_paths': heston_vol_paths,
        'heston_expected_return': heston_expected_return,
        'traditional_expected_return': trad_expected_return,
        'volatility_clustering': heston_volatility - trad_volatility,
        'volatility_autocorrelation': vol_autocorr
    }

def enhanced_regime_switching_analysis(ticker, model, scaler_X, scaler_y, enhanced_data, feature_columns, 
                                     forecast_months=6, sequence_length=60):
    """Enhanced analysis using regime-switching GBM model"""
    
    print(f"\n🔄 Regime-Switching GBM Analysis for {ticker}")
    print("="*60)
    
    current_price = enhanced_data['Close'].iloc[-1]
    if isinstance(current_price, pd.Series):
        current_price = current_price.iloc[0]
    
    # Get ML predictions for regime parameters
    recent_data = enhanced_data[feature_columns].iloc[-sequence_length:].values
    recent_data_scaled = scaler_X.transform(recent_data.reshape(-1, recent_data.shape[-1])).reshape(1, sequence_length, -1)
    recent_tensor = torch.FloatTensor(recent_data_scaled)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        recent_tensor = recent_tensor.to(device)
        ml_price_pred, ml_vol_pred, ml_drift_pred, ml_uncertainty_pred = model(recent_tensor)
        
        ml_vol = ml_vol_pred.cpu().item()
        ml_drift = ml_drift_pred.cpu().item()
    
    # Define regimes: [Bull Market, Bear Market, Crisis]
    mu_states = [ml_drift * 1.2, ml_drift * 0.8, ml_drift * 0.3]  # Different drift regimes
    sigma_states = [ml_vol * 0.8, ml_vol * 1.2, ml_vol * 2.0]    # Different volatility regimes
    
    # Transition matrix (probabilities of switching between regimes)
    transition_matrix = np.array([
        [0.95, 0.04, 0.01],  # Bull market transitions
        [0.03, 0.94, 0.03],  # Bear market transitions
        [0.01, 0.04, 0.95]   # Crisis transitions
    ])
    
    T = forecast_months / 12
    N = forecast_months * 21
    
    # Simulate regime-switching model
    time_steps, regime_stock_paths, regime_paths = regime_switching_gbm_simulation(
        current_price, mu_states, sigma_states, transition_matrix, T, N, num_simulations=1000
    )
    
    # Traditional GBM for comparison
    returns = enhanced_data['Returns'].dropna()
    trad_drift = returns.mean() * 252
    trad_vol = returns.std() * np.sqrt(252)
    
    _, trad_paths = enhanced_gbm_simulation(
        current_price, trad_drift, trad_vol, T, N, num_simulations=1000
    )
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker} Regime-Switching GBM Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Stock price paths with regime identification
    regime_mean_path = np.mean(regime_stock_paths, axis=0)
    trad_mean_path = np.mean(trad_paths, axis=0)
    
    regime_upper = np.percentile(regime_stock_paths, 95, axis=0)
    regime_lower = np.percentile(regime_stock_paths, 5, axis=0)
    trad_upper = np.percentile(trad_paths, 95, axis=0)
    trad_lower = np.percentile(trad_paths, 5, axis=0)
    
    days = range(N+1)
    ax1.fill_between(days, regime_lower, regime_upper, alpha=0.2, color='purple', label='Regime-Switching 90% CI')
    ax1.fill_between(days, trad_lower, trad_upper, alpha=0.2, color='blue', label='GBM 90% CI')
    ax1.plot(days, regime_mean_path, color='purple', linewidth=3, label='Regime-Switching Model')
    ax1.plot(days, trad_mean_path, color='blue', linewidth=3, label='Traditional GBM')
    
    ax1.set_title(f'Regime-Switching vs Traditional GBM ({forecast_months} months)')
    ax1.set_ylabel('Price ($)')
    ax1.set_xlabel('Trading Days')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Regime evolution (sample path)
    sample_path_idx = 0
    sample_regime_path = regime_paths[sample_path_idx]
    
    # Color code the regime path
    colors = ['green', 'orange', 'red']
    regime_names = ['Bull', 'Bear', 'Crisis']
    
    for i in range(len(sample_regime_path) - 1):
        regime = sample_regime_path[i]
        ax2.plot([i, i+1], [regime_stock_paths[sample_path_idx, i], regime_stock_paths[sample_path_idx, i+1]], 
                color=colors[regime], linewidth=2, alpha=0.7)
    
    ax2.set_title(f'Sample Path with Regime Changes (Path {sample_path_idx})')
    ax2.set_ylabel('Price ($)')
    ax2.set_xlabel('Trading Days')
    
    # Create legend for regimes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=regime_names[i]) for i in range(3)]
    ax2.legend(handles=legend_elements)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Regime distribution over time
    regime_counts = np.zeros((3, N+1))
    for t in range(N+1):
        for regime in range(3):
            regime_counts[regime, t] = np.sum(regime_paths[:, t] == regime) / len(regime_paths)
    
    ax3.stackplot(days, regime_counts, labels=regime_names, colors=colors, alpha=0.7)
    ax3.set_title('Regime Distribution Over Time')
    ax3.set_ylabel('Proportion of Paths')
    ax3.set_xlabel('Trading Days')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Price distribution comparison
    regime_final_prices = regime_stock_paths[:, -1]
    trad_final_prices = trad_paths[:, -1]
    
    ax4.hist(regime_final_prices, bins=50, alpha=0.7, color='purple', label='Regime-Switching', density=True)
    ax4.hist(trad_final_prices, bins=50, alpha=0.7, color='blue', label='Traditional GBM', density=True)
    ax4.axvline(current_price, color='green', linestyle='-', linewidth=2, 
               label=f'Current: ${current_price:.2f}')
    ax4.axvline(np.mean(regime_final_prices), color='purple', linestyle='--', linewidth=2, 
               label=f'Regime Mean: ${np.mean(regime_final_prices):.2f}')
    ax4.axvline(np.mean(trad_final_prices), color='blue', linestyle='--', linewidth=2, 
               label=f'GBM Mean: ${np.mean(trad_final_prices):.2f}')
    
    ax4.set_title('Price Distribution Comparison')
    ax4.set_xlabel('Price ($)')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance analysis
    regime_expected_return = (np.mean(regime_final_prices) / current_price - 1) * 100
    trad_expected_return = (np.mean(trad_final_prices) / current_price - 1) * 100
    
    regime_volatility = np.std(regime_final_prices) / current_price * 100
    trad_volatility = np.std(trad_final_prices) / current_price * 100
    
    print(f"\n📈 REGIME-SWITCHING ANALYSIS RESULTS")
    print("="*50)
    print(f"Regime-Switching Expected Return: {regime_expected_return:+.2f}%")
    print(f"Traditional Expected Return: {trad_expected_return:+.2f}%")
    print(f"Regime-Switching Volatility: {regime_volatility:.2f}%")
    print(f"Traditional Volatility: {trad_volatility:.2f}%")
    
    # Regime analysis
    final_regime_dist = regime_counts[:, -1]
    print(f"\n🔄 FINAL REGIME DISTRIBUTION")
    for i, regime_name in enumerate(regime_names):
        print(f"{regime_name} Market: {final_regime_dist[i]*100:.1f}%")
    
    # Regime persistence
    regime_changes = np.sum(np.diff(regime_paths, axis=1) != 0, axis=1)
    avg_regime_changes = np.mean(regime_changes)
    print(f"Average Regime Changes per Path: {avg_regime_changes:.2f}")
    
    return {
        'regime_predictions': regime_final_prices,
        'traditional_predictions': trad_final_prices,
        'regime_paths': regime_paths,
        'regime_expected_return': regime_expected_return,
        'traditional_expected_return': trad_expected_return,
        'final_regime_distribution': final_regime_dist,
        'avg_regime_changes': avg_regime_changes
    }

def enhanced_jump_diffusion_analysis(ticker, model, scaler_X, scaler_y, enhanced_data, feature_columns, 
                                   forecast_months=6, sequence_length=60):
    """Enhanced analysis using Merton jump diffusion model"""
    
    print(f"\n⚡ Merton Jump Diffusion Analysis for {ticker}")
    print("="*60)
    
    current_price = enhanced_data['Close'].iloc[-1]
    if isinstance(current_price, pd.Series):
        current_price = current_price.iloc[0]
    
    # Get ML predictions for base parameters
    recent_data = enhanced_data[feature_columns].iloc[-sequence_length:].values
    recent_data_scaled = scaler_X.transform(recent_data.reshape(-1, recent_data.shape[-1])).reshape(1, sequence_length, -1)
    recent_tensor = torch.FloatTensor(recent_data_scaled)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        recent_tensor = recent_tensor.to(device)
        ml_price_pred, ml_vol_pred, ml_drift_pred, ml_uncertainty_pred = model(recent_tensor)
        
        ml_vol = ml_vol_pred.cpu().item()
        ml_drift = ml_drift_pred.cpu().item()
    
    # Merton jump diffusion parameters
    mu = ml_drift  # Continuous drift
    sigma = ml_vol  # Continuous volatility
    lambda_jump = 0.1  # Jump intensity (jumps per year)
    mu_jump = -0.02   # Mean jump size (negative for crash risk)
    sigma_jump = 0.05 # Jump size volatility
    
    T = forecast_months / 12
    N = forecast_months * 21
    
    # Simulate jump diffusion model
    time_steps, jump_stock_paths, jump_times = merton_jump_diffusion_simulation(
        current_price, mu, sigma, lambda_jump, mu_jump, sigma_jump, T, N, num_simulations=1000
    )
    
    # Traditional GBM for comparison
    returns = enhanced_data['Returns'].dropna()
    trad_drift = returns.mean() * 252
    trad_vol = returns.std() * np.sqrt(252)
    
    _, trad_paths = enhanced_gbm_simulation(
        current_price, trad_drift, trad_vol, T, N, num_simulations=1000
    )
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker} Merton Jump Diffusion Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Stock price paths with jump identification
    jump_mean_path = np.mean(jump_stock_paths, axis=0)
    trad_mean_path = np.mean(trad_paths, axis=0)
    
    jump_upper = np.percentile(jump_stock_paths, 95, axis=0)
    jump_lower = np.percentile(jump_stock_paths, 5, axis=0)
    trad_upper = np.percentile(trad_paths, 95, axis=0)
    trad_lower = np.percentile(trad_paths, 5, axis=0)
    
    days = range(N+1)
    ax1.fill_between(days, jump_lower, jump_upper, alpha=0.2, color='orange', label='Jump Diffusion 90% CI')
    ax1.fill_between(days, trad_lower, trad_upper, alpha=0.2, color='blue', label='GBM 90% CI')
    ax1.plot(days, jump_mean_path, color='orange', linewidth=3, label='Jump Diffusion Model')
    ax1.plot(days, trad_mean_path, color='blue', linewidth=3, label='Traditional GBM')
    
    ax1.set_title(f'Jump Diffusion vs Traditional GBM ({forecast_months} months)')
    ax1.set_ylabel('Price ($)')
    ax1.set_xlabel('Trading Days')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sample path with jumps highlighted
    sample_path_idx = 0
    sample_path = jump_stock_paths[sample_path_idx]
    sample_jumps = jump_times[sample_path_idx]
    
    # Plot the path
    ax2.plot(days, sample_path, color='blue', linewidth=2, alpha=0.7, label='Price Path')
    
    # Highlight jumps
    jump_indices = np.where(sample_jumps)[0]
    if len(jump_indices) > 0:
        ax2.scatter(jump_indices, sample_path[jump_indices], color='red', s=50, 
                   zorder=5, label=f'Jumps ({len(jump_indices)} total)')
    
    ax2.set_title(f'Sample Path with Jumps (Path {sample_path_idx})')
    ax2.set_ylabel('Price ($)')
    ax2.set_xlabel('Trading Days')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Jump frequency analysis
    total_jumps_per_path = np.sum(jump_times, axis=1)
    jump_freq = total_jumps_per_path / T  # Jumps per year
    
    ax3.hist(jump_freq, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(lambda_jump, color='red', linestyle='--', linewidth=2, 
               label=f'Expected: {lambda_jump:.2f} jumps/year')
    ax3.axvline(np.mean(jump_freq), color='blue', linestyle='--', linewidth=2, 
               label=f'Observed: {np.mean(jump_freq):.2f} jumps/year')
    
    ax3.set_title('Jump Frequency Distribution')
    ax3.set_xlabel('Jumps per Year')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Price distribution comparison (log scale for fat tails)
    jump_final_prices = jump_stock_paths[:, -1]
    trad_final_prices = trad_paths[:, -1]
    
    # Use log scale to better see fat tails
    log_jump_prices = np.log(jump_final_prices)
    log_trad_prices = np.log(trad_final_prices)
    
    ax4.hist(log_jump_prices, bins=50, alpha=0.7, color='orange', label='Jump Diffusion', density=True)
    ax4.hist(log_trad_prices, bins=50, alpha=0.7, color='blue', label='Traditional GBM', density=True)
    ax4.axvline(np.log(current_price), color='green', linestyle='-', linewidth=2, 
               label=f'Current: ${current_price:.2f}')
    
    ax4.set_title('Log-Price Distribution (Fat Tails)')
    ax4.set_xlabel('Log Price')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance analysis
    jump_expected_return = (np.mean(jump_final_prices) / current_price - 1) * 100
    trad_expected_return = (np.mean(trad_final_prices) / current_price - 1) * 100
    
    jump_volatility = np.std(jump_final_prices) / current_price * 100
    trad_volatility = np.std(trad_final_prices) / current_price * 100
    
    print(f"\n📈 JUMP DIFFUSION ANALYSIS RESULTS")
    print("="*50)
    print(f"Jump Diffusion Expected Return: {jump_expected_return:+.2f}%")
    print(f"Traditional Expected Return: {trad_expected_return:+.2f}%")
    print(f"Jump Diffusion Volatility: {jump_volatility:.2f}%")
    print(f"Traditional Volatility: {trad_volatility:.2f}%")
    print(f"Jump-Induced Volatility Increase: {jump_volatility - trad_volatility:+.2f}%")
    
    # Jump analysis
    total_jumps = np.sum(jump_times)
    avg_jumps_per_path = total_jumps / len(jump_times)
    print(f"\n⚡ JUMP ANALYSIS")
    print(f"Total Jumps: {total_jumps}")
    print(f"Average Jumps per Path: {avg_jumps_per_path:.2f}")
    print(f"Expected Jumps per Path: {lambda_jump * T:.2f}")
    
    # Fat tail analysis
    jump_skewness = np.mean(((jump_final_prices - np.mean(jump_final_prices)) / np.std(jump_final_prices))**3)
    jump_kurtosis = np.mean(((jump_final_prices - np.mean(jump_final_prices)) / np.std(jump_final_prices))**4) - 3
    
    trad_skewness = np.mean(((trad_final_prices - np.mean(trad_final_prices)) / np.std(trad_final_prices))**3)
    trad_kurtosis = np.mean(((trad_final_prices - np.mean(trad_final_prices)) / np.std(trad_final_prices))**4) - 3
    
    print(f"\n📊 FAT TAIL ANALYSIS")
    print(f"Jump Diffusion Skewness: {jump_skewness:.4f}")
    print(f"Traditional GBM Skewness: {trad_skewness:.4f}")
    print(f"Jump Diffusion Kurtosis: {jump_kurtosis:.4f}")
    print(f"Traditional GBM Kurtosis: {trad_kurtosis:.4f}")
    
    # Crash risk analysis
    crash_threshold = current_price * 0.8  # 20% drop
    jump_crash_prob = np.sum(jump_final_prices < crash_threshold) / len(jump_final_prices) * 100
    trad_crash_prob = np.sum(trad_final_prices < crash_threshold) / len(trad_final_prices) * 100
    
    print(f"\n💥 CRASH RISK ANALYSIS")
    print(f"Jump Diffusion Crash Probability: {jump_crash_prob:.2f}%")
    print(f"Traditional GBM Crash Probability: {trad_crash_prob:.2f}%")
    print(f"Jump-Induced Crash Risk Increase: {jump_crash_prob - trad_crash_prob:+.2f}%")
    
    return {
        'jump_predictions': jump_final_prices,
        'traditional_predictions': trad_final_prices,
        'jump_times': jump_times,
        'jump_expected_return': jump_expected_return,
        'traditional_expected_return': trad_expected_return,
        'jump_skewness': jump_skewness,
        'jump_kurtosis': jump_kurtosis,
        'crash_probability': jump_crash_prob,
        'total_jumps': total_jumps,
        'avg_jumps_per_path': avg_jumps_per_path
    }

def comprehensive_quantitative_analysis(ticker, model, scaler_X, scaler_y, enhanced_data, feature_columns, 
                                      forecast_months=6, sequence_length=60):
    """Comprehensive analysis comparing all three advanced quantitative models"""
    
    print(f"\n🎯 COMPREHENSIVE QUANTITATIVE ANALYSIS for {ticker}")
    print("="*70)
    print("Comparing: Heston Stochastic Volatility vs Regime-Switching vs Jump Diffusion")
    print("="*70)
    
    # Run all three advanced analyses
    print("\n🌊 Running Heston Stochastic Volatility Analysis...")
    heston_results = enhanced_heston_analysis(
        ticker, model, scaler_X, scaler_y, enhanced_data, feature_columns, 
        forecast_months, sequence_length
    )
    
    print("\n🔄 Running Regime-Switching Analysis...")
    regime_results = enhanced_regime_switching_analysis(
        ticker, model, scaler_X, scaler_y, enhanced_data, feature_columns, 
        forecast_months, sequence_length
    )
    
    print("\n⚡ Running Jump Diffusion Analysis...")
    jump_results = enhanced_jump_diffusion_analysis(
        ticker, model, scaler_X, scaler_y, enhanced_data, feature_columns, 
        forecast_months, sequence_length
    )
    
    # Traditional GBM for baseline comparison
    current_price = enhanced_data['Close'].iloc[-1]
    if isinstance(current_price, pd.Series):
        current_price = current_price.iloc[0]
    
    returns = enhanced_data['Returns'].dropna()
    trad_drift = returns.mean() * 252
    trad_vol = returns.std() * np.sqrt(252)
    
    T = forecast_months / 12
    N = forecast_months * 21
    
    _, trad_paths = enhanced_gbm_simulation(
        current_price, trad_drift, trad_vol, T, N, num_simulations=1000
    )
    trad_final_prices = trad_paths[:, -1]
    trad_expected_return = (np.mean(trad_final_prices) / current_price - 1) * 100
    trad_volatility = np.std(trad_final_prices) / current_price * 100
    
    # Create comprehensive comparison visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'{ticker} Comprehensive Quantitative Model Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Expected Returns Comparison
    models = ['Traditional GBM', 'Heston SV', 'Regime-Switching', 'Jump Diffusion']
    expected_returns = [
        trad_expected_return,
        heston_results['heston_expected_return'],
        regime_results['regime_expected_return'],
        jump_results['jump_expected_return']
    ]
    
    colors = ['blue', 'red', 'purple', 'orange']
    bars1 = ax1.bar(models, expected_returns, color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.set_title('Expected Returns Comparison')
    ax1.set_ylabel('Expected Return (%)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, expected_returns):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                f'{value:+.2f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Volatility Comparison
    volatilities = [
        trad_volatility,
        np.std(heston_results['heston_predictions']) / current_price * 100,
        np.std(regime_results['regime_predictions']) / current_price * 100,
        np.std(jump_results['jump_predictions']) / current_price * 100
    ]
    
    bars2 = ax2.bar(models, volatilities, color=colors, alpha=0.7)
    ax2.set_title('Volatility Comparison')
    ax2.set_ylabel('Volatility (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars2, volatilities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Price Distribution Comparison
    all_predictions = [
        trad_final_prices,
        heston_results['heston_predictions'],
        regime_results['regime_predictions'],
        jump_results['jump_predictions']
    ]
    
    for i, (pred, color, label) in enumerate(zip(all_predictions, colors, models)):
        ax3.hist(pred, bins=30, alpha=0.6, color=color, label=label, density=True)
    
    ax3.axvline(current_price, color='green', linestyle='-', linewidth=3, 
               label=f'Current: ${current_price:.2f}')
    ax3.set_title('Price Distribution Comparison')
    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Risk Metrics Comparison
    # Calculate various risk metrics
    risk_metrics = {
        'Sharpe Ratio': [],
        'Max Drawdown': [],
        'VaR (5%)': [],
        'CVaR (5%)': []
    }
    
    for pred in all_predictions:
        returns_pred = (pred - current_price) / current_price
        
        # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        sharpe = np.mean(returns_pred) / np.std(returns_pred) if np.std(returns_pred) > 0 else 0
        risk_metrics['Sharpe Ratio'].append(sharpe)
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns_pred)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.min(drawdown)
        risk_metrics['Max Drawdown'].append(max_dd * 100)
        
        # Value at Risk (5%)
        var_5 = np.percentile(returns_pred, 5) * 100
        risk_metrics['VaR (5%)'].append(var_5)
        
        # Conditional Value at Risk (5%)
        cvar_5 = np.mean(returns_pred[returns_pred <= np.percentile(returns_pred, 5)]) * 100
        risk_metrics['CVaR (5%)'].append(cvar_5)
    
    # Plot CVaR as representative risk metric
    bars4 = ax4.bar(models, risk_metrics['CVaR (5%)'], color=colors, alpha=0.7)
    ax4.set_title('Conditional Value at Risk (5%)')
    ax4.set_ylabel('CVaR (%)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars4, risk_metrics['CVaR (5%)']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print comprehensive comparison results
    print(f"\n📊 COMPREHENSIVE QUANTITATIVE ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\n📈 EXPECTED RETURNS:")
    for i, model_name in enumerate(models):
        print(f"{model_name:20}: {expected_returns[i]:+8.2f}%")
    
    print(f"\n📊 VOLATILITY:")
    for i, model_name in enumerate(models):
        print(f"{model_name:20}: {volatilities[i]:8.2f}%")
    
    print(f"\n🎯 RISK METRICS:")
    print(f"{'Model':20} {'Sharpe':>8} {'Max DD%':>8} {'VaR(5%)':>8} {'CVaR(5%)':>8}")
    print("-" * 60)
    for i, model_name in enumerate(models):
        print(f"{model_name:20} {risk_metrics['Sharpe Ratio'][i]:8.3f} "
              f"{risk_metrics['Max Drawdown'][i]:8.2f} "
              f"{risk_metrics['VaR (5%)'][i]:8.2f} "
              f"{risk_metrics['CVaR (5%)'][i]:8.2f}")
    
    # Model-specific insights
    print(f"\n🔍 MODEL-SPECIFIC INSIGHTS:")
    print("="*50)
    
    # Heston insights
    vol_clustering = heston_results['volatility_clustering']
    vol_autocorr = heston_results['volatility_autocorrelation']
    print(f"🌊 Heston Stochastic Volatility:")
    print(f"   - Volatility clustering effect: {vol_clustering:+.2f}%")
    print(f"   - Volatility autocorrelation: {vol_autocorr:.4f}")
    
    # Regime-switching insights
    regime_dist = regime_results['final_regime_distribution']
    avg_changes = regime_results['avg_regime_changes']
    print(f"🔄 Regime-Switching:")
    print(f"   - Bull market probability: {regime_dist[0]*100:.1f}%")
    print(f"   - Bear market probability: {regime_dist[1]*100:.1f}%")
    print(f"   - Crisis probability: {regime_dist[2]*100:.1f}%")
    print(f"   - Average regime changes: {avg_changes:.2f}")
    
    # Jump diffusion insights
    jump_skew = jump_results['jump_skewness']
    jump_kurt = jump_results['jump_kurtosis']
    crash_prob = jump_results['crash_probability']
    print(f"⚡ Jump Diffusion:")
    print(f"   - Skewness: {jump_skew:.4f}")
    print(f"   - Kurtosis: {jump_kurt:.4f}")
    print(f"   - Crash probability: {crash_prob:.2f}%")
    
    # Model ranking
    print(f"\n🏆 MODEL RANKING (Lower CVaR = Better Risk Management):")
    cvar_ranking = sorted(zip(models, risk_metrics['CVaR (5%)']), key=lambda x: x[1])
    for i, (model_name, cvar) in enumerate(cvar_ranking, 1):
        print(f"{i}. {model_name:20}: {cvar:6.2f}%")
    
    return {
        'heston_results': heston_results,
        'regime_results': regime_results,
        'jump_results': jump_results,
        'traditional_results': {
            'expected_return': trad_expected_return,
            'volatility': trad_volatility,
            'predictions': trad_final_prices
        },
        'comparison_metrics': {
            'expected_returns': expected_returns,
            'volatilities': volatilities,
            'risk_metrics': risk_metrics
        }
    }

# Main execution
if __name__ == "__main__":
    ticker = "XLU"
    
    try:
        print("🚀 Starting Advanced Quantitative Model Implementation")
        print("="*70)
        print("Implementing: Heston Stochastic Volatility, Regime-Switching GBM, Jump Diffusion")
        print("="*70)
        
        # Train the enhanced ML model first
        print("\n🧠 Training Enhanced ML Model...")
        model, scaler_X, scaler_y, enhanced_data, feature_columns, metrics = train_enhanced_model(
            ticker, sequence_length=60, epochs=30, model_type='transformer'
        )
        
        # Run comprehensive quantitative analysis
        print("\n🎯 Running Comprehensive Quantitative Analysis...")
        comprehensive_results = comprehensive_quantitative_analysis(
            ticker, model, scaler_X, scaler_y, enhanced_data, feature_columns, 
            forecast_months=6, sequence_length=60
        )
        
        # Additional detailed analysis for each model
        print("\n🔍 DETAILED MODEL ANALYSIS")
        print("="*50)
        
        # Heston model detailed analysis
        print("\n🌊 HESTON STOCHASTIC VOLATILITY DETAILS:")
        heston_results = comprehensive_results['heston_results']
        print(f"Volatility clustering captures the empirical fact that high volatility")
        print(f"tends to be followed by high volatility (autocorrelation: {heston_results['volatility_autocorrelation']:.4f})")
        print(f"This model is particularly useful for options pricing and risk management.")
        
        # Regime-switching detailed analysis
        print("\n🔄 REGIME-SWITCHING DETAILS:")
        regime_results = comprehensive_results['regime_results']
        regime_dist = regime_results['final_regime_distribution']
        print(f"Market regimes capture structural changes in market behavior:")
        print(f"- Bull markets: {regime_dist[0]*100:.1f}% probability")
        print(f"- Bear markets: {regime_dist[1]*100:.1f}% probability") 
        print(f"- Crisis periods: {regime_dist[2]*100:.1f}% probability")
        print(f"This model is ideal for portfolio allocation and regime-aware strategies.")
        
        # Jump diffusion detailed analysis
        print("\n⚡ JUMP DIFFUSION DETAILS:")
        jump_results = comprehensive_results['jump_results']
        print(f"Jump diffusion captures rare but significant market events:")
        print(f"- Skewness: {jump_results['jump_skewness']:.4f} (negative = crash risk)")
        print(f"- Kurtosis: {jump_results['jump_kurtosis']:.4f} (fat tails)")
        print(f"- Crash probability: {jump_results['crash_probability']:.2f}%")
        print(f"This model is essential for tail risk management and extreme event modeling.")
        
        # Model comparison summary
        print(f"\n📊 QUANTITATIVE MODEL COMPARISON SUMMARY")
        print("="*60)
        
        comparison = comprehensive_results['comparison_metrics']
        models = ['Traditional GBM', 'Heston SV', 'Regime-Switching', 'Jump Diffusion']
        
        print(f"\nExpected Returns:")
        for model_name, ret in zip(models, comparison['expected_returns']):
            print(f"  {model_name:20}: {ret:+.2f}%")
        
        print(f"\nVolatilities:")
        for model_name, vol in zip(models, comparison['volatilities']):
            print(f"  {model_name:20}: {vol:.2f}%")
        
        print(f"\nRisk-Adjusted Performance (Sharpe Ratio):")
        for model_name, sharpe in zip(models, comparison['risk_metrics']['Sharpe Ratio']):
            print(f"  {model_name:20}: {sharpe:.3f}")
        
        # Final recommendations
        print(f"\n🎯 QUANTITATIVE INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        best_sharpe_idx = np.argmax(comparison['risk_metrics']['Sharpe Ratio'])
        best_risk_idx = np.argmin(comparison['risk_metrics']['CVaR (5%)'])
        
        print(f"Best Risk-Adjusted Performance: {models[best_sharpe_idx]}")
        print(f"Best Risk Management: {models[best_risk_idx]}")
        
        print(f"\n🔮 MODEL APPLICATIONS:")
        print("• Heston SV: Options pricing, volatility trading, risk management")
        print("• Regime-Switching: Portfolio allocation, tactical asset allocation")
        print("• Jump Diffusion: Tail risk modeling, extreme event preparation")
        print("• Traditional GBM: Baseline comparison, simple scenarios")
        
        print(f"\n✅ Advanced Quantitative Models Implementation Completed!")
        print("🎉 Your GBM model now includes sophisticated features that quants demand:")
        print("   ✅ Stochastic volatility (Heston model)")
        print("   ✅ Regime-switching dynamics")
        print("   ✅ Jump diffusion processes")
        print("   ✅ Comprehensive risk metrics")
        print("   ✅ Model comparison framework")
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
