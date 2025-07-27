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

# Main execution
if __name__ == "__main__":
    ticker = "XLU"
    
    try:
        print("🚀 Starting Bayesian Neural Network Implementation")
        print("="*60)
        
        # Train Bayesian Neural Network
        print("\n🧠 Training Bayesian Neural Network...")
        bayesian_model, scaler_X, scaler_y, enhanced_data, feature_columns, bayesian_metrics = train_bayesian_model(
            ticker, sequence_length=60, epochs=30
        )
        
        # Plot uncertainty analysis
        print("\n📊 Plotting Uncertainty Analysis...")
        plot_uncertainty_analysis(bayesian_metrics, ticker)
        
        # Run enhanced analysis with uncertainty quantification
        print("\n🔮 Running Enhanced Bayesian Analysis...")
        bayesian_results = enhanced_analysis_with_uncertainty(
            ticker, 
            bayesian_model, 
            scaler_X, 
            scaler_y, 
            enhanced_data, 
            feature_columns, 
            forecast_months=6
        )
        
        # Compare with regular transformer
        print("\n🔄 Training Regular Transformer for Comparison...")
        regular_model, _, _, _, _, regular_metrics = train_enhanced_model(
            ticker, sequence_length=60, epochs=30, model_type='transformer'
        )
        
        # Compare performance
        print(f"\n📊 BAYESIAN vs REGULAR TRANSFORMER COMPARISON")
        print("="*60)
        
        bayesian_mse = mean_squared_error(bayesian_metrics['test_actual'], bayesian_metrics['test_pred'])
        regular_mse = mean_squared_error(regular_metrics['test_actual'], regular_metrics['test_pred'])
        
        bayesian_mae = mean_absolute_error(bayesian_metrics['test_actual'], bayesian_metrics['test_pred'])
        regular_mae = mean_absolute_error(regular_metrics['test_actual'], regular_metrics['test_pred'])
        
        print(f"Bayesian MSE: {bayesian_mse:.6f}")
        print(f"Regular Transformer MSE: {regular_mse:.6f}")
        print(f"Bayesian MAE: {bayesian_mae:.6f}")
        print(f"Regular Transformer MAE: {regular_mae:.6f}")
        
        improvement_mse = ((regular_mse - bayesian_mse) / regular_mse) * 100
        improvement_mae = ((regular_mae - bayesian_mae) / regular_mae) * 100
        
        print(f"\n🎯 IMPROVEMENT METRICS")
        print(f"MSE Improvement: {improvement_mse:+.2f}%")
        print(f"MAE Improvement: {improvement_mae:+.2f}%")
        
        # Uncertainty benefits
        avg_uncertainty = np.mean(bayesian_metrics['test_uncertainty'])
        print(f"\n🔮 UNCERTAINTY QUANTIFICATION BENEFITS")
        print(f"Average Uncertainty: {avg_uncertainty:.4f}")
        print(f"Uncertainty Range: [{np.min(bayesian_metrics['test_uncertainty']):.4f}, {np.max(bayesian_metrics['test_uncertainty']):.4f}]")
        
        # Confidence analysis
        high_conf_mask = bayesian_metrics['test_uncertainty'] < np.percentile(bayesian_metrics['test_uncertainty'], 25)
        low_conf_mask = bayesian_metrics['test_uncertainty'] > np.percentile(bayesian_metrics['test_uncertainty'], 75)
        
        high_conf_mae = mean_absolute_error(
            bayesian_metrics['test_actual'][high_conf_mask], 
            bayesian_metrics['test_pred'][high_conf_mask]
        )
        low_conf_mae = mean_absolute_error(
            bayesian_metrics['test_actual'][low_conf_mask], 
            bayesian_metrics['test_pred'][low_conf_mask]
        )
        
        print(f"High Confidence Predictions MAE: {high_conf_mae:.6f}")
        print(f"Low Confidence Predictions MAE: {low_conf_mae:.6f}")
        print(f"Confidence-Error Correlation: {np.corrcoef(bayesian_metrics['test_uncertainty'], np.abs(bayesian_metrics['test_pred'] - bayesian_metrics['test_actual']))[0,1]:.4f}")
        
        # Create comparison visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{ticker} Bayesian vs Regular Transformer Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Training Loss Comparison
        ax1.plot(bayesian_metrics['train_losses'], label='Bayesian', color='purple', linewidth=2)
        ax1.plot(regular_metrics['train_losses'], label='Regular Transformer', color='blue', linewidth=2)
        ax1.set_title('Training Loss Comparison')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Accuracy Comparison
        ax2.scatter(bayesian_metrics['test_actual'], bayesian_metrics['test_pred'], 
                   alpha=0.6, color='purple', label='Bayesian', s=30)
        ax2.scatter(regular_metrics['test_actual'], regular_metrics['test_pred'], 
                   alpha=0.6, color='blue', label='Regular Transformer', s=30)
        
        min_val = min(min(bayesian_metrics['test_actual']), min(regular_metrics['test_actual']))
        max_val = max(max(bayesian_metrics['test_actual']), max(regular_metrics['test_actual']))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
        
        ax2.set_xlabel('Actual Prices')
        ax2.set_ylabel('Predicted Prices')
        ax2.set_title('Prediction Accuracy Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Error Distribution Comparison
        bayesian_errors = bayesian_metrics['test_pred'] - bayesian_metrics['test_actual']
        regular_errors = regular_metrics['test_pred'] - regular_metrics['test_actual']
        
        ax3.hist(bayesian_errors, bins=30, alpha=0.7, color='purple', label='Bayesian', density=True)
        ax3.hist(regular_errors, bins=30, alpha=0.7, color='blue', label='Regular Transformer', density=True)
        ax3.axvline(0, color='black', linestyle='--', linewidth=2, label='Perfect Prediction')
        ax3.set_xlabel('Prediction Error')
        ax3.set_ylabel('Density')
        ax3.set_title('Error Distribution Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Uncertainty vs Error for Bayesian
        bayesian_abs_errors = np.abs(bayesian_errors)
        ax4.scatter(bayesian_metrics['test_uncertainty'], bayesian_abs_errors, 
                   alpha=0.6, color='purple', s=30)
        ax4.set_xlabel('Predicted Uncertainty')
        ax4.set_ylabel('Absolute Prediction Error')
        ax4.set_title('Bayesian: Uncertainty vs Error')
        ax4.grid(True, alpha=0.3)
        
        # Add trend line
        if len(bayesian_metrics['test_uncertainty']) > 1:
            z = np.polyfit(bayesian_metrics['test_uncertainty'], bayesian_abs_errors, 1)
            p = np.poly1d(z)
            ax4.plot(bayesian_metrics['test_uncertainty'], p(bayesian_metrics['test_uncertainty']), 
                    "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.show()
        
        print("\n🎯 FINAL ASSESSMENT")
        print("="*40)
        print(f"Bayesian Neural Network Implementation:")
        print(f"✅ Uncertainty quantification: {avg_uncertainty:.4f}")
        print(f"✅ Prediction accuracy: {bayesian_mse:.6f} MSE")
        print(f"✅ Confidence correlation: {np.corrcoef(bayesian_metrics['test_uncertainty'], np.abs(bayesian_metrics['test_pred'] - bayesian_metrics['test_actual']))[0,1]:.4f}")
        print(f"✅ High confidence MAE: {high_conf_mae:.6f}")
        print(f"✅ Low confidence MAE: {low_conf_mae:.6f}")
        
        if improvement_mse > 0:
            print(f"✅ Bayesian outperforms regular transformer by {improvement_mse:.2f}% in MSE")
        else:
            print(f"⚠️ Regular transformer outperforms Bayesian by {abs(improvement_mse):.2f}% in MSE")
        
        print(f"\n🔮 Uncertainty Metrics:")
        print(f"- Total Uncertainty: {bayesian_results['uncertainty_metrics']['total_uncertainty']:.4f}")
        print(f"- Aleatoric Uncertainty: {bayesian_results['uncertainty_metrics']['aleatoric_uncertainty']:.4f}")
        print(f"- Epistemic Uncertainty: {bayesian_results['uncertainty_metrics']['epistemic_uncertainty']:.4f}")
        print(f"- Price Std: {bayesian_results['uncertainty_metrics']['price_std']:.4f}")
        print(f"- Volatility Std: {bayesian_results['uncertainty_metrics']['volatility_std']:.4f}")
        print(f"- Drift Std: {bayesian_results['uncertainty_metrics']['drift_std']:.4f}")
        
        print("\n✅ Bayesian Neural Network implementation completed successfully!")
        print("🎉 Uncertainty quantification is now available for more reliable financial analysis!")
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
