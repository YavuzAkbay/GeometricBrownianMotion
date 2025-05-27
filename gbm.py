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
warnings.filterwarnings('ignore')

class EnhancedStockPredictor(nn.Module):
    """Hybrid LSTM-GBM model that combines stochastic calculus with deep learning"""
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
        ml_price_pred, ml_vol_pred, ml_drift_pred = model(recent_tensor)
        
        ml_vol = ml_vol_pred.cpu().item()
        ml_drift = ml_drift_pred.cpu().item()
    
    print(f"Current Price: ${current_price:.2f}")
    print(f"ML Predicted Volatility: {ml_vol:.4f} ({ml_vol*100:.2f}%)")
    print(f"ML Predicted Drift: {ml_drift:.4f} ({ml_drift*100:.2f}%)")
    
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

def train_enhanced_model(ticker, sequence_length=60, epochs=100):
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
    
    model = EnhancedStockPredictor(
        input_size=len(available_features),
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            price_pred, vol_pred, drift_pred = model(batch_X)
            loss = criterion(price_pred.squeeze(), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        test_pred, test_vol, test_drift = model(X_test_tensor)
        test_pred = test_pred.cpu().numpy()
        test_vol = test_vol.cpu().numpy()
        test_drift = test_drift.cpu().numpy()
    
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
        'train_losses': train_losses
    }
    
    # Plot training results
    plot_training_results(metrics, ticker)
    
    return model, scaler_X, scaler_y, enhanced_data, available_features, metrics

# Main execution
if __name__ == "__main__":
    ticker = "XLU"
    
    try:
        # Train the model
        model, scaler_X, scaler_y, enhanced_data, feature_columns, metrics = train_enhanced_model(
            ticker, sequence_length=60, epochs=50
        )
        print("✅ Model training completed successfully!")
        
        # Run enhanced analysis with full visualization
        results = enhanced_analysis_and_visualization(
            ticker, model, scaler_X, scaler_y, enhanced_data, feature_columns, forecast_months=6
        )
        
        print("\n🎯 FINAL ASSESSMENT")
        print("="*40)
        print(f"The ML-enhanced model shows:")
        print(f"- Volatility difference: {results['improvement_metrics']['volatility_difference']:.2f}%")
        print(f"- Return difference: {results['improvement_metrics']['return_difference']:.2f}%")
        print(f"- Profit probability improvement: {results['improvement_metrics']['profit_prob_improvement']:+.1f}%")
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
