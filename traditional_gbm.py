#!/usr/bin/env python3
"""
Traditional Geometric Brownian Motion (GBM) Implementation
=========================================================

A clean, simple implementation of traditional GBM for stock price simulation.
This file contains only the basic GBM model without advanced features.

Features:
- Basic GBM simulation
- Parameter estimation from historical data
- Risk metrics calculation
- Simple visualization
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent plot windows
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# BUG FIX NOTE: traditional_gbm_simulation previously used the Euler-Maruyama
# scheme (dS = μS dt + σS dW), which accumulates discretisation bias.
# The exact log-normal solution S_{t+dt} = S_t * exp((μ - ½σ²)dt + σ√dt Z)
# is used here instead. It is exact for any dt and eliminates the O(dt) bias.
# ─────────────────────────────────────────────────────────────────────────────
def traditional_gbm_simulation(S0, mu, sigma, T, N, num_simulations=1000, seed=None):
    """
    Traditional Geometric Brownian Motion Simulation (exact log-normal scheme)

    Parameters:
    - S0: Initial stock price
    - mu: Drift parameter (annualized)
    - sigma: Volatility parameter (annualized)
    - T: Time horizon (in years)
    - N: Number of time steps
    - num_simulations: Number of simulation paths
    - seed: Optional random seed for reproducibility

    Returns:
    - time_steps: Array of time points
    - stock_paths: Array of stock price paths (num_simulations x N+1)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    time_steps = np.linspace(0, T, N + 1)

    # ── BUG FIX: vectorised exact GBM – no per-path Python loop ──────────────
    # Draw all increments at once: shape (num_simulations, N)
    Z = np.random.standard_normal((num_simulations, N))
    # log-increments: (μ - ½σ²)dt + σ√dt·Z
    log_increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    # Prepend a column of zeros (log of S0) then cumsum to get log prices
    log_paths = np.concatenate(
        [np.zeros((num_simulations, 1)), np.cumsum(log_increments, axis=1)],
        axis=1,
    )
    stock_paths = S0 * np.exp(log_paths)

    return time_steps, stock_paths


# ─────────────────────────────────────────────────────────────────────────────
# BUG FIX NOTE: estimate_gbm_parameters previously returned
#   mu_hat = mean(log_returns) * 252
# which estimates the *log-return* mean m = μ - ½σ².  To recover the true GBM
# drift μ (the arithmetic drift that appears in E[S_T] = S_0 e^{μT}) we must
# add back the Itô correction:
#   μ = m + ½σ²
# Without this, every simulation is systematically biased downward.
# ─────────────────────────────────────────────────────────────────────────────
def estimate_gbm_parameters(stock_data, period='1y'):
    """
    Estimate GBM parameters from historical stock data.

    Returns:
    - mu   : Estimated GBM drift μ (annualized, Itô-corrected)
    - sigma: Estimated annualized volatility σ
    """
    close = stock_data['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    log_returns = np.log(close / close.shift(1)).dropna()

    sigma = log_returns.std() * np.sqrt(252)          # annualised vol
    m     = log_returns.mean() * 252                  # annualised log-return mean
    mu    = m + 0.5 * sigma ** 2                      # BUG FIX: Itô correction

    return mu, sigma


# ─────────────────────────────────────────────────────────────────────────────
# BUG FIX NOTE: Sharpe ratio previously computed mean(R)/std(R) without
# subtracting a risk-free rate.  That is the *reward-to-variability* ratio,
# not the Sharpe ratio.  A standard annualised risk-free rate is deducted.
# The ratio also needs to be annualised; for a T-year horizon the conversion
# is sqrt(1/T) * (mean/std).
# ─────────────────────────────────────────────────────────────────────────────
def calculate_risk_metrics(final_prices, initial_price, T=1.0, risk_free_rate=0.03):
    """
    Calculate comprehensive risk metrics for GBM simulation results.

    Parameters:
    - final_prices   : Array of final prices from simulation
    - initial_price  : Initial stock price
    - T              : Simulation horizon in years (used for annualisation)
    - risk_free_rate : Annualised risk-free rate (default 3 %)

    Returns:
    - Dictionary containing various risk metrics
    """
    returns = (final_prices - initial_price) / initial_price

    mean_r = np.mean(returns)
    std_r  = np.std(returns, ddof=1)

    # ── BUG FIX: proper annualised Sharpe ratio ───────────────────────────────
    # Convert the total-period risk-free to match horizon T
    rf_period = (1 + risk_free_rate) ** T - 1          # e.g. 3% annual → T-year
    excess_r  = mean_r - rf_period
    ann_factor = np.sqrt(1.0 / T) if T > 0 else 1.0   # annualise to 1-yr basis
    sharpe = ann_factor * excess_r / std_r if std_r > 0 else 0.0

    var_threshold = np.percentile(returns, 5)

    metrics = {
        'expected_return'   : mean_r * 100,
        'volatility'        : std_r * 100,
        'sharpe_ratio'      : sharpe,                  # BUG FIX: risk-free adjusted + annualised
        'var_5'             : var_threshold * 100,
        'cvar_5'            : np.mean(returns[returns <= var_threshold]) * 100,
        'max_drawdown'      : np.min(returns) * 100,
        'skewness'          : np.mean(((returns - mean_r) / std_r) ** 3) if std_r > 0 else 0.0,
        'kurtosis'          : np.mean(((returns - mean_r) / std_r) ** 4) - 3 if std_r > 0 else 0.0,
        'profit_probability': np.sum(returns > 0) / len(returns) * 100,
    }

    return metrics


def plot_gbm_analysis(time_steps, stock_paths, initial_price, ticker, forecast_months):
    """
    Create comprehensive visualization of GBM simulation results.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker} Traditional GBM Analysis ({forecast_months} months)',
                 fontsize=16, fontweight='bold')

    # Plot 1: Sample paths
    sample_paths = min(10, len(stock_paths))
    for i in range(sample_paths):
        ax1.plot(time_steps, stock_paths[i], alpha=0.7, linewidth=1)
    ax1.plot(time_steps, np.mean(stock_paths, axis=0), 'r-', linewidth=3, label='Mean Path')
    ax1.axhline(y=initial_price, color='black', linestyle='--', alpha=0.7,
                label=f'Initial: ${initial_price:.2f}')
    ax1.set_title('GBM Sample Paths')
    ax1.set_ylabel('Stock Price ($)')
    ax1.set_xlabel('Time (years)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final price distribution
    final_prices = stock_paths[:, -1]
    ax2.hist(final_prices, bins=50, alpha=0.7, color='blue', edgecolor='black', density=True)
    ax2.axvline(initial_price, color='red', linestyle='-', linewidth=2,
                label=f'Initial: ${initial_price:.2f}')
    ax2.axvline(np.mean(final_prices), color='green', linestyle='--', linewidth=2,
                label=f'Mean: ${np.mean(final_prices):.2f}')
    ax2.set_title('Final Price Distribution')
    ax2.set_xlabel('Final Price ($)')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Confidence intervals
    mean_path  = np.mean(stock_paths, axis=0)
    upper_95   = np.percentile(stock_paths, 95, axis=0)
    lower_5    = np.percentile(stock_paths, 5,  axis=0)
    upper_75   = np.percentile(stock_paths, 75, axis=0)
    lower_25   = np.percentile(stock_paths, 25, axis=0)

    ax3.fill_between(time_steps, lower_5, upper_95, alpha=0.2, color='red',    label='90% CI')
    ax3.fill_between(time_steps, lower_25, upper_75, alpha=0.3, color='orange', label='50% CI')
    ax3.plot(time_steps, mean_path, 'b-', linewidth=3, label='Mean Path')
    ax3.axhline(y=initial_price, color='black', linestyle='--', alpha=0.7,
                label=f'Initial: ${initial_price:.2f}')
    ax3.set_title('Price Evolution with Confidence Intervals')
    ax3.set_ylabel('Stock Price ($)')
    ax3.set_xlabel('Time (years)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Risk metrics
    T = time_steps[-1]
    metrics = calculate_risk_metrics(final_prices, initial_price, T=T)

    metric_names  = ['Expected Return', 'Volatility', 'Sharpe Ratio', 'VaR (5%)', 'CVaR (5%)']
    metric_values = [
        metrics['expected_return'],
        metrics['volatility'],
        metrics['sharpe_ratio'],
        metrics['var_5'],
        metrics['cvar_5'],
    ]
    bars = ax4.bar(metric_names, metric_values,
                   color=['green', 'blue', 'orange', 'red', 'purple'], alpha=0.7)
    ax4.set_title('Risk Metrics')
    ax4.set_ylabel('Value (%)')
    ax4.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.,
                 height + (0.1 if height >= 0 else -0.3),
                 f'{value:.2f}',
                 ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()


def analyze_stock_gbm(ticker, forecast_months=6, num_simulations=1000):
    """
    Complete GBM analysis for a given stock ticker.
    """
    print(f"📈 Traditional GBM Analysis for {ticker}")
    print("=" * 50)

    print(f"📊 Fetching data for {ticker}...")

    # Robust fetch: try yf.download first, fall back to yf.Ticker.history
    # This handles the yfinance bug where it prepends '$' to the ticker symbol
    # causing "possibly delisted; no price data found" for valid tickers.
    # Fix: upgrade yfinance  →  pip install --upgrade yfinance
    stock_data = yf.download(ticker, period='2y', auto_adjust=True, progress=False)

    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)

    if stock_data.empty:
        print(f"⚠ yf.download failed for {ticker}, retrying with Ticker.history()...")
        try:
            t = yf.Ticker(ticker)
            stock_data = t.history(period='2y', auto_adjust=True)
            stock_data.index = stock_data.index.tz_localize(None)
        except Exception as e:
            raise ValueError(
                f"Could not fetch data for {ticker}.\n"
                f"  → Try: pip install --upgrade yfinance\n"
                f"  → Original error: {e}"
            )

    if stock_data.empty:
        raise ValueError(
            f"Could not fetch data for {ticker}.\n"
            f"  → Try: pip install --upgrade yfinance"
        )

    current_price = float(stock_data['Close'].iloc[-1])
    print(f"Current Price: ${current_price:.2f}")

    print("🔍 Estimating GBM parameters (Itô-corrected)...")
    mu, sigma = estimate_gbm_parameters(stock_data)
    print(f"Estimated Drift (μ): {mu:.4f} ({mu*100:.2f}% annual)")
    print(f"Estimated Volatility (σ): {sigma:.4f} ({sigma*100:.2f}% annual)")

    T = forecast_months / 12
    N = forecast_months * 21

    print(f"Simulation Parameters:")
    print(f"  Time Horizon: {T:.2f} years ({forecast_months} months)")
    print(f"  Time Steps:   {N} (daily)")
    print(f"  Simulations:  {num_simulations}")

    print("🚀 Running GBM simulation (exact log-normal)...")
    time_steps, stock_paths = traditional_gbm_simulation(
        current_price, mu, sigma, T, N, num_simulations, seed=42
    )

    print("📊 Calculating risk metrics...")
    final_prices = stock_paths[:, -1]
    metrics = calculate_risk_metrics(final_prices, current_price, T=T)

    print(f"\n📈 GBM ANALYSIS RESULTS")
    print("=" * 40)
    print(f"Expected Return:    {metrics['expected_return']:+.2f}%")
    print(f"Volatility:         {metrics['volatility']:.2f}%")
    print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.3f}  (risk-free adjusted, annualised)")
    print(f"VaR (5%):           {metrics['var_5']:.2f}%")
    print(f"CVaR (5%):          {metrics['cvar_5']:.2f}%")
    print(f"Profit Probability: {metrics['profit_probability']:.1f}%")
    print(f"Skewness:           {metrics['skewness']:.3f}")
    print(f"Kurtosis:           {metrics['kurtosis']:.3f}")

    print(f"\n💰 PRICE STATISTICS")
    print("=" * 40)
    print(f"Mean Final Price:   ${np.mean(final_prices):.2f}")
    print(f"Median Final Price: ${np.median(final_prices):.2f}")
    print(f"Min Final Price:    ${np.min(final_prices):.2f}")
    print(f"Max Final Price:    ${np.max(final_prices):.2f}")
    print(f"Price Range:        ${np.max(final_prices) - np.min(final_prices):.2f}")

    percentiles = [5, 25, 50, 75, 95]
    print(f"\n📊 PRICE PERCENTILES")
    print("=" * 40)
    for p in percentiles:
        print(f"{p}th percentile: ${np.percentile(final_prices, p):.2f}")

    print(f"\n📊 Creating visualization...")
    plot_gbm_analysis(time_steps, stock_paths, current_price, ticker, forecast_months)

    return {
        'ticker'          : ticker,
        'current_price'   : current_price,
        'mu'              : mu,
        'sigma'           : sigma,
        'time_steps'      : time_steps,
        'stock_paths'     : stock_paths,
        'final_prices'    : final_prices,
        'metrics'         : metrics,
        'forecast_months' : forecast_months,
    }


def compare_multiple_stocks(tickers, forecast_months=6, num_simulations=1000):
    """
    Compare GBM analysis across multiple stocks.
    """
    print(f"🔍 Multi-Stock GBM Comparison")
    print("=" * 50)

    results = {}
    for ticker in tickers:
        print(f"\n📈 Analyzing {ticker}...")
        try:
            results[ticker] = analyze_stock_gbm(ticker, forecast_months, num_simulations)
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {e}")

    if len(results) < 2:
        print("❌ Need at least 2 successful analyses for comparison")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Multi-Stock GBM Comparison ({forecast_months} months)',
                 fontsize=16, fontweight='bold')

    ticker_names     = list(results.keys())
    expected_returns = [results[t]['metrics']['expected_return'] for t in ticker_names]
    volatilities     = [results[t]['metrics']['volatility']       for t in ticker_names]
    sharpe_ratios    = [results[t]['metrics']['sharpe_ratio']     for t in ticker_names]

    # Plot 1
    bars1 = ax1.bar(ticker_names, expected_returns, color='green', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.set_title('Expected Returns Comparison')
    ax1.set_ylabel('Expected Return (%)')
    ax1.tick_params(axis='x', rotation=45)
    for bar, v in zip(bars1, expected_returns):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h + (0.1 if h >= 0 else -0.3),
                 f'{v:+.2f}%', ha='center', va='bottom' if h >= 0 else 'top', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2
    bars2 = ax2.bar(ticker_names, volatilities, color='blue', alpha=0.7)
    ax2.set_title('Volatility Comparison')
    ax2.set_ylabel('Volatility (%)')
    ax2.tick_params(axis='x', rotation=45)
    for bar, v in zip(bars2, volatilities):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., h + h * 0.01,
                 f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3
    bars3 = ax3.bar(ticker_names, sharpe_ratios, color='orange', alpha=0.7)
    ax3.set_title('Sharpe Ratio Comparison (risk-free adjusted)')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.tick_params(axis='x', rotation=45)
    for bar, v in zip(bars3, sharpe_ratios):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., h + h * 0.01,
                 f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4
    for ticker in ticker_names:
        ax4.hist(results[ticker]['final_prices'], bins=30, alpha=0.6,
                 label=ticker, density=True)
    ax4.set_title('Final Price Distributions')
    ax4.set_xlabel('Final Price ($)')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    print(f"\n📊 COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Ticker':<10} {'Return%':<10} {'Vol%':<10} {'Sharpe':<10} {'VaR%':<10} {'CVaR%':<10}")
    print("-" * 60)
    for t in ticker_names:
        m = results[t]['metrics']
        print(f"{t:<10} {m['expected_return']:>+8.2f} {m['volatility']:>8.2f} "
              f"{m['sharpe_ratio']:>8.3f} {m['var_5']:>8.2f} {m['cvar_5']:>8.2f}")

    return results


# Main execution
if __name__ == "__main__":
    print("🚀 Traditional GBM Analysis")
    print("=" * 40)

    ticker = "AAPL"
    print(f"\n📈 Analyzing {ticker}...")
    result = analyze_stock_gbm(ticker, forecast_months=6, num_simulations=1000)

    print(f"\n✅ Traditional GBM analysis completed!")
    print("🎉 This is the basic GBM model without advanced features.")
