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
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def traditional_gbm_simulation(S0, mu, sigma, T, N, num_simulations=1000):
    """
    Traditional Geometric Brownian Motion Simulation
    
    Parameters:
    - S0: Initial stock price
    - mu: Drift parameter (annualized)
    - sigma: Volatility parameter (annualized)
    - T: Time horizon (in years)
    - N: Number of time steps
    - num_simulations: Number of simulation paths
    
    Returns:
    - time_steps: Array of time points
    - stock_paths: Array of stock price paths
    """
    dt = T / N
    time_steps = np.linspace(0, T, N+1)
    
    # Initialize array for stock paths
    stock_paths = np.zeros((num_simulations, N+1))
    stock_paths[:, 0] = S0
    
    np.random.seed(42)  # For reproducibility
    
    for i in range(num_simulations):
        for t in range(N):
            # Generate random increment
            dW = np.random.normal(0, np.sqrt(dt))
            
            # GBM formula: dS = μSdt + σSdW
            dS = mu * stock_paths[i, t] * dt + sigma * stock_paths[i, t] * dW
            stock_paths[i, t+1] = stock_paths[i, t] + dS
    
    return time_steps, stock_paths

def estimate_gbm_parameters(stock_data, period='1y'):
    """
    Estimate GBM parameters from historical stock data
    
    Parameters:
    - stock_data: DataFrame with 'Close' prices
    - period: Time period for parameter estimation
    
    Returns:
    - mu: Estimated drift parameter (annualized)
    - sigma: Estimated volatility parameter (annualized)
    """
    # Calculate log returns
    returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1)).dropna()
    
    # Estimate parameters
    mu = returns.mean() * 252  # Annualize
    sigma = returns.std() * np.sqrt(252)  # Annualize
    
    return mu, sigma

def calculate_risk_metrics(final_prices, initial_price):
    """
    Calculate comprehensive risk metrics for GBM simulation results
    
    Parameters:
    - final_prices: Array of final prices from simulation
    - initial_price: Initial stock price
    
    Returns:
    - Dictionary containing various risk metrics
    """
    returns = (final_prices - initial_price) / initial_price
    
    metrics = {
        'expected_return': np.mean(returns) * 100,
        'volatility': np.std(returns) * 100,
        'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
        'var_5': np.percentile(returns, 5) * 100,
        'cvar_5': np.mean(returns[returns <= np.percentile(returns, 5)]) * 100,
        'max_drawdown': np.min(returns) * 100,
        'skewness': np.mean(((returns - np.mean(returns)) / np.std(returns))**3),
        'kurtosis': np.mean(((returns - np.mean(returns)) / np.std(returns))**4) - 3,
        'profit_probability': np.sum(returns > 0) / len(returns) * 100
    }
    
    return metrics

def plot_gbm_analysis(time_steps, stock_paths, initial_price, ticker, forecast_months):
    """
    Create comprehensive visualization of GBM simulation results
    
    Parameters:
    - time_steps: Array of time points
    - stock_paths: Array of stock price paths
    - initial_price: Initial stock price
    - ticker: Stock ticker symbol
    - forecast_months: Forecast period in months
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{ticker} Traditional GBM Analysis ({forecast_months} months)', fontsize=16, fontweight='bold')
    
    # Plot 1: Sample paths
    sample_paths = min(10, len(stock_paths))
    for i in range(sample_paths):
        ax1.plot(time_steps, stock_paths[i], alpha=0.7, linewidth=1)
    
    ax1.plot(time_steps, np.mean(stock_paths, axis=0), 'r-', linewidth=3, label='Mean Path')
    ax1.axhline(y=initial_price, color='black', linestyle='--', alpha=0.7, label=f'Initial: ${initial_price:.2f}')
    
    ax1.set_title('GBM Sample Paths')
    ax1.set_ylabel('Stock Price ($)')
    ax1.set_xlabel('Time (years)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final price distribution
    final_prices = stock_paths[:, -1]
    ax2.hist(final_prices, bins=50, alpha=0.7, color='blue', edgecolor='black', density=True)
    ax2.axvline(initial_price, color='red', linestyle='-', linewidth=2, label=f'Initial: ${initial_price:.2f}')
    ax2.axvline(np.mean(final_prices), color='green', linestyle='--', linewidth=2, 
               label=f'Mean: ${np.mean(final_prices):.2f}')
    
    ax2.set_title('Final Price Distribution')
    ax2.set_xlabel('Final Price ($)')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confidence intervals
    mean_path = np.mean(stock_paths, axis=0)
    upper_95 = np.percentile(stock_paths, 95, axis=0)
    lower_5 = np.percentile(stock_paths, 5, axis=0)
    upper_75 = np.percentile(stock_paths, 75, axis=0)
    lower_25 = np.percentile(stock_paths, 25, axis=0)
    
    ax3.fill_between(time_steps, lower_5, upper_95, alpha=0.2, color='red', label='90% CI')
    ax3.fill_between(time_steps, lower_25, upper_75, alpha=0.3, color='orange', label='50% CI')
    ax3.plot(time_steps, mean_path, 'b-', linewidth=3, label='Mean Path')
    ax3.axhline(y=initial_price, color='black', linestyle='--', alpha=0.7, label=f'Initial: ${initial_price:.2f}')
    
    ax3.set_title('Price Evolution with Confidence Intervals')
    ax3.set_ylabel('Stock Price ($)')
    ax3.set_xlabel('Time (years)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Risk metrics
    metrics = calculate_risk_metrics(final_prices, initial_price)
    
    metric_names = ['Expected Return', 'Volatility', 'Sharpe Ratio', 'VaR (5%)', 'CVaR (5%)']
    metric_values = [
        metrics['expected_return'],
        metrics['volatility'],
        metrics['sharpe_ratio'],
        metrics['var_5'],
        metrics['cvar_5']
    ]
    
    bars = ax4.bar(metric_names, metric_values, color=['green', 'blue', 'orange', 'red', 'purple'], alpha=0.7)
    ax4.set_title('Risk Metrics')
    ax4.set_ylabel('Value (%)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_stock_gbm(ticker, forecast_months=6, num_simulations=1000):
    """
    Complete GBM analysis for a given stock ticker
    
    Parameters:
    - ticker: Stock ticker symbol
    - forecast_months: Forecast period in months
    - num_simulations: Number of simulation paths
    
    Returns:
    - Dictionary containing analysis results
    """
    print(f"📈 Traditional GBM Analysis for {ticker}")
    print("="*50)
    
    # Fetch stock data
    print(f"📊 Fetching data for {ticker}...")
    stock_data = yf.download(ticker, period='2y')
    
    if stock_data.empty:
        raise ValueError(f"Could not fetch data for {ticker}")
    
    # Get current price
    current_price = stock_data['Close'].iloc[-1]
    print(f"Current Price: ${current_price:.2f}")
    
    # Estimate GBM parameters
    print("🔍 Estimating GBM parameters...")
    mu, sigma = estimate_gbm_parameters(stock_data)
    print(f"Estimated Drift (μ): {mu:.4f} ({mu*100:.2f}% annual)")
    print(f"Estimated Volatility (σ): {sigma:.4f} ({sigma*100:.2f}% annual)")
    
    # Set up simulation parameters
    T = forecast_months / 12  # Convert months to years
    N = forecast_months * 21  # Approximate trading days
    
    print(f"Simulation Parameters:")
    print(f"  Time Horizon: {T:.2f} years ({forecast_months} months)")
    print(f"  Time Steps: {N} (daily)")
    print(f"  Simulations: {num_simulations}")
    
    # Run GBM simulation
    print("🚀 Running GBM simulation...")
    time_steps, stock_paths = traditional_gbm_simulation(
        current_price, mu, sigma, T, N, num_simulations
    )
    
    # Calculate risk metrics
    print("📊 Calculating risk metrics...")
    final_prices = stock_paths[:, -1]
    metrics = calculate_risk_metrics(final_prices, current_price)
    
    # Display results
    print(f"\n📈 GBM ANALYSIS RESULTS")
    print("="*40)
    print(f"Expected Return: {metrics['expected_return']:+.2f}%")
    print(f"Volatility: {metrics['volatility']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"VaR (5%): {metrics['var_5']:.2f}%")
    print(f"CVaR (5%): {metrics['cvar_5']:.2f}%")
    print(f"Profit Probability: {metrics['profit_probability']:.1f}%")
    print(f"Skewness: {metrics['skewness']:.3f}")
    print(f"Kurtosis: {metrics['kurtosis']:.3f}")
    
    # Price statistics
    print(f"\n💰 PRICE STATISTICS")
    print("="*40)
    print(f"Mean Final Price: ${np.mean(final_prices):.2f}")
    print(f"Median Final Price: ${np.median(final_prices):.2f}")
    print(f"Min Final Price: ${np.min(final_prices):.2f}")
    print(f"Max Final Price: ${np.max(final_prices):.2f}")
    print(f"Price Range: ${np.max(final_prices) - np.min(final_prices):.2f}")
    
    # Percentiles
    percentiles = [5, 25, 50, 75, 95]
    print(f"\n📊 PRICE PERCENTILES")
    print("="*40)
    for p in percentiles:
        price_p = np.percentile(final_prices, p)
        print(f"{p}th percentile: ${price_p:.2f}")
    
    # Create visualization
    print(f"\n📊 Creating visualization...")
    plot_gbm_analysis(time_steps, stock_paths, current_price, ticker, forecast_months)
    
    return {
        'ticker': ticker,
        'current_price': current_price,
        'mu': mu,
        'sigma': sigma,
        'time_steps': time_steps,
        'stock_paths': stock_paths,
        'final_prices': final_prices,
        'metrics': metrics,
        'forecast_months': forecast_months
    }

def compare_multiple_stocks(tickers, forecast_months=6, num_simulations=1000):
    """
    Compare GBM analysis across multiple stocks
    
    Parameters:
    - tickers: List of stock ticker symbols
    - forecast_months: Forecast period in months
    - num_simulations: Number of simulation paths
    """
    print(f"🔍 Multi-Stock GBM Comparison")
    print("="*50)
    
    results = {}
    
    for ticker in tickers:
        print(f"\n📈 Analyzing {ticker}...")
        try:
            result = analyze_stock_gbm(ticker, forecast_months, num_simulations)
            results[ticker] = result
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {str(e)}")
    
    if len(results) < 2:
        print("❌ Need at least 2 successful analyses for comparison")
        return
    
    # Create comparison visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Multi-Stock GBM Comparison ({forecast_months} months)', fontsize=16, fontweight='bold')
    
    # Plot 1: Expected returns comparison
    ticker_names = list(results.keys())
    expected_returns = [results[t]['metrics']['expected_return'] for t in ticker_names]
    
    bars1 = ax1.bar(ticker_names, expected_returns, color='green', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.set_title('Expected Returns Comparison')
    ax1.set_ylabel('Expected Return (%)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars1, expected_returns):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                f'{value:+.2f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Volatility comparison
    volatilities = [results[t]['metrics']['volatility'] for t in ticker_names]
    
    bars2 = ax2.bar(ticker_names, volatilities, color='blue', alpha=0.7)
    ax2.set_title('Volatility Comparison')
    ax2.set_ylabel('Volatility (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars2, volatilities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sharpe ratio comparison
    sharpe_ratios = [results[t]['metrics']['sharpe_ratio'] for t in ticker_names]
    
    bars3 = ax3.bar(ticker_names, sharpe_ratios, color='orange', alpha=0.7)
    ax3.set_title('Sharpe Ratio Comparison')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars3, sharpe_ratios):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final price distributions
    for i, ticker in enumerate(ticker_names):
        final_prices = results[ticker]['final_prices']
        ax4.hist(final_prices, bins=30, alpha=0.6, label=ticker, density=True)
    
    ax4.set_title('Final Price Distributions')
    ax4.set_xlabel('Final Price ($)')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison table
    print(f"\n📊 COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Ticker':<10} {'Return%':<10} {'Vol%':<10} {'Sharpe':<10} {'VaR%':<10} {'CVaR%':<10}")
    print("-" * 60)
    
    for ticker in ticker_names:
        metrics = results[ticker]['metrics']
        print(f"{ticker:<10} {metrics['expected_return']:>+8.2f} {metrics['volatility']:>8.2f} "
              f"{metrics['sharpe_ratio']:>8.3f} {metrics['var_5']:>8.2f} {metrics['cvar_5']:>8.2f}")
    
    return results

# Main execution
if __name__ == "__main__":
    # Example usage
    print("🚀 Traditional GBM Analysis")
    print("="*40)
    
    # Single stock analysis
    ticker = "AAPL"
    print(f"\n📈 Analyzing {ticker}...")
    result = analyze_stock_gbm(ticker, forecast_months=6, num_simulations=1000)
    
    # Multi-stock comparison (uncomment to use)
    # print(f"\n🔍 Multi-stock comparison...")
    # tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    # results = compare_multiple_stocks(tickers, forecast_months=6, num_simulations=1000)
    
    print(f"\n✅ Traditional GBM analysis completed!")
    print("🎉 This is the basic GBM model without advanced features.")
