# GBM model with Yang-Zhang volatiily measure. Thought is this should perform better for higher volatility stocks
# Still need to compare estimated parameters (mu and sigma) and implement backtesting to compare.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def read_csv_data(directory, filename):
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path)
    required_columns = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain columns: {', '.join(required_columns)}")
    return df

def calculate_returns(prices):
    return np.log(prices[1:] / prices[:-1])

def yang_zhang_volatility(df, window, periods_per_year):
    """
    Calculate Yang-Zhang volatility estimator.
    
    :param df: DataFrame with 'Open', 'High', 'Low', 'Close' columns
    :param window: Rolling window for volatility calculation
    :param periods_per_year: Number of periods in a year for the desired timeframe
    :return: Series of annualized volatility estimates
    """
    open_price = df['Open']
    close_price = df['Close']
    high_price = df['High']
    low_price = df['Low']
    
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    
    log_ho = (high_price / open_price).apply(np.log)
    log_lo = (low_price / open_price).apply(np.log)
    log_co = (close_price / open_price).apply(np.log)
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    close_vol = rs.rolling(window=window).mean()
    
    oo_returns = open_price.pct_change()
    open_vol = oo_returns.rolling(window=window).var()
    
    co_returns = close_price.pct_change()
    overnight_vol = (co_returns - oo_returns).rolling(window=window).var()
    
    yzvol = open_vol + k * close_vol + (1 - k) * overnight_vol
    
    return np.sqrt(yzvol * periods_per_year)

def estimate_gbm_params(returns, volatility, dt):
    mu = returns.mean() / dt + 0.5 * volatility.iloc[-1]**2
    sigma = volatility.iloc[-1]
    return mu, sigma

def simulate_gbm(S0, mu, sigma, T, dt, num_paths=1000):
    num_steps = int(T / dt)
    times = np.linspace(0, T, num_steps)
    dW = norm.rvs(scale=np.sqrt(dt), size=(num_steps, num_paths))
    
    W = np.cumsum(dW, axis=0)
    time_grid = np.reshape(times, (num_steps, 1))
    
    return S0 * np.exp((mu - 0.5 * sigma**2) * time_grid + sigma * W)

def main(directory, filename):
    # Read data from CSV
    df = read_csv_data(directory, filename)
    
    # Calculate returns
    returns = calculate_returns(df['Close'])
    
    # Calculate Yang-Zhang volatility for different timeframes
    vol_weekly = yang_zhang_volatility(df, window=5, periods_per_year=52)
    vol_biweekly = yang_zhang_volatility(df, window=10, periods_per_year=26)
    
    # Estimate parameters for both timeframes
    dt_weekly = 1/52
    dt_biweekly = 1/26
    mu_weekly, sigma_weekly = estimate_gbm_params(returns, vol_weekly, dt_weekly)
    mu_biweekly, sigma_biweekly = estimate_gbm_params(returns, vol_biweekly, dt_biweekly)
    
    print(f"Weekly parameters: μ (drift) = {mu_weekly:.4f}, σ (volatility) = {sigma_weekly:.4f}")
    print(f"Bi-weekly parameters: μ (drift) = {mu_biweekly:.4f}, σ (volatility) = {sigma_biweekly:.4f}")
    
    # Simulate future paths for both timeframes
    S0 = df['Close'].iloc[-1]  # last observed price
    T = 0.5  # simulate for 6 months
    num_paths = 1000
    
    paths_weekly = simulate_gbm(S0, mu_weekly, sigma_weekly, T, dt_weekly, num_paths)
    paths_biweekly = simulate_gbm(S0, mu_biweekly, sigma_biweekly, T, dt_biweekly, num_paths)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Historical Data')
    for i in range(min(50, num_paths)):  # Plot first 50 paths for clarity
        plt.plot(np.arange(len(df), len(df) + paths_weekly.shape[0]), 
                 paths_weekly[:, i], alpha=0.1, color='blue')
        plt.plot(np.arange(len(df), len(df) + paths_biweekly.shape[0]), 
                 paths_biweekly[:, i], alpha=0.1, color='red')
    plt.plot(np.arange(len(df), len(df) + paths_weekly.shape[0]), 
             np.mean(paths_weekly, axis=1), color='blue', label='Mean Weekly Forecast')
    plt.plot(np.arange(len(df), len(df) + paths_biweekly.shape[0]), 
             np.mean(paths_biweekly, axis=1), color='red', label='Mean Bi-weekly Forecast')
    plt.legend()
    plt.title('GBM with Yang-Zhang Volatility: Historical Data and Simulated Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.show()

    # Plot historical volatilities
    plt.figure(figsize=(12, 6))
    plt.plot(vol_weekly, label='Weekly Volatility')
    plt.plot(vol_biweekly, label='Bi-weekly Volatility')
    plt.title('Historical Yang-Zhang Volatility')
    plt.xlabel('Time Steps')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    plt.show()

    # Calculate and print some statistics
    final_prices_weekly = paths_weekly[-1, :]
    final_prices_biweekly = paths_biweekly[-1, :]
    
    print(f"\nForecasted price statistics after {T} years (Weekly):")
    print(f"Mean: {np.mean(final_prices_weekly):.2f}")
    print(f"Median: {np.median(final_prices_weekly):.2f}")
    print(f"5th percentile: {np.percentile(final_prices_weekly, 5):.2f}")
    print(f"95th percentile: {np.percentile(final_prices_weekly, 95):.2f}")
    
    print(f"\nForecasted price statistics after {T} years (Bi-weekly):")
    print(f"Mean: {np.mean(final_prices_biweekly):.2f}")
    print(f"Median: {np.median(final_prices_biweekly):.2f}")
    print(f"5th percentile: {np.percentile(final_prices_biweekly, 5):.2f}")
    print(f"95th percentile: {np.percentile(final_prices_biweekly, 95):.2f}")

if __name__ == "__main__":
    directory = "C:/Users/markc/PythonAI/EquityAnalysis"  # Replace with your directory path
    filename = "ASTS.csv"  # Replace with your CSV filename
    main(directory, filename)