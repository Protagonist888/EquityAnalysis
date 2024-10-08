# Geometric Browian Motion algorithm for stock prices
# Uses Garman-Klass estimator to measure volatlity
## Hypothesize that better measuring and predicting volatility will give me higher confidence in 
# future price movements (withint 1 to 2 weeks) when overall volatility is high
# Try another version of this, but with Yang-Zhang estimator which would be better in markets with drift


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def read_csv_data(directory, filename):
    """
    Read price data from a CSV file.
    
    :param directory: Directory containing the CSV file
    :param filename: Name of the CSV file
    :return: pandas DataFrame with price data
    """
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path)
    required_columns = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain columns: {', '.join(required_columns)}")
    return df

def calculate_returns(prices):
    """Calculate log returns from price data."""
    return np.log(prices[1:] / prices[:-1])

def garman_klass_volatility(df, window=10, periods_per_year=26):
    """
    Calculate Garman-Klass volatility estimator.
    
    :param df: DataFrame with 'Open', 'High', 'Low', 'Close' columns
    :param window: Rolling window for volatility calculation
    :return: Series of annualized volatility estimates
    """
    log_hl = (df['High'] / df['Low']).apply(np.log)
    log_co = (df['Close'] / df['Open']).apply(np.log)
    
    rs = 0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
    
    return np.sqrt(rs.rolling(window=window).mean())

def estimate_gbm_params(returns, volatility, dt=1/252):
    """
    Estimate parameters for Geometric Brownian Motion.
    
    :param returns: Log returns
    :param volatility: Garman-Klass volatility estimate
    :param dt: Time step (default is daily)
    :return: mu (drift), sigma (volatility)
    """
    mu = returns.mean() / dt + 0.5 * volatility.iloc[-1]**2
    sigma = volatility.iloc[-1]  # Use the most recent volatility estimate
    return mu, sigma

def simulate_gbm(S0, mu, sigma, T, dt, num_paths=1000):
    """
    Simulate Geometric Brownian Motion.
    
    :param S0: Initial stock price
    :param mu: Drift coefficient
    :param sigma: Volatility
    :param T: Total time
    :param dt: Time step
    :param num_paths: Number of simulation paths
    :return: Simulated price paths
    """
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

    # Calculate Garman-Klass volatility for different timeframes
    volatility_annual = garman_klass_volatility(df, window=30, periods_per_year=252) # annual (original)
    volatility_biweekly = garman_klass_volatility(df) #bi-weekly (new default)
    volatility_weekly = garman_klass_volatility(df, window=5, periods_per_year=52)

    # Estimate parameters
    dt = 1/26  # assuming daily data
    mu, sigma = estimate_gbm_params(returns, volatility_biweekly, dt)
    
    print(f"Estimated parameters: μ (drift) = {mu:.4f}, σ (volatility) = {sigma:.4f}")
    
    # Simulate future paths
    S0 = df['Close'].iloc[-1]  # last observed price
    T = 26/52  # simulate for 26 weeks (half a year)
    num_paths = 1000
    simulated_paths = simulate_gbm(S0, mu, sigma, T, dt, num_paths)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Historical Data')
    for i in range(min(100, num_paths)):  # Plot first 100 paths for clarity
        plt.plot(np.arange(len(df), len(df) + simulated_paths.shape[0]), 
                 simulated_paths[:, i], alpha=0.1, color='gray')
    plt.plot(np.arange(len(df), len(df) + simulated_paths.shape[0]), 
             np.mean(simulated_paths, axis=1), color='red', label='Mean Forecast')
    plt.legend()
    plt.title('GBM with Garman-Klass Volatility: Historical Data and Simulated Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.show()

   # Plot historical volatilities
    plt.figure(figsize=(12, 6))
    plt.plot(volatility_annual, label='Annual Volatility')
    plt.plot(volatility_biweekly, label='Bi-weekly Volatility')
    plt.plot(volatility_weekly, label='Weekly Volatility')
    plt.title('Historical Garman-Klass Volatility Estimates')
    plt.xlabel('Time Steps')
    plt.ylabel('Annualized Volatility')
    plt.legend()
    plt.show()

    # Calculate and print some statistics (adjusted for bi-weekly forecasting)
    final_prices = simulated_paths[-1, :]
    print(f"Forecasted price statistics after 26 weeks:")
    print(f"Mean: {np.mean(final_prices):.2f}")
    print(f"Median: {np.median(final_prices):.2f}")
    print(f"5th percentile: {np.percentile(final_prices, 5):.2f}")
    print(f"95th percentile: {np.percentile(final_prices, 95):.2f}")

if __name__ == "__main__":
    directory = "C:/Users/markc/PythonAI/EquityAnalysis"  # Replace with your directory path
    filename = "ASTS.csv"  # Replace with your CSV filename
    main(directory, filename)