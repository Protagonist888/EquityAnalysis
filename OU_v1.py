#Ornstein-Uhlenbeck Process
# Hypothesis: used to estimate path of stock prices




import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

def read_csv_data(directory, filename):
    """
    Read closing price data from a CSV file.
    
    :param directory: Directory containing the CSV file
    :param filename: Name of the CSV file
    :return: numpy array of closing prices
    """
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path)
    if 'Close' not in df.columns:
        raise ValueError("CSV file must contain a 'Close' column")
    return df['Close'].values

def calculate_returns(prices):
    """Calculate log returns from price data."""
    return np.log(prices[1:] / prices[:-1])

def estimate_ou_params(returns, dt=1):
    # Estimate μ (mu) - long-term mean
    mu = np.mean(returns)
    
    # Estimate σ (sigma) - volatility
    sigma = np.std(returns)
    
    # Estimate θ (theta) - rate of mean reversion
    def ou_loglikelihood(theta):
        ou_returns = theta * (mu - returns[:-1]) * dt + sigma * np.sqrt(dt) * np.random.normal(size=len(returns)-1)
        return -np.sum(np.log(np.abs(ou_returns - returns[1:])))
    
    result = minimize(ou_loglikelihood, x0=[0.1], method='L-BFGS-B', bounds=[(0, None)])
    theta = result.x[0]
    
    return mu, sigma, theta

def simulate_ou_process(X0, mu, sigma, theta, T, dt, num_paths=1):
    num_steps = int(T / dt)
    X = np.zeros((num_paths, num_steps + 1))
    X[:, 0] = X0
    
    for t in range(1, num_steps + 1):
        dW = np.random.normal(0, np.sqrt(dt), size=num_paths)
        dX = theta * (mu - X[:, t-1]) * dt + sigma * dW
        X[:, t] = X[:, t-1] + dX
    
    return X

def main(directory, filename):
    # Read data from CSV
    prices = read_csv_data(directory, filename)
    
    # Calculate returns
    returns = calculate_returns(prices)
    
    # Estimate parameters
    dt = 1/252  # assuming daily data
    est_mu, est_sigma, est_theta = estimate_ou_params(returns, dt)
    
    print(f"Estimated parameters: μ = {est_mu:.4f}, σ = {est_sigma:.4f}, θ = {est_theta:.4f}")
    
    # Simulate future paths
    num_paths = 1000
    future_T = 1  # simulate for 1 year
    X_future = simulate_ou_process(np.log(prices[-1]), est_mu, est_sigma, est_theta, future_T, dt, num_paths)
    
    # Convert log prices back to actual prices
    prices_future = np.exp(X_future)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label='Historical Data')
    plt.plot(np.arange(len(prices), len(prices) + X_future.shape[1]), prices_future.T, alpha=0.1, color='gray')
    plt.plot(np.arange(len(prices), len(prices) + X_future.shape[1]), np.mean(prices_future, axis=0), color='red', label='Mean Forecast')
    plt.axhline(np.exp(est_mu), color='green', linestyle='--', label='Estimated Long-term Mean')
    plt.legend()
    plt.title('Ornstein-Uhlenbeck Process: Historical Data and Simulated Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.show()

if __name__ == "__main__":
    directory = "C:/Users/mchung/Personal/EquityAnalysis"  # Replace with your directory path
    filename = "ASTS.csv"  # Replace with your CSV filename
    main(directory, filename)