# VaR_cVaR_calcs
# Sample script for daily historical equity datas
import pandas as pd
import numpy as np
from scipy import stats
import os

def load_data(file_path):
    """
    Load equity data from a CSV file.
    
    Args:
    file_path (str): Path to the CSV file
    
    Returns:
    pd.DataFrame: DataFrame containing the equity data
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df = df.sort_values('Date')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_returns(df):
    """
    Calculate daily returns based on adjusted close prices.
    
    Args:
    df (pd.DataFrame): DataFrame containing equity data
    
    Returns:
    pd.Series: Daily returns
    """
    return df['Adj Close'].pct_change().dropna()

def historical_var_cvar(returns, confidence_level):
    """
    Calculate VaR and cVaR using historical simulation method.
    
    Args:
    returns (pd.Series): Daily returns
    confidence_level (float): Confidence level (e.g., 0.95 for 95%)
    
    Returns:
    tuple: (VaR, cVaR)
    """
    var = returns.quantile(1 - confidence_level)
    cvar = returns[returns <= var].mean()
    return -var, -cvar

def parametric_var_cvar(returns, confidence_level):
    """
    Calculate VaR and cVaR using parametric variance-covariance method.
    
    Args:
    returns (pd.Series): Daily returns
    confidence_level (float): Confidence level (e.g., 0.95 for 95%)
    
    Returns:
    tuple: (VaR, cVaR)
    """
    mu = returns.mean()
    sigma = returns.std()
    z_score = stats.norm.ppf(1 - confidence_level)
    var = -(mu + sigma * z_score)
    cvar = -(mu + sigma * stats.norm.pdf(z_score) / (1 - confidence_level))
    return var, cvar

def main():
    # Set the file path
    file_name = "ASTS.csv"  # Update this with your specific file name
    file_path = os.path.join(os.getcwd(), file_name)
    
    # Load data
    df = load_data(file_path)
    if df is None:
        return
    
    # Calculate returns
    returns = calculate_returns(df)
    
    # Set confidence level and time horizon
    confidence_level = 0.95
    time_horizon = 252  # Assuming 252 trading days in a year
    
    # Annualize returns for 1-year time horizon
    annual_returns = returns * np.sqrt(time_horizon)
    
    # Calculate VaR and cVaR using both methods
    hist_var, hist_cvar = historical_var_cvar(annual_returns, confidence_level)
    param_var, param_cvar = parametric_var_cvar(annual_returns, confidence_level)
    
    # Print results
    print(f"Results for 95% confidence level, 1-year time horizon:")
    print(f"\nHistorical Simulation Method:")
    print(f"VaR: {hist_var:.4f}")
    print(f"cVaR: {hist_cvar:.4f}")
    print(f"\nParametric Variance-Covariance Method:")
    print(f"VaR: {param_var:.4f}")
    print(f"cVaR: {param_cvar:.4f}")

if __name__ == "__main__":
    main()

# Areas for improvement and further exploration:
# 1. Implement Monte Carlo simulation for VaR and cVaR calculation
# 2. Add functionality to calculate VaR and cVaR for different time horizons
# 3. Incorporate volatility clustering using GARCH models
# 4. Implement bootstrapping for more robust cVaR estimation
# 5. Add visualization of return distribution and VaR/cVaR
# 6. Implement stress testing scenarios
# 7. Add option to calculate VaR and cVaR for a portfolio of assets
# 8. Implement backtesting to validate VaR model accuracy
# 9. Add error handling for missing or incorrect data in the CSV file
# 10. Implement rolling window analysis for dynamic VaR and cVaR estimation