import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

def load_data(file_path):
    """
    Load equity data from a CSV file.

    Args:
    file_path (str): Path to the CSV file containing equity data.

    Returns:
    pd.DataFrame or None: DataFrame containing the equity data if successful, None otherwise.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df = df.sort_values('Date')
        required_columns = ['Date', 'Adj Close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {', '.join(required_columns)}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def calculate_returns(df, lookback_days=365):
    """
    Calculate daily returns based on adjusted close prices for the past year or entire period, whichever is shorter.

    Args:
    df (pd.DataFrame): DataFrame containing equity data with 'Date' and 'Adj Close' columns.
    lookback_days (int): Number of days to look back for calculating returns.

    Returns:
    pd.Series: Daily returns.
    """
    end_date = df['Date'].max()
    start_date = max(df['Date'].min(), end_date - timedelta(days=lookback_days))
    df_subset = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    return df_subset['Adj Close'].pct_change().dropna()

def calculate_var_cvar(returns, confidence_level):
    """
    Calculate Value at Risk (VaR) and Conditional Value at Risk (cVaR) using historical simulation method.

    Args:
    returns (pd.Series): Daily returns.
    confidence_level (float): Confidence level for VaR and cVaR calculation (e.g., 0.95 for 95%).

    Returns:
    tuple: (VaR, cVaR)
    """
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")
    var = returns.quantile(1 - confidence_level)
    cvar = returns[returns <= var].mean()
    return -var, -cvar

def stress_test_scenarios(returns, scenarios):
    """
    Perform stress testing on returns based on given scenarios.

    Args:
    returns (pd.Series): Daily returns.
    scenarios (dict): Dictionary of stress test scenarios.

    Returns:
    dict: Stress test results containing VaR and cVaR for each scenario.
    """
    results = {}
    for scenario_name, scenario_params in scenarios.items():
        stressed_returns = returns.copy()
        
        if 'shock' in scenario_params:
            stressed_returns += scenario_params['shock']
        
        if 'volatility_multiplier' in scenario_params:
            stressed_returns *= scenario_params['volatility_multiplier']
        
        if 'tail_event' in scenario_params:
            tail_event = scenario_params['tail_event']
            stressed_returns.iloc[-tail_event['duration']:] = tail_event['return']
        
        var, cvar = calculate_var_cvar(stressed_returns, 0.95)
        results[scenario_name] = {'VaR': var, 'cVaR': cvar}
    
    return results

def analyze_ticker(file_path, scenarios):
    """
    Analyze a single ticker from a CSV file.

    Args:
    file_path (str): Path to the CSV file.
    scenarios (dict): Dictionary of stress test scenarios.

    Returns:
    dict: Analysis results for the ticker.
    """
    df = load_data(file_path)
    if df is None:
        return None
    
    returns = calculate_returns(df)
    if returns.empty:
        print(f"No valid returns data for {file_path}")
        return None
    
    var, cvar = calculate_var_cvar(returns, 0.95)
    stress_results = stress_test_scenarios(returns, scenarios)
    
    return {
        'returns': returns,
        'VaR': var,
        'cVaR': cvar,
        'stress_results': stress_results
    }

def analyze_portfolio(ticker_results):
    """
    Analyze the portfolio as a whole.

    Args:
    ticker_results (dict): Dictionary of analysis results for each ticker.

    Returns:
    dict: Analysis results for the portfolio.
    """
    # Combine returns with equal weights
    all_returns = pd.DataFrame({ticker: data['returns'] for ticker, data in ticker_results.items()})
    portfolio_returns = all_returns.mean(axis=1)
    
    var, cvar = calculate_var_cvar(portfolio_returns, 0.95)
    stress_results = stress_test_scenarios(portfolio_returns, scenarios)
    
    return {
        'returns': portfolio_returns,
        'VaR': var,
        'cVaR': cvar,
        'stress_results': stress_results
    }

def generate_report(ticker_results, portfolio_results):
    """
    Generate a comprehensive report of the analysis results.

    Args:
    ticker_results (dict): Dictionary of analysis results for each ticker.
    portfolio_results (dict): Analysis results for the portfolio.

    Returns:
    str: Formatted report string.
    """
    report = "Multi-Ticker Portfolio Risk Analysis Report\n"
    report += "="*50 + "\n\n"
    
    report += "Portfolio Summary:\n"
    report += f"VaR (95%): {portfolio_results['VaR']:.4f}\n"
    report += f"cVaR (95%): {portfolio_results['cVaR']:.4f}\n\n"
    
    report += "Portfolio Stress Test Results:\n"
    for scenario, result in portfolio_results['stress_results'].items():
        report += f"  {scenario}:\n"
        report += f"    VaR (95%): {result['VaR']:.4f}\n"
        report += f"    cVaR (95%): {result['cVaR']:.4f}\n"
    report += "\n"
    
    report += "Individual Ticker Analysis:\n"
    for ticker, results in ticker_results.items():
        report += f"{ticker}:\n"
        report += f"  VaR (95%): {results['VaR']:.4f}\n"
        report += f"  cVaR (95%): {results['cVaR']:.4f}\n"
        report += "  Stress Test Results:\n"
        for scenario, result in results['stress_results'].items():
            report += f"    {scenario}:\n"
            report += f"      VaR (95%): {result['VaR']:.4f}\n"
            report += f"      cVaR (95%): {result['cVaR']:.4f}\n"
        report += "\n"
    
    return report

def main():
    # Define stress test scenarios
    global scenarios
    scenarios = {
        'Market Crash': {'shock': -0.10},
        'High Volatility': {'volatility_multiplier': 2},
        'Prolonged Decline': {'tail_event': {'duration': 30, 'return': -0.02}},
        'Mixed Scenario': {'shock': -0.05, 'volatility_multiplier': 1.5}
    }
    
    # Process all CSV files in the current directory
    ticker_results = {}
    for file_name in os.listdir('.'):
        if file_name.endswith('.csv'):
            ticker = os.path.splitext(file_name)[0]
            results = analyze_ticker(file_name, scenarios)
            if results:
                ticker_results[ticker] = results
    
    if not ticker_results:
        print("No valid CSV files found for analysis.")
        return
    
    # Analyze portfolio
    portfolio_results = analyze_portfolio(ticker_results)
    
    # Generate and save report
    report = generate_report(ticker_results, portfolio_results)
    with open('portfolio_risk_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("Analysis complete. Report saved as 'portfolio_risk_analysis_report.txt'")

if __name__ == "__main__":
    main()