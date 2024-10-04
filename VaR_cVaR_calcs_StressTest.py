# Takes VaR_cVaR_calcs.py to next level with stress testing

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load equity data from a CSV file."""
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df = df.sort_values('Date')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_returns(df):
    """Calculate daily returns based on adjusted close prices."""
    return df['Adj Close'].pct_change().dropna()

def calculate_var_cvar(returns, confidence_level):
    """Calculate VaR and cVaR using historical simulation method."""
    var = returns.quantile(1 - confidence_level)
    cvar = returns[returns <= var].mean()
    return -var, -cvar

def stress_test_scenarios(returns, scenarios):
    """
    Perform stress testing on returns based on given scenarios.
    
    Args:
    returns (pd.Series): Daily returns
    scenarios (dict): Dictionary of stress test scenarios
    
    Returns:
    dict: Stress test results
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

def plot_stress_test_results(original_var, original_cvar, stress_results):
    """Plot stress test results."""
    scenarios = list(stress_results.keys())
    vars = [result['VaR'] for result in stress_results.values()]
    cvars = [result['cVaR'] for result in stress_results.values()]

    x = range(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, vars, width, label='VaR')
    ax.bar([i + width for i in x], cvars, width, label='cVaR')

    ax.axhline(y=original_var, color='r', linestyle='--', label='Original VaR')
    ax.axhline(y=original_cvar, color='g', linestyle='--', label='Original cVaR')

    ax.set_ylabel('Value')
    ax.set_title('Stress Test Results')
    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()

def main():
    # Load and prepare data
    df = load_data('ASTS.csv')
    if df is None:
        return
    
    returns = calculate_returns(df)
    
    # Calculate original VaR and cVaR
    original_var, original_cvar = calculate_var_cvar(returns, 0.95)
    print(f"Original VaR (95%): {original_var:.4f}")
    print(f"Original cVaR (95%): {original_cvar:.4f}")
    
    # Define stress test scenarios
    scenarios = {
        'Market Crash': {'shock': -0.10},  # 10% market drop
        'High Volatility': {'volatility_multiplier': 2},  # Double volatility
        'Prolonged Decline': {'tail_event': {'duration': 30, 'return': -0.02}},  # 2% daily loss for 30 days
        'Mixed Scenario': {'shock': -0.05, 'volatility_multiplier': 1.5}  # 5% drop and 50% increased volatility
    }
    
    # Perform stress tests
    stress_results = stress_test_scenarios(returns, scenarios)
    
    # Print stress test results
    print("\nStress Test Results:")
    for scenario, result in stress_results.items():
        print(f"{scenario}:")
        print(f"  VaR (95%): {result['VaR']:.4f}")
        print(f"  cVaR (95%): {result['cVaR']:.4f}")
    
    # Plot stress test results
    plot_stress_test_results(original_var, original_cvar, stress_results)

if __name__ == "__main__":
    main()