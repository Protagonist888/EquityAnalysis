import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load equity data from a CSV file.

    Args:
    file_path (str): Path to the CSV file containing equity data.

    Returns:
    pd.DataFrame or None: DataFrame containing the equity data if successful, None otherwise.

    Usage:
    - Ensure your CSV file has columns: Date, Open, High, Low, Close, Adj Close, Volume.
    - Date should be in a format recognizable by pandas (e.g., YYYY-MM-DD).
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df = df.sort_values('Date')
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain columns: {', '.join(required_columns)}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_returns(df):
    """
    Calculate daily returns based on adjusted close prices.

    Args:
    df (pd.DataFrame): DataFrame containing equity data with 'Adj Close' column.

    Returns:
    pd.Series: Daily returns.

    Usage:
    - This function assumes the input DataFrame has an 'Adj Close' column.
    - To use a different price column, modify the column name in the function.
    """
    if 'Adj Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Adj Close' column")
    return df['Adj Close'].pct_change().dropna()

def calculate_var_cvar(returns, confidence_level):
    """
    Calculate Value at Risk (VaR) and Conditional Value at Risk (cVaR) using historical simulation method.

    Args:
    returns (pd.Series): Daily returns.
    confidence_level (float): Confidence level for VaR and cVaR calculation (e.g., 0.95 for 95%).

    Returns:
    tuple: (VaR, cVaR)

    Usage:
    - Adjust confidence_level to change the risk measure (e.g., 0.99 for 99% confidence).
    - VaR and cVaR are returned as positive values representing losses.
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

    Usage:
    - Define scenarios in the main function. Each scenario can include:
      * 'shock': A fixed percentage to add to all returns (e.g., -0.10 for a 10% drop)
      * 'volatility_multiplier': A factor to multiply returns by (e.g., 2 to double volatility)
      * 'tail_event': A dictionary with 'duration' (number of days) and 'return' (daily return during the event)
    - Customize scenarios to model specific market conditions or hypothetical events.
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
    """
    Plot stress test results comparing original and stressed VaR and cVaR.

    Args:
    original_var (float): Original VaR.
    original_cvar (float): Original cVaR.
    stress_results (dict): Dictionary containing stress test results.

    Returns:
    None (displays plot)

    Usage:
    - This function creates a bar plot comparing original and stressed risk measures.
    - Customize colors, labels, or layout by modifying the plotting code.
    """
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
    file_path = 'equity_data.csv'  # Update this path as needed
    df = load_data(file_path)
    if df is None:
        return
    
    try:
        returns = calculate_returns(df)
    except ValueError as e:
        print(f"Error calculating returns: {e}")
        return
    
    # Calculate original VaR and cVaR
    confidence_level = 0.95  # Adjust this value to change the confidence level
    try:
        original_var, original_cvar = calculate_var_cvar(returns, confidence_level)
        print(f"Original VaR ({confidence_level*100}%): {original_var:.4f}")
        print(f"Original cVaR ({confidence_level*100}%): {original_cvar:.4f}")
    except ValueError as e:
        print(f"Error calculating VaR and cVaR: {e}")
        return
    
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
        print(f"  VaR ({confidence_level*100}%): {result['VaR']:.4f}")
        print(f"  cVaR ({confidence_level*100}%): {result['cVaR']:.4f}")
    
    # Plot stress test results
    plot_stress_test_results(original_var, original_cvar, stress_results)

if __name__ == "__main__":
    main()

# Suggestions for improvement to better capture volatility and manage risk:

# 1. Implement GARCH models:
#    - Use GARCH models to capture volatility clustering and time-varying volatility.
#    - This can provide more accurate volatility forecasts, especially during periods of market stress.

# 2. Incorporate Monte Carlo simulation:
#    - Use Monte Carlo methods to generate a large number of possible price paths.
#    - This can provide a more comprehensive view of potential outcomes and tail risks.

# 3. Add Rolling Window Analysis:
#    - Implement a rolling window approach for VaR and cVaR calculations.
#    - This can help capture changing market dynamics and provide a more dynamic risk assessment.

# 4. Include Extreme Value Theory (EVT):
#    - Apply EVT techniques to model tail risks more accurately.
#    - This can improve the estimation of low-probability, high-impact events.

# 5. Implement Expected Shortfall (ES):
#    - Calculate Expected Shortfall in addition to VaR and cVaR.
#    - ES is considered more coherent and provides a better measure of tail risk.

# 6. Add Scenario Generation:
#    - Develop more sophisticated methods for generating stress scenarios.
#    - Consider using historical crisis periods or expert judgement to create more realistic scenarios.

# 7. Incorporate Factor Models:
#    - Implement factor models to capture systematic risk factors.
#    - This can provide insights into the sources of portfolio risk and improve risk decomposition.

# 8. Add Liquidity Risk Modeling:
#    - Incorporate liquidity risk into the VaR and stress testing framework.
#    - This can account for the potential impact of market liquidity on risk measures.

# 9. Implement Backtesting:
#    - Add rigorous backtesting procedures to validate the VaR model.
#    - This can help assess the accuracy of the risk measures and refine the model.

# 10. Consider Regime-Switching Models:
#     - Implement regime-switching models to capture different market states.
#     - This can improve risk estimation across varying market conditions.