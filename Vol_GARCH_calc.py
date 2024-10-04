# This script walks through steps to calculate vol using GARCH and VIX prices as input
'''
A few notes and potential next steps:
-The forecast accuracy measure uses squared returns as a proxy for realized volatility. There are more sophisticated measures of realized volatility that could be implemented for a more accurate comparison.
-You might want to consider implementing a rolling window forecast to get a more robust measure of forecast accuracy over time.
-The script assumes that your VIX.csv file is in the same directory. Make sure this is the case when running the script.
-- Add sentiment score, book depth (e.g. bi/ask spreads,volume, options implied volatility, regime-switching models) as another variable
--  
'''

import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Step 1: Data Preparation
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    df['returns'] = df['Adj Close'].pct_change()
    
    # Remove NaN and inf values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Visualization of raw data and returns
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(df['Adj Close'])
    ax1.set_title('VIX Adjusted Close Price')
    ax2.plot(df['returns'])
    ax2.set_title('VIX Returns')
    plt.tight_layout()
    plt.show()
    
    return df


# Step 2: Check for Stationarity
def check_stationarity(series):
    result = adfuller(series.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    
    # Visualization of stationarity
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    ax1.plot(series)
    ax1.set_title('Original Series')
    plot_acf(series, ax=ax2)
    plot_pacf(series, ax=ax3)
    plt.tight_layout()
    plt.show()

# Step 3: Fit GARCH model
def fit_garch_model(returns):
    model = arch_model(returns, vol='Garch', p=1, q=1)
    results = model.fit(disp='off')
    return results

# Step 4: Diagnostic Checking
def diagnostic_checks(model_results, returns):
    std_resid = model_results.resid / model_results.conditional_volatility
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
    # ACF and PACF of standardized residuals
    plot_acf(std_resid, ax=ax1)
    ax1.set_title('ACF of Standardized Residuals')
    plot_pacf(std_resid, ax=ax2)
    ax2.set_title('PACF of Standardized Residuals')
    
    # QQ plot
    ax3.scatter(np.random.normal(0, 1, len(std_resid)), np.sort(std_resid))
    ax3.plot(ax3.get_xlim(), ax3.get_xlim(), ls="--", c=".3")
    ax3.set_title('Q-Q plot of Standardized Residuals')
    
    # Histogram of standardized residuals
    sns.histplot(std_resid, kde=True, ax=ax4)
    ax4.set_title('Histogram of Standardized Residuals')
    
    plt.tight_layout()
    plt.show()

# Step 5: Volatility Forecasting
def forecast_volatility(model_results, forecast_horizon=14):
    forecast = model_results.forecast(horizon=forecast_horizon)
    return forecast.variance.iloc[-1]



# Step 7: Forecast Accuracy
def forecast_accuracy(true_values, forecasts):
    mse = mean_squared_error(true_values, forecasts)
    rmse = np.sqrt(mse)
    return {'MSE': mse, 'RMSE': rmse}

if __name__ == "__main__":
    # Load and prepare data
    df = load_and_prepare_data('VIX.csv')
    
    # Check if we have enough data after cleaning
    if len(df) < 30:  # Arbitrary threshold, adjust as needed
        print("Not enough data points after cleaning. Please check your input data.")
    else:
        # Check stationarity
        print("\nChecking stationarity of returns:")
        check_stationarity(df['returns'])
        
        # Fit GARCH(1,1) model
        garch_results = fit_garch_model(df['returns'])
        print("\nGARCH(1,1) Model Summary:")
        print(garch_results.summary())
        
        # Perform diagnostic checks
        print("\nPerforming diagnostic checks...")
        diagnostic_checks(garch_results, df['returns'])
        
        # Forecast volatility
        forecast_horizon = 14  # Forecasting for the next 14 days
        garch_forecast = forecast_volatility(garch_results, forecast_horizon)
        
        # Plot forecast
        plt.figure(figsize=(12, 6))
        plt.plot(garch_forecast, label='GARCH(1,1)')
        plt.title(f'Volatility Forecast for the Next {forecast_horizon} Days')
        plt.xlabel('Days')
        plt.ylabel('Forecasted Variance')
        plt.legend()
        plt.show()
        
        # Calculate forecast accuracy
        # We'll use the last 14 days of our data as "true" values
        true_volatility = df['returns'][-forecast_horizon:]**2  # Squared returns as proxy for realized volatility
        garch_accuracy = forecast_accuracy(true_volatility, garch_forecast)
        
        print("\nForecast Accuracy:")
        print("GARCH(1,1):", garch_accuracy)