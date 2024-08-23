# Measure correlation between volatility and price over two week windows


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the stock price data from a CSV file
file_path = 'C:/Users/markc/PythonAI/EquityAnalysis/ASTS.csv'
stock_data = pd.read_csv(file_path)
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data = stock_data.sort_values('Date')

# 2. Function to calculate standard deviation for a given window
def calculate_std(data, window):
    return data['Close'].rolling(window=window).std()

# 3. Function to calculate future returns
def calculate_future_returns(data, days):
    return data['Close'].pct_change(periods=days).shift(-days)

# 4. Calculate standard deviations and future returns
for window in range(3, 15):  # Extended to 14 days
    stock_data[f'std_{window}d'] = calculate_std(stock_data, window)

for days in range(1, 15):  # Extended to 14 days
    stock_data[f'future_return_{days}d'] = calculate_future_returns(stock_data, days)

# 5. Calculate correlations
correlation_matrix = pd.DataFrame(index=range(3, 15), columns=range(1, 15))

for window in range(3, 15):
    for days in range(1, 15):
        correlation = stock_data[f'std_{window}d'].corr(stock_data[f'future_return_{days}d'])
        correlation_matrix.loc[window, days] = correlation

# Convert correlation matrix to float
correlation_matrix = correlation_matrix.astype(float)

# 6. Print correlation table
print("Correlation Table:")
print(correlation_matrix)

# 7. Create heatmap
plt.figure(figsize=(16, 12))  # Increased figure size for better readability
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation between Price Volatility and Future Returns (2-week window)')
plt.xlabel('Days Ahead')
plt.ylabel('Volatility Window (Days)')
plt.tight_layout()  # Adjust the plot to ensure all elements are visible
plt.show()

# 8. Find the highest correlation
max_corr = correlation_matrix.max().max()
max_corr_location = correlation_matrix.stack().idxmax()

print(f"\nHighest correlation: {max_corr:.4f}")
print(f"Found between {max_corr_location[0]}-day volatility and {max_corr_location[1]}-day future return")

# 9. Additional analysis: Average correlation for each volatility window
avg_correlations = correlation_matrix.mean(axis=1)
print("\nAverage correlation for each volatility window:")
print(avg_correlations)

# 10. Plot average correlations
plt.figure(figsize=(10, 6))
avg_correlations.plot(kind='bar')
plt.title('Average Correlation by Volatility Window')
plt.xlabel('Volatility Window (Days)')
plt.ylabel('Average Correlation')
plt.tight_layout()
plt.show()


