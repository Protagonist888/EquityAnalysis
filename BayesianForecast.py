# Using Bayesian inference to predict next stock price move.
# Belief is that something that's happened in the recent past will incluence the near future.
# Next steps - play with variance, add volume, stock metadata, play with prior and posterior distributions


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Load the stock price data from a CSV file
file_path = 'C:/Users/markc/PythonAI/EquityAnalysis/ASTS.csv'  # Replace with your actual file path
stock_data = pd.read_csv(file_path)

# Assuming the CSV has columns: 'Date', 'Open', 'High', 'Low', 'Close'
# Ensure the Date column is parsed as datetime if it's included
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# 2. Calculate variance based on the past week (High - Low)
# Assuming we're focusing on the last week (last 7 days)
recent_data = stock_data.tail(7)
variances = recent_data['High'] - recent_data['Low']
mean_variance = np.mean(variances)
print(f"Mean variance over the past week: {mean_variance:.2f}")

# 3. Set up the prior distribution for the future price movement
# Assume a prior belief that the future price movement has a mean close to the last known close price
prior_mean = stock_data['Close'].iloc[-1]
prior_std = np.std(stock_data['Close'])
print(f"Prior mean: {prior_mean:.2f}, Prior std: {prior_std:.2f}")

# Define the prior distribution
prior = norm(loc=prior_mean, scale=prior_std)

# 4. Define the likelihood function based on the recent data variance
# Assume the likelihood of future price movement follows a normal distribution
# centered around the recent close price with variance estimated from the past week
likelihood_std = np.sqrt(mean_variance)
likelihood = norm(loc=prior_mean, scale=likelihood_std)

# 5. Compute the posterior distribution
# Since prior and likelihood are both normal, the posterior is also normal.
posterior_mean = (prior_mean/prior_std**2 + prior_mean/likelihood_std**2) / (1/prior_std**2 + 1/likelihood_std**2)
posterior_std = np.sqrt(1 / (1/prior_std**2 + 1/likelihood_std**2))
posterior = norm(loc=posterior_mean, scale=posterior_std)

print(f"Posterior mean: {posterior_mean:.2f}, Posterior std: {posterior_std:.2f}")

# 6. Plot the prior, likelihood, and posterior distributions
x = np.linspace(prior_mean - 3*prior_std, prior_mean + 3*prior_std, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x, prior.pdf(x), 'r-', label=f'Prior (mean={prior_mean:.2f}, std={prior_std:.2f})')
plt.plot(x, likelihood.pdf(x), 'g-', label=f'Likelihood (mean={prior_mean:.2f}, std={likelihood_std:.2f})')
plt.plot(x, posterior.pdf(x), 'b-', label=f'Posterior (mean={posterior_mean:.2f}, std={posterior_std:.2f})')
plt.fill_between(x, 0, posterior.pdf(x), color='blue', alpha=0.1)
plt.title('Bayesian Inference for Future Stock Price Movement')
plt.xlabel('Price Movement')
plt.ylabel('Density')
plt.legend()
plt.show()
