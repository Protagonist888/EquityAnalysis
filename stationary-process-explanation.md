# Stationary Process in Time Series Analysis

## Definition
A stationary process is a stochastic (random) process whose statistical properties do not change over time.

## Key Characteristics
1. Constant mean
2. Constant variance
3. Constant autocovariance (for any given lag)

## Importance
- Allows for meaningful statistical inference
- Enables forecasting
- Simplifies mathematical modeling

## Types of Stationarity
1. Strictly stationary: All statistical properties are constant over time
2. Weakly stationary (or covariance stationary): Mean, variance, and autocovariance are constant over time

## Testing for Stationarity
- Visual inspection (plotting the series)
- Statistical tests:
  - Augmented Dickey-Fuller (ADF) test
  - KPSS test
  - Phillips-Perron test

## Dealing with Non-Stationary Data
- Differencing
- Detrending
- Seasonal adjustment

## Examples
- Stationary: White noise process
- Non-stationary: Random walk, time series with trend or seasonality

Remember: Most statistical models, including GARCH, assume stationarity. Ensuring your data is stationary is a crucial step in time series analysis.
