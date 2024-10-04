# Residuals in Statistical Modeling

## Definition
Residuals are the differences between the observed values and the values predicted by a model.

## Formula
Residual = Observed value - Predicted value

## Importance
- Measure the unexplained variation in a model
- Help assess model fit
- Useful for detecting patterns the model hasn't captured

## Properties of Good Residuals
1. Zero mean
2. Constant variance (homoscedasticity)
3. No autocorrelation
4. Normal distribution (for many statistical tests)

## Analyzing Residuals
- Visual inspection:
  - Residual plots
  - Q-Q plots for normality
- Statistical tests:
  - Durbin-Watson test for autocorrelation
  - Breusch-Pagan test for heteroscedasticity
  - Shapiro-Wilk test for normality

## In GARCH Models
- GARCH models work with standardized residuals
- Standardized residuals = Residuals / Estimated standard deviation
- Should be approximately white noise if the model is well-specified

Remember: Analyzing residuals is crucial for validating model assumptions and identifying areas for model improvement.
