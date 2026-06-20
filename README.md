This project develops a SARIMAX-ML hybrid model for forecasting the USD/KRW exchange rate
using monthly macroeconomic variables from January 2010 to April 2026 as exogenous predictors.

Following the Box-Jenkins methodology, data transformations were first applied and the model
structure was identified based on the autocorrelation function and partial autocorrelation function of
each variable. A grid search based on the corrected Akaike Information Criterion selected the
SARIMAX(1,0,1)(0,0,0)_
12 model as the final specification. Residual diagnostic tests indicated that no
significant autocorrelation remained in the residuals and that the normality assumption was
reasonably satisfied. However, the SARIMAX-only model failed to outperform the random walk
benchmark.

To address the nonlinear patterns not captured by the SARIMAX-only model, a two-step hybrid
framework was constructed by training machine learning models on the in-sample residuals of the
SARIMAX model. Specifically, three machine learning models - LGBM, SVR, and GPR - were
employed for residual learning. The results show that the SARIMAX-SVR model achieved the best
forecasting performance, recording a test RMSE of 36.76 and a directional accuracy of 71.79%,
thereby consistently outperforming the random walk benchmark.
