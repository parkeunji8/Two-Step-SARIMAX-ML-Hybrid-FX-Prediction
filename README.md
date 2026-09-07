# USD/KRW Exchange Rate Forecasting: Two Step SARIMAX-ML Hybrid Fx Prediction

## Overview

This project develops a SARIMAX-ML hybrid model for forecasting the USD/KRW exchange rate.
Monthly macroeconomic variables are used as exogenous predictors, and a two-step hybrid framework
is proposed to capture nonlinear patterns that a linear time series model alone cannot explain.

## Data

- **Target variable**: USD/KRW exchange rate (monthly)
- **Period**: January 2010 – April 2026
- **Exogenous variables**: Monthly macroeconomic indicators (e.g., CPI, interest rates, current
  account balance, VIX, DXY, WTI, USD/CNY)
- **Sources**: Yahoo Finance (`yfinance`) for market data; Finaeon for macroeconomic indicators


## Methodology

The modeling process follows the Box-Jenkins methodology:

1. **Data transformation** — Stationarity was induced via log-differencing and other
   transformations, guided by the ACF/PACF of each variable.
2. **Model identification** — A grid search over SARIMAX orders, evaluated using the corrected
   Akaike Information Criterion (AICc), selected **SARIMAX(1,0,1)(0,0,0)₁₂** as the final
   specification.
3. **Residual diagnostics** — Diagnostic tests confirmed no significant autocorrelation remained
   in the residuals, and the normality assumption was reasonably satisfied. However, the
   SARIMAX-only model failed to outperform the random walk benchmark.
4. **Hybrid residual correction** — To capture nonlinear patterns not explained by SARIMAX, a
   two-step hybrid framework was built: three machine learning models — **LGBM**, **SVR**,
   **GPR** — were trained on the in-sample residuals of the SARIMAX model to learn the remaining
   structure.

## Results

The SARIMAX-SVR hybrid achieved the best forecasting performance among the three residual
learners, consistently outperforming the random walk benchmark.

| Model | Test RMSE | Directional Accuracy |
|---|---|---|
| Random Walk (benchmark) | 37.65 | 50% (theoretical) |
| SARIMAX only | 39.35 | 48.72 |
| **SARIMAX + SVR** | **36.76** | **71.79%** |
| SARIMAX + LGBM | 39.99 | 51.28% |
| SARIMAX + GPR | 39.70 | 48.72% |


## Repository Structure

    ├── notebooks/          # model experimentation
    ├── models/
    │   ├── final_model_svr.py  # Final SARIMAX + SVR pipeline
    │   ├── final_model_gpr.py  # Final SARIMAX + GPR pipeline
    │   ├── final_model_lgbm.py  # Final SARIMAX + LGBM pipeline
    ├── results/
    │   ├── figures/        # EDA and final result visualizations
    ├── requirements.txt
    └── README.md
