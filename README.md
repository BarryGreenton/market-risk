# A risk score for the Chinese equity market

How can trading activity, liquidity and returns be combined into a daily measure of market risk?

For the 2024 Greater Bay Area financial modeling competition, the team examined eight indicators across four dimensions. Five principal components explained over 80% of the standardized variation. Turnover, short-term reversal and the AR sentiment indicator were retained for the entropy-weighted score. The study then compared forecasts and explored allocation across equity, bonds and cash.

The report gives weights of 0.7884, 0.0699 and 0.1417 for the three retained indicators. The public example extracts the weighting functions from the competition submission and demonstrates them on a small illustrative panel.

## Run the example

From this project directory, using Python 3.12 in a virtual environment:

```sh
python -m pip install -r requirements.txt
python example.py
```

Two additional modules preserve the competition submission’s [ARIMA model selection](forecast.py) and [historical VaR and allocation](allocation.py) routines. They use `statsmodels` and `scipy`, respectively.

In the English adaptation the ADF routine returns the final differenced series from its recursive call.

## Original research code

The five submitted scripts correspond to the competition questions. Q1 constructs indicators, Q2 combines dimension reduction with ARIMA forecasting, Q3 simulates allocation with a VaR constraint, and Q4 compares CAPM-implied and observed returns. The scripts use the original workbook or CSV layouts and column names.

[Browse the research source](research-source/) · [GitHub profile](https://github.com/BarryGreenton)
