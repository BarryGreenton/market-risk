import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

def ADF_test(data, count=0):
    adf_result = adfuller(data)
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    if adf_result[1] < 0.05:
        print('Stationarity test rejected the unit-root null.')
    else:
        print('Stationarity test did not reject the unit-root null.')
        count = count + 1
        print('Applying difference of order %s' % count)
        data = data.diff(1).dropna()
        return ADF_test(data, count)
    return data


def grid_search_arima(df, p_values, d_values, q_values):
    best_aic = np.inf
    best_order = None
    best_model = None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(df['score'], order=(p, d, q))
                    fitted_model = model.fit()
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                        best_model = fitted_model
                except Exception as e:
                    print(f'Error fitting ARIMA({p},{d},{q}): {e}')
                    continue
    return (best_model, best_order, best_aic)
