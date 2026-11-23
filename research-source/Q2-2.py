import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load the data
df = pd.read_csv('scores.csv')
df = df[['trade_date','score']]

# ADF检验 + 差分调整
def ADF_test(data,count=0):
    adf_result = adfuller(data)
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])
    if adf_result[1] < 0.05:
        print('时间序列是平稳的')
    else:
        print('时间序列是非平稳的')
        count = count + 1
        print('进行了%s阶差分后的结果如下' % count)
        data = data.diff(1).dropna()
        ADF_test(data,count)
    return data

df['score'] = ADF_test(df['score'])

# 网格搜索
def grid_search_arima(df, p_values, d_values, q_values):
    best_aic = np.inf  # 初始化最小AIC
    best_order = None  # 初始化最优参数
    best_model = None  # 初始化最优模型
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    # 拟合 ARIMA 模型
                    model = ARIMA(df['score'], order=(p, d, q))
                    fitted_model = model.fit()
                    
                    # 如果当前模型的 AIC 更小，更新最优模型
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                        best_model = fitted_model
                except Exception as e:
                    print(f"Error fitting ARIMA({p},{d},{q}): {e}")
                    continue
                    
    return best_model, best_order, best_aic

p_values = range(0, 5)  # 选择 AR 部分阶数（例如：0 到 4）
d_values = range(0, 3)  # 选择差分阶数（例如：0 到 2）
q_values = range(0, 5)  # 选择 MA 部分阶数（例如：0 到 4）

best_model, best_order, best_aic = grid_search_arima(df['score'], p_values, d_values, q_values)

# 输出最佳模型的参数和 AIC 值
print(f"Best ARIMA Model: {best_order}")
print(f"Best AIC: {best_aic}")
print(best_model.summary())
    
arima_model = ARIMA(df['score'], order=(3, 2, 4))
arima_fitted = arima_model.fit()
print(arima_fitted.summary())

# 预测未来5天的得分
forecast_horizon = 5
arima_forecast = arima_fitted.forecast(steps=forecast_horizon)
print("ARIMA模型预测的未来得分：", arima_forecast)

# 使用模型对测试集进行预测（假设数据按80%/20%分割）
train_size = int(len(df) * 0.8)
train, test = df['score'][:train_size], df['score'][train_size:]

# ARIMA在测试集上的预测
arima_pred = arima_fitted.predict(start=test.index[0], end=test.index[-1])
mse = mean_squared_error(test, arima_pred)
rmse = mse ** 0.5
mape = mean_absolute_error(test, arima_pred) / test.mean() * 100

print("ARIMA模型 MSE:", mse)
print("ARIMA模型 RMSE:", rmse)
print("ARIMA模型 MAPE:", mape)

df['trade_date'] = pd.to_datetime(df['trade_date'])
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


plt.figure(figsize=(24, 6))
plt.plot(df['score'], label='真实值')
plt.plot(arima_pred, label='ARIMA预测值', linestyle='--')
plt.title('ARIMA模型预测效果')
plt.ylabel('风险得分')
plt


plt.grid()
plt.margins(x=0)
plt.savefig('Q2-3.png')