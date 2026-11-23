import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# 上证指数
df = pd.read_csv('data/0000001.csv')
df = df.rename(columns={'pct_chg':'index_chg'})

# 国债利率
rate = pd.read_csv('BND_Exchange.csv')
rate.columns = ['trade_date','rate','drate','mrate']
rate['trade_date'] = pd.to_datetime(rate['trade_date'])

# 沪深300
index = pd.read_csv('../data/indexSH.csv')
index['trade_date'] = pd.to_datetime(index['trade_date'],format='%Y%m%d')
index = index.sort_values('trade_date')
index = index[['trade_date','close','pct_chg']]
index = pd.merge(index, df[['trade_date','index_chg']], on='trade_date', how='left')

# 任务2计算的beta
beta = pd.read_csv('../Q2/data/result2.csv')
beta['trade_date'] = pd.to_datetime(beta['trade_date'])
index = pd.merge(index, beta[['trade_date','weight_beta']], on='trade_date', how='left')
index.ffill(inplace=True)

index['return'] = index['pct_chg']
index = index.drop(columns=['pct_chg'])
index = pd.merge(index, rate[['trade_date','drate']], on='trade_date', how='left')

index['expected_return'] = rate['drate'] + index['weight_beta'] * (index['index_chg'] - rate['drate'])
index['excess_return'] = index['return'] - index['expected_return']

mean_return = index['expected_return'].mean() 
std_dev_return = index['expected_return'].std()
z = 1.96  
index['lower_bound'] = mean_return - z * std_dev_return
index['upper_bound'] = mean_return + z * std_dev_return
print("每日预期收益率的均值为:", mean_return)
print("预期收益率的95%置信区间为: [", index['lower_bound'].iloc[0], ",", index['upper_bound'].iloc[0], "]")

# 绘制预期收益率与实际收益率的对比图
plt.figure(figsize=(12, 6))
plt.plot(index['trade_date'], index['expected_return'], label='Expected Return')
plt.plot(index['trade_date'], index['return'], label='Actual Return')
plt.legend()
plt.savefig('Q4-1.png')

# 绘制预期收益率与实际收益率的残差图
plt.figure(figsize=(12, 6))
plt.plot(index['trade_date'], index['expected_return'] - index['return'], label='Residual')
plt.legend()
plt.savefig('Q4-2.png')