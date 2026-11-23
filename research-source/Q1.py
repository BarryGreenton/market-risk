import pandas as pd
import numpy as np
import time

def get_pre_date(date, n):
    trade_cal = pro.trade_cal(exchange='SSE', start_date='20130101', end_date=date, is_open=1)
    trade_cal['cal_date'] = pd.to_datetime(trade_cal['cal_date'])
    trade_cal = trade_cal.sort_values(by='cal_date', ascending=False)
    
    pre_date = trade_cal.iloc[n]['cal_date']
    
    return pre_date.strftime('%Y%m%d')


def target(ts_code,start_date,end_date):
    start_date_26 = get_pre_date(start_date, 30) 
    
    daily_data = pro.daily(ts_code=ts_code, start_date=start_date_26, end_date=end_date)
    
    if daily_data.empty:
        print(f"No data for {ts_code} from {start_date_26} to {end_date}")
        return pd.DataFrame()  # 返回空的DataFrame

    daily_data = daily_data.sort_values(by='trade_date', ascending=True)
    
    daily_data['return'] = daily_data['close'].pct_change()
    
    daily_data['log_return'] = np.log(daily_data['close']/daily_data['close'].shift(1)) 
    daily_data['reverse'] = daily_data['log_return'].rolling(window=30, min_periods=1).mean()
    daily_data['damihud'] = np.abs(daily_data['log_return'])*1000000 / daily_data['amount'] 
    daily_data['amihud'] = daily_data['damihud'].rolling(window=20).mean() 
    daily_data['amihud'] = daily_data['amihud'].fillna(0)
    
    daily_data['AR_num'] = daily_data['high'] - daily_data['open']
    daily_data['AR_den'] = daily_data['open'] - daily_data['low']
    daily_data['AR'] = daily_data['AR_num'].rolling(window=26).sum() / daily_data['AR_den'].rolling(window=26).sum()*100
    daily_data['BR_num'] = daily_data['high'] - daily_data['close'].shift(1) #
    daily_data['BR_den'] = daily_data['close'].shift(1) - daily_data['low']
    daily_data['BR'] = daily_data['BR_num'].rolling(window=26).sum() / daily_data['BR_den'].rolling(window=26).sum()*100
    

    # 保留指定日期的数据
    daily_data = daily_data[daily_data['trade_date'] >= start_date]
    daily_data = daily_data[daily_data['trade_date'] <= end_date]
    
    return daily_data


basic_data = []
result_data = []

for i in range(2014,2025):
    w_df = pd.read_csv('re_data/hs300_'+str(i)+'.csv')
    w_df = w_df[['con_code', 'weight']]
    w_df['weight'] = w_df['weight'] / 100
    w_df = w_df.set_index('con_code')
    con_codes = w_df.index 
    start = str(i)+'0101'
    end = str(i)+'1231'
    
    all_data = []
    j = 0
    for code in con_codes:
        daily_ex = target(code, start, end)
        all_data.append(daily_ex)
        time.sleep(0.2)
    
    all_data = pd.concat(all_data, ignore_index=True)
    basic_data.append(all_data)
       
    all_data = all_data[['ts_code', 'trade_date', 'return','log_return','amihud', 'AR', 'BR', 'reverse']]
    all_data.sort_values(by=['ts_code', 'trade_date'], ignore_index=True)
    merge_df = pd.merge(all_data, w_df, left_on='ts_code', right_index=True)
    
    merge_df['weight_return'] = merge_df['return'] * merge_df['weight']*100
    merge_df['weight_amihud'] = merge_df['amihud'] * merge_df['weight']
    merge_df['weight_AR'] = merge_df['AR'] * merge_df['weight']
    merge_df['weight_BR'] = merge_df['BR'] * merge_df['weight']
    merge_df['weight_reverse'] = merge_df['reverse'] * merge_df['weight']
    
    daily_return = merge_df.groupby('trade_date')['weight_return'].sum().reset_index()
    daily_amihud = merge_df.groupby('trade_date')['weight_amihud'].sum().reset_index()
    daily_AR = merge_df.groupby('trade_date')['weight_AR'].sum().reset_index()
    daily_BR = merge_df.groupby('trade_date')['weight_BR'].sum().reset_index()
    daily_reverse = merge_df.groupby('trade_date')['weight_reverse'].sum().reset_index()

    result = pd.merge(daily_return, daily_amihud, on='trade_date')
    result = pd.merge(result, daily_AR, on='trade_date')
    result = pd.merge(result, daily_BR, on='trade_date')
    result = pd.merge(result, daily_reverse, on='trade_date')
    
    
    result_data.append(result)
    print(f"Save {i} successfully!")
    
basic_data = pd.concat(basic_data, ignore_index=True)
result_data = pd.concat(result_data, ignore_index=True)
basic_data.to_csv('Q1-2/basic_data.csv', index=False)
result_data.to_csv('Q1-2/result.csv', index=False)

