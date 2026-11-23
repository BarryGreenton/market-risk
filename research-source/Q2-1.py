import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the data
data_1 = pd.read_csv('../Q2/data/result.csv')
data_1 = data_1[['trade_date','weight_turnover','weight_beta','weight_LNCAP']]
print(data_1.shape)

data_2 = pd.read_csv('../Q1/result.csv')
print(data_2.shape)

data = pd.merge(data_1, data_2, on='trade_date', how='inner')
data['weight_return'] = data['weight_return']
data['weight_reverse'] = data['weight_reverse']
data['weight_amihud'] = data['weight_amihud']
print(data.shape)

# 选取特征
F = data[['weight_turnover','weight_beta','weight_reverse','weight_LNCAP','weight_return','weight_amihud','weight_AR','weight_BR']]

X = F.copy()

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = ['turnover','beta', 'reverse', 'LNCAP', 'return','amihud','AR','BR']) 

# PCA
pca = PCA()
p = pca.fit(X)

# 方差贡献率
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

# 根据方差贡献率选择主成分个数
threshold = 0.8
n = next(i for i, x in enumerate(np.cumsum(explained_variance)) if x > threshold)
print(n)

# 计算主成分得分
scores = pca.transform(X)[:, :n+1]
scores_df = pd.DataFrame(scores, columns=['PC{}'.format(i+1) for i in range(n+1)])
print(scores_df)

# 根据各主成分的方差贡献分配权重，计算综合得分
weights = pca.components_[:n+1]
composite_score = np.dot(scores, weights)
composite_score_df = pd.DataFrame(composite_score, columns=['composite_score'])
print(composite_score_df)

# 综合得分与原始指标的相关性
correlation = np.corrcoef(X.T, composite_score)
correlation_df = pd.DataFrame(correlation[-1, :-1], index=X.columns, columns=['correlation_with_composite_score'])
print(correlation_df)

############################################

# 根据part 1的结果。选取turnover,reverse,AR作为特征
scores_df = data.loc[:, ['trade_date', 'weight_turnover','weight_reverse', 'weight_AR']]
scores_df.dropna(inplace=True)

# 对各指标进行正向化处理并归一化到【0，100】
scores_df['weight_turnover'] = (scores_df['weight_turnover'] - scores_df['weight_turnover'].min()) / (scores_df['weight_turnover'].max() - scores_df['weight_turnover'].min()) * 100
scores_df['weight_reverse'] = (scores_df['weight_reverse'] - scores_df['weight_reverse'].min()) / (scores_df['weight_reverse'].max() - scores_df['weight_reverse'].min()) * 100
scores_df['weight_AR'] = (scores_df['weight_AR'] - scores_df['weight_AR'].min()) / (scores_df['weight_AR'].max() - scores_df['weight_AR'].min()) * 100

# 熵权法计算权重
x = scores_df[['weight_turnover','weight_reverse','weight_AR']]

# 计算熵值
def calculate_entropy(df):
    # 归一化处理
    df_norm = df / df.sum(axis=0)
    
    # 计算熵值
    entropy = -np.nansum(df_norm * np.log(df_norm + 1e-9), axis=0) / np.log(len(df))
    
    return entropy

# 计算权重
def calculate_weights(entropy):
    redundancy = 1 - entropy
    weights = redundancy / redundancy.sum()
    return weights

# 计算综合得分
def calculate_composite_score(df, weights):
    composite_score = np.dot(df, weights)
    return composite_score

entropy = calculate_entropy(x)
weights = calculate_weights(entropy)
composite_score = calculate_composite_score(x, weights)
scores_df['score'] = composite_score
print(scores_df)
print(weights)

# 保存结果
scores_df.to_csv('result.csv', index=False)
