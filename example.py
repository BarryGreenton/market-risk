import numpy as np
import pandas as pd

def calculate_entropy(df):
    df_norm = df / df.sum(axis=0)
    entropy = -np.nansum(df_norm * np.log(df_norm + 1e-09), axis=0) / np.log(len(df))
    return entropy


def calculate_weights(entropy):
    redundancy = 1 - entropy
    weights = redundancy / redundancy.sum()
    return weights


def calculate_composite_score(df, weights):
    composite_score = np.dot(df, weights)
    return composite_score

if __name__ == "__main__":
    panel = pd.DataFrame({"turnover": [0., 20., 100.], "reversal": [25., 0., 100.], "AR": [100., 75., 0.]})
    weights = calculate_weights(calculate_entropy(panel))
    assert np.isclose(weights.sum(), 1)
    print(calculate_composite_score(panel, weights))
