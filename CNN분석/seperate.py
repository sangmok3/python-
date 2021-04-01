import pandas as pd
import numpy as np

# data X,Y분류


def X_sep(data):
    X = data.drop([data.columns[0], data.columns[-1]], axis=1)
    return X


def Y_sep(data):
    Y = data.iloc[:, -1]
    return Y


# cnn형태로 데이터 변환 (데이터, 컬럼 나누는 개수)
def cnn_sep(data, k):
    d1 = data.loc[:, data.columns[0:k]]
    d2 = data.loc[:, data.columns[0+k:k+k]]
    d3 = data.loc[:, data.columns[0+(k*2):k+(k*2)]]
    d4 = data.loc[:, data.columns[0+(k*3):k+(k*3)]]
    d5 = data.loc[:, data.columns[0+(k*4):k+(k*4)]]
    d6 = data.loc[:, data.columns[0+(k*5):k+(k*5)]]
    d7 = data.loc[:, data.columns[0+(k*6):k+(k*6)]]

    dd1 = d1.values
    dd2 = d2.values
    dd3 = d3.values
    dd4 = d4.values
    dd5 = d5.values
    dd6 = d6.values
    dd7 = d7.values

    ddd1 = dd1.reshape(len(data), 1, k)
    ddd2 = dd2.reshape(len(data), 1, k)
    ddd3 = dd3.reshape(len(data), 1, k)
    ddd4 = dd4.reshape(len(data), 1, k)
    ddd5 = dd5.reshape(len(data), 1, k)
    ddd6 = dd6.reshape(len(data), 1, k)
    ddd7 = dd7.reshape(len(data), 1, k)

    df = np.array([np.transpose(i)
                   for i in zip(ddd1, ddd2, ddd3, ddd4, ddd5, ddd6, ddd7)])
    df = df.reshape(len(data), 1, k, int(len(data.columns)/k))

    return df
