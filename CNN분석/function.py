# 데이터 기본정보 확인(최소,4분위,평균,최대,결측치 개수)
def basic(data):
    X = data
    basic_box = [['null']*6 for _ in range(len(X.iloc[0, :]))]
    for i in range(len(X.iloc[0, :])):
        basic_box[i][0] = X.iloc[:, i].min()
        basic_box[i][1] = X.iloc[:, i].quantile(.25)
        basic_box[i][2] = X.iloc[:, i].mean()
        basic_box[i][3] = X.iloc[:, i].quantile(.75)
        basic_box[i][4] = X.iloc[:, i].max()
        basic_box[i][5] = X.iloc[:, i].isna().sum()

    return basic_box


# sigmoid 후 이진분류 할때
def classi(_):
    y_pred = _
    for i in range(len(y_pred)):
        if y_pred[i] < 0.5:
            y_pred[i] = 0
        else:
            y_pred[i] = 1

    return y_pred
