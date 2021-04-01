import pandas as pd
import numpy as np
import os
import seperate as sp
import function as fc
import model as md
import cutoff as cut

from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from keras.callbacks import EarlyStopping
from keras.models import load_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Conv3D, MaxPooling3D, AveragePooling3D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


train = pd.read_csv(os.getcwd()[0:44]+'data/train_usr.csv')
test = pd.read_csv(os.getcwd()[0:44]+'data/detect_usr.csv')

x_train = sp.X_sep(train)
y_train = sp.Y_sep(train)
x_test = sp.X_sep(test)
y_test = sp.Y_sep(test)


# 3Ddata shape create

x_train = sp.cnn_sep(x_train, 50)  # sp.cnn_sep(데이터,나눌 컬럼 개수)
x_test = sp.cnn_sep(x_test, 50)

# CNN돌릴때 categorical이면 데이터를 카테고리로 변환시켜야 함, categorical = [0,1]로 구분
targets_train = tf.keras.utils.to_categorical(y_train).astype(np.integer)
targets_test = tf.keras.utils.to_categorical(y_test).astype(np.integer)


x_train = x_train.reshape((14113, 1, 1, 50, 7))
x_test = x_test.reshape((4210, 1, 1, 50, 7))

# _cnn모델 새로 학습할 때
model = md.cnn_model2(x_train.shape[0])  # 모델 불러와서 학습시킬때
# model.save('cnn_model_2.h5') #모델 저장할 때
# _Compile model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(x_train, targets_train, epochs=100, batch_size=20,
          validation_split=0.2, callbacks=[es])
# _overfitting방지를 위해 dropout 20%적용

re_tg = model.predict(x_test)
tg = pd.DataFrame(y_test, columns=['0', '1'])
re_tgs = pd.DataFrame(re_tg, columns=['0확률', 'cut_off'])
y_pred = re_tgs['cut_off']
result = pd.concat([y_pred, y_test], axis=1)
cut.cut_off(result)
