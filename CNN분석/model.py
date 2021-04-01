import pandas as pd
from keras.models import Sequential, Model
from keras.layers import LSTM, Input, RepeatVector
import keras
from keras.layers import Dense, Activation, BatchNormalization, Dropout, Conv1D, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import glorot_uniform
from keras.layers import Dropout
import keras.backend as K
from keras.activations import relu, sigmoid, tanh

# dnn모델


def dnn_model(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation(tanh))

    model.add(Dense(256, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation(tanh))

    model.add(Dense(256, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation(tanh))

    model.add(Dense(128, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation(tanh))

    model.add(Dense(64, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation(tanh))

    model.add(Dense(32, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation(tanh))

    model.add(Dense(1, kernel_initializer='glorot_uniform'))
    model.add(Activation(sigmoid))
    return model


# CNN모델
def cnn_model(input_dim):
    # Designing the CNN Model
    model = Sequential()

    # 필터수(20), 필터크기(1,4) (세로,가로)
    model.add(
        Conv2D(20, (1, 4), input_shape=x_train[0].shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(30, (1, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))

    model.add(Dropout(0.2))  # 데이터를 20% 비율로 0으로 설정함(Overfitting 방지)
    model.add(Flatten())  # 모든 필터가 생성한 Feature Map을 한개의 배열에 저장하여 전 결합층에 전달
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # n_classes:분류수,
    # softmax : 전체 클래스에 해당할 비율을 출력함
    return model


# CNN모델
def cnn_model2(input_dim):
    # Designing the CNN Model
    model = Sequential()

    # 필터수(20), 필터크기(1,1,1) (세로,가로)
    model.add(
        Conv3D(16, (1, 1, 7), input_shape=x_train[0].shape, activation='relu'))
    model.add(AveragePooling3D(pool_size=(1, 1, 2)))
    model.add(Conv3D(32, (1, 1, 1), activation='relu'))
    model.add(AveragePooling3D(pool_size=(1, 1, 2)))
    model.add(Conv3D(64, (1, 1, 1), activation='relu'))
    model.add(AveragePooling3D(pool_size=(1, 1, 2)))
    model.add(Conv3D(128, (1, 1, 1), activation='relu'))
    model.add(AveragePooling3D(pool_size=(1, 1, 2)))

    model.add(Flatten())  # 모든 필터가 생성한 Feature Map을 한개의 배열에 저장하여 전 결합층에 전달
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # n_classes:분류수,
    # softmax : 전체 클래스에 해당할 비율을 출력함
    return model
