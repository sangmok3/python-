import sys
import pandas as pd
#import crypto as cp
import models as md
import functions as fc
import logging
import logging.config
import json

# train, test 데이터 정보 가져오기(데이터 경로, 형태, 결측치확인, info,데이터 head, 타겟 칼럼)


def extractLogger(name):
    jsonConfig = json.load(open("./properties/logger.json"))
    logging.config.dictConfig(jsonConfig)
    logger = logging.getLogger(name)
    return logger


def getInfo(config):
    # read data
    trainPath = config.get("data", "trainPath")
    testPath = config.get("data", "testPath")
    print(trainPath)
    print(testPath)
    targetColumn = config.get("data", "targetColumn")
    df_train = pd.read_csv(trainPath)
    df_test = pd.read_csv(testPath)
    print("Data shape")
    print(df_train.shape)
    print("Null Check")
    print(df_train.isnull().sum())
    print("Data Set Information")
    print(df_train.info())
    print("Sample data")
    print(df_train.head())
    print("Normal and Target Data Info")
    print(df_train[targetColumn].value_counts())
    print(df_train[targetColumn].value_counts(normalize=True))
    print("----------------------------------")
    print("Data shape")
    print(df_test.shape)
    print("Null Check")
    print(df_test.isnull().sum())
    print("Data Set Information")
    print(df_test.info())
    print("Sample data")
    print(df_test.head())
    print("Normal and Target Data Info")
    print(df_test[targetColumn].value_counts())
    print(df_test[targetColumn].value_counts(normalize=True))

# 모델정보 입력 후 불러오기


def sortOption(option, config):
    if option == 'info':
        fc.getInfo(config)
    elif option == 'rfc':
        md.rfc(config)
    elif option == 'rfr':
        md.rfr(config)
    elif option == 'xgb':
        md.xgb(config)
    elif option == 'svm':
        md.svm(config)
    elif option == 'lrc':
        md.lrc(config)
    elif option == 'dnn':
        md.dnn(config)
    elif option == 'encrypt':
        try:
            cp.load_key()
        except:
            cp.write_key()
        print(cp.encrypt(""))
    elif option == 'decrypt':
        print(cp.decrypt(""))
    elif option == 'sim':
        md.simulation(config)
    else:
        raise ValueError(f'invalid argument {argv}')
