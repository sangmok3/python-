from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from decimal import Decimal
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pickle
import os
from collections import Counter
import dbConnector as dc
import numpy as np
import pandas as pd
import logging
import logging.config
import json
import timeit
import ast

jsonConfig = json.load(open("./properties/logger.json"))
logging.config.dictConfig(jsonConfig)
logger = logging.getLogger("__main__")


def generateParam(params):
    result = []
    parameters = params.split(",")
    for param in parameters:
        if '.' in param:
            result.append(np.float16(param))
        else:
            result.append(int(param))
    return result


# 칼럼drop함수
def dropColumnData(df, dropColumns):
    actualDropColumns = []
    X = df
    for dropColumn in dropColumns:
        if dropColumn in df:
            actualDropColumns.append(dropColumn)
    for actualDropColumn in actualDropColumns:
        X = X.drop(labels=[actualDropColumn], axis=1)
    return X


# Train, Test데이터 칼럼 drop
def Preprocessor(df_train, df_test, dropColumns, targetColumn):
    ''' 
    drop and train-test split
    '''
    # Drop useless columns
    X_train = dropColumnData(df_train, dropColumns)
    X_test = dropColumnData(df_test, dropColumns)

    # Generate label
    y_train = df_train.loc[:, targetColumn]
    y_test = df_test.loc[:, targetColumn]

    return X_train, X_test, y_train, y_test


# 데이터 비대칭 문제 해결을 위한 함수(현재모델에 사용 안함-결과 안좋음)
def Sampler(X_train, y_train, method='smotetomek', njobs=-1):
    '''
    We sample data so that we can train to classify data better when handling imbalanced data.
    There are plenty of methods to sample, but this function suggests 4 methods.
    -------------------------
    method of sampling (str)
    -------------------------
    'none' : not applied
    'under' : Random Under Sampling
    'over' : SMOTE
    'both' : Ramdom Under Sampling + SMOTE
    'smotetomek' : SMOTETomek
    '''
    randomState = 42

    if method == 'under':
        # Undersampling
        # Under-samples majority class
        rus = RandomUnderSampler(
            random_state=randomState, sampling_strategy=0.04)
        sampler = Pipeline([("under", rus)])
    elif method == 'over':
        # Oversampling
        # Over-samples minority class
        # This methods oversamples the data but the oversampled data is NOT always minority class.
        # Oversampled data classified through classifier in the method.
        smote = SMOTE(random_state=randomState, sampling_strategy=0.0405)
        sampler = Pipeline([("over", smote)])
    elif method == 'both':
        ## Undersampling + Oversampling
        rus = RandomUnderSampler(
            random_state=randomState, sampling_strategy=0.04)
        smote = SMOTE(random_state=randomState, sampling_strategy=0.0405)
        sampler = Pipeline([("under", rus), ("over", smote)])
    elif method == 'smotetomek':
        # Combination method
        # This method is combination of SMOTE and Tomek's link.
        # Tomek's link removes the majority class after finding the pair of different classes (Undersampling).
        smotetomek = SMOTETomek(random_state=randomState,
                                sampling_strategy=0.0410)
        sampler = Pipeline([("smotetomek", smotetomek)])
    else:
        pass

    if method != 'none':
        X_train, y_train = sampler.fit_resample(X_train, y_train)

    return X_train, y_train


# 데이터 스케일러(현재 안씀-데이터 스케일 결과 차이없음 & 옵션으로 조정)
def Scaler(X_train, X_test, scaleMethod):
    '''
    We scale data in case the variables are very diffrent unit to each other.
    If the scaler did not be applied, training loss could be high.
    '''
    # In case scaler did not be applied
    if scaleMethod == 'none':
        scaler = 'none'
        logger.info("The scaler did not be applied")

    # In case scaler is applied
    else:
        scaleMethod = scaleMethod.lower()
        if scaleMethod == 'standard':
            scaler = StandardScaler()
        elif scaleMethod == 'minmax':
            scaler = MinMaxScaler()
        elif scaleMethod == 'robust':
            scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        logger.info("X_train data: {0}".format(X_train))
        logger.info("X_test data: {0}".format(X_test))

    return X_train, X_test, scaler


# 비율 구하는 함수
def VariableGenerator(data, useYN=True):
    '''
    Generate variables using original dataset.
    Add columns to insert below useYN if you want.
    '''
    if useYN:
        data['TEMP31'] = data.TEMP1 / (data.TEMP1 + data.TEMP2)  # 결제 성공률
        data['TEMP32'] = data.TEMP1 / data.TEMP10  # 구매 시도 비 결제 성공률
        data['TEMP33'] = (data.TEMP2 + data.TEMP4 + data.TEMP14 + data.TEMP15 + data.TEMP16 +
                          data.TEMP17 + data.TEMP18 + data.TEMP19 + data.TEMP20) / data.TEMP10  # 의심 탐지 건수
        data['TEMP34'] = (data.TEMP22 + data.TEMP24 +
                          data.TEMP26 + data.TEMP28) / data.TEMP1  # V-Bucks 구매비율
        data['TEMP35'] = (data.TEMP23 + data.TEMP25 + data.TEMP27 +
                          data.TEMP29) / data.TEMP10  # V-Bucks 구매 실패 비율
        data['TEMP36'] = data.TEMP1 / data.TEMP5  # 결제일 대비 결제 횟수
        data['TEMP37'] = data.TEMP11 / data.TEMP10  # 카드 결제 시도 건수 대비 구매 시도
        data['TEMP38'] = (data.TEMP14 + data.TEMP15 +
                          data.TEMP16 + data.TEMP17) / data.TEMP10  # PG사 에러 비율
        data['TEMP39'] = (data.TEMP5 - data.TEMP6) / data.TEMP5  # 1일 이내 다결제 비율
        data = data.fillna(0)

    return data


# 데이터 결과 금액구하는 함수
def simulation(config):
    # Paths & Saving options
    testData = config.get("data", "testData")
    simulationPath = config.get("simulation", "simulationPath")
    modelFile = config.get("data", "fileName")
    modelPath = config.get("simulation", "modelPath")
    dropColumns = config.get("simulation", "dropColumn").split(",")
    keyColumn = config.get("simulation", "keyColumn")
    predictMode = config.get("simulation", "predictMode")
    scoreYN = config.get("simulation", "scoreYN")
    scaleMethod = config.get("data", "scaleMethod")

    # Load data
    df = pd.read_csv(os.path.join(simulationPath, testData))
    keys = df[keyColumn].values
    X_test = dropColumnData(df, dropColumns)

    # Generate variables
    # X_test = VariableGenerator(X_test)

    # Load model & scaler
    model = pickle.load(open(os.path.join(modelPath, modelFile+".sav"), "rb"))
    if scaleMethod != 'none':
        scaler = pickle.load(
            open(os.path.join(modelPath, modelFile+"_scaler.pkl"), "rb"))
        X_test = scaler.transform(X_test)

    # Produce scores
    if predictMode == "cutoff":
        # Produce predicted __probabilites__
        pred = pd.DataFrame(model.predict_proba(X_test))
        # Charge-back prediction probabilities remained
        detect = pred[1]

        if scoreYN == "Y":
            # insert USR_ID, BAT_DTTM, SCORE, MODEL_NM
            dc.insertScoreResult(keys, detect, modelFile)
        else:
            # insert KEYID, MODEL, SCORE
            dc.insertModelResult1(keys, detect, modelFile)
    else:
        # Produce predicted __classes__
        pred = model.predict(X_test)
        detect = np.where(pred == 1)
        result = np.take(keys, detect)
        for row in result[0]:
            # insert KEYID, MODEL
            dc.insertModelResult2(row, modelFile)

# rfc모델


def rfc(config):
    start = timeit.default_timer()

    # Paths & Saving options
    trainPath = config.get("data", "trainPath") + \
        config.get("data", "trainData")
    testPath = config.get("data", "testPath") + config.get("data", "testData")
    dropColumns = config.get("data", "dropColumn").split(",")
    targetColumn = config.get("data", "targetColumn")
    keyColumn = config.get("data", "keyColumn")
    savingPath = config.get("data", "savingPath")
    scaleMethod = config.get("data", "scaleMethod")
    modelFile = config.get("data", "fileName")

    # Model options
    nJobs = int(config.get("RandomForestClassifier", "n_jobs"))
    randomState = int(config.get("RandomForestClassifier", "random_state"))
    cvVal = int(config.get("RandomForestClassifier", "cv"))
    verboseVal = int(config.get("RandomForestClassifier", "verbose"))
    nEstimators = config.get("RandomForestClassifier", "n_estimators")
    maxDepth = config.get("RandomForestClassifier", "max_depth")
    minSamplesLeaf = config.get("RandomForestClassifier", "min_samples_leaf")
    minSampleSplit = config.get("RandomForestClassifier", "min_samples_split")

    logger.info("Training Model: RandomForestClassifier")
    logger.info("Model Name: {0}".format(modelFile))
    logger.info("Scale Method: {0}".format(scaleMethod))

    # Load data
    df_train = pd.read_csv(trainPath)
    df_test = pd.read_csv(testPath)

    # Split data
    X_train, X_test, y_train, y_test = Preprocessor(
        df_train, df_test, dropColumns, targetColumn)
    cols = X_train.columns.tolist()
    del df_train, df_test

    logger.info(savingPath)
    logger.info("Train data infomation")
    logger.info(X_train.shape)
    logger.info(Counter(y_train))
    logger.info("Original columns to train")
    logger.info(cols)
    logger.info("Test data infomation")
    logger.info(X_test.shape)
    logger.info(Counter(y_test))

    # Generate variables
    X_train = VariableGenerator(X_train)
    X_test = VariableGenerator(X_test)
    cols = X_train.columns.tolist()

    # Scale data
    X_train, X_test, scaler = Scaler(X_train, X_test, scaleMethod)

    # Sampling
    X_train, y_train = Sampler(X_train, y_train)

    logger.info("Generated columns to train")
    logger.info(cols)
    logger.info("Sampled train data infomation")
    logger.info(X_train.shape)
    logger.info(Counter(y_train))

    # Prevent view warnings
    if scaleMethod == "none":
        X_train.is_copy = False
        X_test.is_copy = False

    # Training parameters and Grid-search to search best model
    params = {'n_estimators': generateParam(nEstimators),
              'max_depth': generateParam(maxDepth),
              'min_samples_leaf': generateParam(minSamplesLeaf),
              'min_samples_split': generateParam(minSampleSplit)
              }
    rf_clf = RandomForestClassifier(random_state=randomState, n_jobs=nJobs)
    grid_cv = GridSearchCV(rf_clf, param_grid=params,
                           cv=cvVal, n_jobs=nJobs, verbose=verboseVal)
    grid_cv.fit(X_train, y_train)
    bestParam = grid_cv.best_params_
    logger.info("%s %s", '최적 하이퍼 파라미터: ', bestParam)
    logger.info("%s %s", '최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))
    grid_cv_df = pd.DataFrame(grid_cv.cv_results_)
    grid_cv_df.sort_values(by=['rank_test_score'], inplace=True)
    logger.info(
        grid_cv_df[['params', 'mean_test_score', 'rank_test_score']].head(10))

    # The best hyper-parameter model
    rfcModel = RandomForestClassifier(n_estimators=bestParam.get("n_estimators"),
                                      max_depth=bestParam.get("max_depth"),
                                      min_samples_leaf=bestParam.get(
                                          "min_samples_leaf"),
                                      min_samples_split=bestParam.get(
                                          "min_samples_split"),
                                      random_state=randomState,
                                      n_jobs=nJobs)
    rfcModel.fit(X_train, y_train)
    pred = rfcModel.predict(X_test)

    # Test results
    logger.info("%s %s", 'accuracy', metrics.accuracy_score(y_test, pred))
    logger.info("%s %s", 'precision', metrics.precision_score(y_test, pred))
    logger.info("%s %s", 'recall', metrics.recall_score(y_test, pred))
    logger.info("%s %s", 'f1', metrics.f1_score(y_test, pred))
    logger.info(metrics.classification_report(y_test, pred))
    logger.info(metrics.confusion_matrix(y_test, pred))

    # Saving model products
    # Model
    pickle.dump(rfcModel, open(os.path.join(
        savingPath, modelFile+".sav"), 'wb'))

    # Scaler
    if scaleMethod != 'none':
        pickle.dump(scaler, open(os.path.join(
            savingPath, modelFile+"_scaler.pkl"), "wb"))

    # Feature importances
    imp = pd.DataFrame(
        data={'columns': cols, 'RFC': list(rfcModel.feature_importances_)})
    imp.to_csv(os.path.join(savingPath, modelFile+"_imp.csv"), index=False)

    stop = timeit.default_timer()
    logger.info("수행시간", (stop - start))


# XGB모델
def xgb(config):
    start = timeit.default_timer()

    # Paths & Saving options
    trainPath = config.get("data", "trainPath") + \
        config.get("data", "trainData")
    testPath = config.get("data", "testPath") + config.get("data", "testData")
    dropColumns = config.get("data", "dropColumn").split(",")
    targetColumn = config.get("data", "targetColumn")
    keyColumn = config.get("data", "keyColumn")
    savingPath = config.get("data", "savingPath")
    scaleMethod = config.get("data", "scaleMethod")
    modelFile = config.get("data", "fileName")

    # Model options
    nJobs = int(config.get("XGBClassifier", "n_jobs"))
    testSize = float(config.get("XGBClassifier", "test_size"))
    randomState = int(config.get("XGBClassifier", "random_state"))
    cvVal = int(config.get("XGBClassifier", "cv"))
    verboseVal = int(config.get("XGBClassifier", "verbose"))
    maxDepth = config.get("XGBClassifier", "max_depth")
    learning_rate = config.get("XGBClassifier", "learning_rate")
    nEstimators = config.get("XGBClassifier", "n_estimators")

    logger.info("Training Model: XGBClassifier")
    logger.info("Model Name: {0}".format(modelFile))
    logger.info("Scale Method: {0}".format(scaleMethod))

    # Load data
    df_train = pd.read_csv(trainPath)
    df_test = pd.read_csv(testPath)

    # Split data
    X_train, X_test, y_train, y_test = Preprocessor(
        df_train, df_test, dropColumns, targetColumn)
    cols = X_train.columns.tolist()
    del df_train, df_test

    logger.info(savingPath)
    logger.info("Train data infomation")
    logger.info(X_train.shape)
    logger.info(Counter(y_train))
    logger.info("Original columns to train")
    logger.info(cols)
    logger.info("Test data infomation")
    logger.info(X_test.shape)
    logger.info(Counter(y_test))

    # Generate variables
    X_train = VariableGenerator(X_train)
    X_test = VariableGenerator(X_test)
    cols = X_train.columns.tolist()

    # Scale data
    X_train, X_test, scaler = Scaler(X_train, X_test, scaleMethod)

    # Sampling
    X_train, y_train = Sampler(X_train, y_train)

    logger.info("Generated columns to train")
    logger.info(cols)
    logger.info("Sampled train data infomation")
    logger.info(X_train.shape)
    logger.info(Counter(y_train))

    # Prevent view warnings
    if scaleMethod == "none":
        X_train.is_copy = False
        y_train.is_copy = False

    # Training parameters and Grid-search to search best model
    params = {'learning_rate': generateParam(learning_rate),
              'max_depth': generateParam(maxDepth),
              'n_estimators': generateParam(nEstimators)
              }
    xgb_clf = XGBClassifier(random_state=randomState, n_jobs=nJobs)
    grid_cv = GridSearchCV(xgb_clf, param_grid=params,
                           cv=cvVal, n_jobs=nJobs, verbose=verboseVal)
    grid_cv.fit(X_train, y_train,
                #            eval_set = [(X_val, y_val)],
                #            early_stopping_rounds=500,
                verbose=verboseVal)
    bestParam = grid_cv.best_params_
    logger.info("%s %s", '최적 하이퍼 파라미터: ', bestParam)
    logger.info("%s %s", '최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))
    grid_cv_df = pd.DataFrame(grid_cv.cv_results_)
    grid_cv_df.sort_values(by=['rank_test_score'], inplace=True)
    logger.info(
        grid_cv_df[['params', 'mean_test_score', 'rank_test_score']].head(10))

    # The best hyper-parameter model
    XGBModel = XGBClassifier(n_estimators=bestParam.get("n_estimators"),
                             max_depth=bestParam.get("max_depth"),
                             learning_rate=bestParam.get("learning_rate"),
                             min_samples_split=bestParam.get(
                                 "min_samples_split"),
                             random_state=randomState,
                             missing=-1,
                             eval_metric='auc',
                             nthread=4,
                             n_jobs=nJobs)
    XGBModel.fit(X_train, y_train,
                 #             eval_set=[(X_val, y_val)],
                 #             early_stopping_rounds=500,
                 verbose=verboseVal)
    pred = XGBModel.predict(X_test)

    # Test results
    logger.info("test result")
    logger.info("%s %s", 'accuracy', metrics.accuracy_score(y_test, pred))
    logger.info("%s %s", 'precision', metrics.precision_score(y_test, pred))
    logger.info("%s %s", 'recall', metrics.recall_score(y_test, pred))
    logger.info("%s %s", 'f1', metrics.f1_score(y_test, pred))
    logger.info(metrics.classification_report(y_test, pred))
    logger.info(metrics.confusion_matrix(y_test, pred))

    # Saving model products
    # Model
    pickle.dump(XGBModel, open(os.path.join(
        savingPath, modelFile+".sav"), 'wb'))

    # Scaler
    if scaleMethod != 'none':
        pickle.dump(scaler, open(os.path.join(
            savingPath, modelFile+"_scaler.pkl"), "wb"))

    # Feature importances
    imp = pd.DataFrame(
        data={'columns': cols, 'XGB': list(XGBModel.feature_importances_)})
    imp.to_csv(os.path.join(savingPath, modelFile+"_imp.csv"), index=False)

    stop = timeit.default_timer()
    logger.info("%s %s", "수행시간", (stop - start))


# SVM모델
def svm(config):
    start = timeit.default_timer()

    # Paths & Saving options
    trainPath = config.get("data", "trainPath") + \
        config.get("data", "trainData")
    testPath = config.get("data", "testPath") + config.get("data", "testData")
    dropColumns = config.get("data", "dropColumn").split(",")
    targetColumn = config.get("data", "targetColumn")
    keyColumn = config.get("data", "keyColumn")
    savingPath = config.get("data", "savingPath")
    scaleMethod = config.get("data", "scaleMethod")
    modelFile = config.get("data", "fileName")

    # Model options
    nJobs = int(config.get("SVMClassifier", "n_jobs"))
    randomState = int(config.get("SVMClassifier", "random_state"))
    verboseVal = int(config.get("SVMClassifier", "verbose"))
    kernels = config.get("SVMClassifier", "kernel").split(",")
    csVal = ast.literal_eval(config.get("SVMClassifier", "c"))
    gammas = config.get("SVMClassifier", "gamma").split(",")
    decisionFuncionShape = config.get(
        "SVMClassifier", "decision_function_shape").split(",")

    logger.info("Training Model: SVMClassifier")
    logger.info("Model Name: {0}".format(modelFile))
    logger.info("Scale Method: {0}".format(scaleMethod))

    # Load data
    df_train = pd.read_csv(trainPath)
    df_test = pd.read_csv(testPath)

    # Split data
    X_train, X_test, y_train, y_test = Preprocessor(
        df_train, df_test, dropColumns, targetColumn)
    cols = X_train.columns.tolist()
    del df_train, df_test

    logger.info(savingPath)
    logger.info("Train data infomation")
    logger.info(X_train.shape)
    logger.info(Counter(y_train))
    logger.info("Original columns to train")
    logger.info(cols)
    logger.info("Test data infomation")
    logger.info(X_test.shape)
    logger.info(Counter(y_test))

    # Generate variables
    X_train = VariableGenerator(X_train)
    X_test = VariableGenerator(X_test)
    cols = X_train.columns.tolist()

    # Scale data
    X_train, X_test, scaler = Scaler(X_train, X_test, scaleMethod)

    # Sampling
    X_train, y_train = Sampler(X_train, y_train)

    logger.info("Generated columns to train")
    logger.info(cols)
    logger.info("Sampled train data infomation")
    logger.info(X_train.shape)
    logger.info(Counter(y_train))

    # Prevent view warnings
    if scaleMethod == "none":
        X_train.is_copy = False
        X_test.is_copy = False

    # Training parameters and Grid-search to search best model
    params = {'kernel': kernels,
              'C': csVal,
              'gamma': gammas,
              'decision_function_shape': decisionFuncionShape
              }
    svm_clf = SVC(random_state=randomState, class_weight='balanced')
    grid_cv = GridSearchCV(svm_clf, param_grid=params,
                           n_jobs=nJobs, verbose=verboseVal)
    grid_cv.fit(X_train, y_train)
    bestParam = grid_cv.best_params_
    logger.info("%s %s", '최적 하이퍼 파라미터: ', bestParam)
    logger.info("%s %s", '최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))
    grid_cv_df = pd.DataFrame(grid_cv.cv_results_)
    grid_cv_df.sort_values(by=['rank_test_score'], inplace=True)
    logger.info(
        grid_cv_df[['params', 'mean_test_score', 'rank_test_score']].head(10))

    # The best hyper-parameter model
    SVMmodel = SVC(kernel=bestParam.get("kernel"),
                   C=bestParam.get("C"),
                   gamma=bestParam.get("gamma"),
                   decision_function_shape=bestParam.get(
                       "decision_function_shape"),
                   class_weight='balanced',
                   random_state=randomState
                   )
    SVMmodel.fit(X_train, y_train)
    pred = SVMmodel.predict(X_test)

    # Test results
    logger.info("%s %s", 'accuracy', metrics.accuracy_score(y_test, pred))
    logger.info("%s %s", 'precision', metrics.precision_score(y_test, pred))
    logger.info("%s %s", 'recall', metrics.recall_score(y_test, pred))
    logger.info("%s %s", 'f1', metrics.f1_score(y_test, pred))
    logger.info(metrics.classification_report(y_test, pred))
    logger.info(metrics.confusion_matrix(y_test, pred))

    # Saving model products
    # Model
    pickle.dump(SVMmodel, open(os.path.join(
        savingPath, modelFile+".sav"), 'wb'))

    # Scaler
    if scaleMethod != 'none':
        pickle.dump(scaler, open(os.path.join(
            savingPath, modelFile+"_scaler.pkl"), "wb"))

    # Feature importances
    # imp = pd.DataFrame(
    #     data={'columns': cols, 'SVM': SVMmodel.coef_[0]})
    # imp.to_csv(os.path.join(savingPath, modelFile+"_imp.csv"), index=False)

    stop = timeit.default_timer()
    logger.info("%s %s", "수행시간", (stop - start))


# LogisticRegressionCV 모델
def lrc(config):
    start = timeit.default_timer()

    # Paths & Saving options
    trainPath = config.get("data", "trainPath") + \
        config.get("data", "trainData")
    testPath = config.get("data", "testPath") + config.get("data", "testData")
    dropColumns = config.get("data", "dropColumn").split(",")
    targetColumn = config.get("data", "targetColumn")
    keyColumn = config.get("data", "keyColumn")
    savingPath = config.get("data", "savingPath")
    scaleMethod = config.get("data", "scaleMethod")
    modelFile = config.get("data", "fileName")

    # Model options
    nJobs = int(config.get("LRClassifier", "n_jobs"))
    randomState = int(config.get("LRClassifier", "random_state"))
    verboseVal = int(config.get("LRClassifier", "verbose"))
    Csval = ast.literal_eval(config.get("LRClassifier", "Cs"))

    logger.info("Training Model: LRClassifier")
    logger.info("Model Name: {0}".format(modelFile))
    logger.info("Scale Method: {0}".format(scaleMethod))

    # Load data
    df_train = pd.read_csv(trainPath)
    df_test = pd.read_csv(testPath)

    # Split data
    X_train, X_test, y_train, y_test = Preprocessor(
        df_train, df_test, dropColumns, targetColumn)
    cols = X_train.columns.tolist()
    del df_train, df_test

    logger.info(savingPath)
    logger.info("Train data infomation")
    logger.info(X_train.shape)
    logger.info(Counter(y_train))
    logger.info("Original columns to train")
    logger.info(cols)
    logger.info("Test data infomation")
    logger.info(X_test.shape)
    logger.info(Counter(y_test))

    # Generate variables
    X_train = VariableGenerator(X_train)
    X_test = VariableGenerator(X_test)
    cols = X_train.columns.tolist()

    # Scale data
    X_train, X_test, scaler = Scaler(X_train, X_test, scaleMethod)

    # Sampling
    X_train, y_train = Sampler(X_train, y_train)

    logger.info("Generated columns to train")
    logger.info(cols)
    logger.info("Sampled train data infomation")
    logger.info(X_train.shape)
    logger.info(Counter(y_train))

    # Prevent view warnings
    if scaleMethod == "none":
        X_train.is_copy = False
        X_test.is_copy = False

    # Training parameters and Grid-search to search best model
    params = {'Cs': Csval}
    lr_clf = LogisticRegressionCV(fit_intercept=True, cv=5, dual=False, scoring=None, solver='saga', tol=0.0001,
                                  max_iter=200, class_weight={0: 1, 1: 100}, n_jobs=nJobs, verbose=verboseVal, refit=True,
                                  intercept_scaling=1.0, multi_class='ovr', random_state=randomState, l1_ratios=[0.5],
                                  penalty='l2'
                                  )
    grid_cv = GridSearchCV(lr_clf, param_grid=params,
                           n_jobs=nJobs, verbose=verboseVal)
    grid_cv.fit(X_train, y_train)
    bestParam = grid_cv.best_params_
    logger.info("%s %s", '최적 하이퍼 파라미터: ', bestParam)
    logger.info("%s %s", '최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))
    grid_cv_df = pd.DataFrame(grid_cv.cv_results_)
    grid_cv_df.sort_values(by=['rank_test_score'], inplace=True)
    logger.info(
        grid_cv_df[['params', 'mean_test_score', 'rank_test_score']].head(10))

    # The best hyper-parameter model
    LRmodel = LogisticRegressionCV(Cs=bestParam.get("Cs"))
    LRmodel.fit(X_train, y_train)
    pred = LRmodel.predict(X_test)

    # Test results
    logger.info("%s %s", 'accuracy', metrics.accuracy_score(y_test, pred))
    logger.info("%s %s", 'precision', metrics.precision_score(y_test, pred))
    logger.info("%s %s", 'recall', metrics.recall_score(y_test, pred))
    logger.info("%s %s", 'f1', metrics.f1_score(y_test, pred))
    logger.info(metrics.classification_report(y_test, pred))
    logger.info(metrics.confusion_matrix(y_test, pred))

    # Saving model products
    # Model
    pickle.dump(LRmodel, open(os.path.join(
        savingPath, modelFile+".sav"), 'wb'))

    # Scaler
    if scaleMethod != 'none':
        pickle.dump(scaler, open(os.path.join(
            savingPath, modelFile+"_scaler.pkl"), "wb"))

    # Feature importances
    imp = pd.DataFrame(data={'columns': cols, 'LRC': LRmodel.coef_[0]})
    imp.to_csv(os.path.join(savingPath, modelFile+"_imp.csv"), index=False)

    stop = timeit.default_timer()
    logger.info("%s %s", "수행시간", (stop - start))

# DNN모델


def dnn_model(input_dim):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.initializers import glorot_uniform, he_uniform
    from tensorflow.keras.activations import relu, sigmoid

    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation(relu))

    model.add(Dense(64, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation(relu))

    model.add(Dense(64, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation(relu))

    model.add(Dense(32, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation(relu))

    model.add(Dense(16, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation(relu))

    model.add(Dense(8, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation(relu))

    model.add(Dense(1, kernel_initializer='glorot_uniform'))
    model.add(Activation(sigmoid))
    return model

# LSTM모델

def vanilla_LSTM(input_dim):
    model = Sequential()
    model.add(LSTM(32, input_shape = input_dim.shape[1:], kernel_initializer = 'he_uniform', activation = 'relu',
    #               recurrent_regularizer = regularizers.l1(0.01), bias_regularizer = None, recurrent_dropout = None, 
                  activity_regularizer = None, dropout = 0.2, return_sequences = True))
    model.add(LSTM(32, activation = 'relu', return_sequences = True))
    model.add(LSTM(16, activation = 'relu', return_sequences = False))
    # model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))
    return model

# CNN모델 

def vanilla_CNN(input_dim):
    model = Sequential()
    model.add(Conv1D(32, (3), input_shape=input_dim, activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

# AE모델 
def LSTM_AE(input_dim):
    model = Sequential()
    model.add(LSTM(9, activation = 'relu', dropout = 0.2,
                   return_sequences = True, input_shape = input_dim))
    model.add(LSTM(4, activation = 'relu', dropout = 0.2,
                   retirn_sequences = True))
    model.add(LSTM(3, activation = 'relu', dropout = 0.2,
                   retirn_sequences = True))
    model.add(LSTM(2))
    model.add(RepeatVector(input_dim[0]))
    
    model.add(LSTM(3, activation = 'relu', dropout = 0.2,
                   return_sequences = True))
    model.add(LSTM(4, activation = 'relu', dropout = 0.2,
                   retirn_sequences = True))
    model.add(LSTM(9, activation = 'relu', dropout = 0.2,
                   return_sequences = True))
    model.add(LSTM(input_dim[1], return_sequences = True))
    return model
