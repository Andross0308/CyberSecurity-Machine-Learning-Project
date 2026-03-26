import pandas as pd
import config

from sklearn.preprocessing import TargetEncoder, RobustScaler

def targetEncoderDataFrame(DataFrame, encoder, test=False):
    X = DataFrame[config.ENCODER_COLUMNS]
    Y = DataFrame[config.CLASS_COLUMN]
    if test:
        return encoder.transform(X)
    else:
        return encoder.fit_transform(X,Y)

def separateDataBase(file):
    return file.drop(columns=config.CLASS_COLUMN), file[config.CLASS_COLUMN]


def encodeDataFrame(train, test):
    encoder = TargetEncoder()
    trainEncoder = targetEncoderDataFrame(train, encoder)
    train[config.ENCODER_COLUMNS] = trainEncoder

    #Encode String in DatBase
    testEncoder = targetEncoderDataFrame(test, encoder, test=True)
    test[config.ENCODER_COLUMNS] = testEncoder
    X_train, Y_train = separateDataBase(train)
    X_test, Y_test = separateDataBase(test)
    return X_train, X_test, Y_train, Y_test

def scaleDataFrame(X_train, X_test):
    scaler = RobustScaler()
    scaler_train = scaler.fit_transform(X_train)
    scaler_test = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(scaler_train, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler_test, columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled
