import pandas as pd
import config

from sklearn.preprocessing import TargetEncoder, RobustScaler


def targetEncoderDataFrame(DataFrame, encoder, test=False):
    """
    Encodes the DataFrame using a Target Encoder
    for the string values columns(protocol_type,service,flag)
    :param DataFrame: Data Frame to  Encode
    :param encoder: TargetEncoder that is fitted with the values of the train DataFrame and transform both DataFrames
    :param test: Boolean that says the DataFrame is the test so that the Encoder doesn't fit, only transforms
                Default value = false
    :return:  columns of the DataFrame after Encoding
    """

    X = DataFrame[config.ENCODER_COLUMNS]
    Y = DataFrame[config.CLASS_COLUMN]
    if test:
        return encoder.transform(X)
    else:
        return encoder.fit_transform(X,Y)

def separateDataBase(file):
    """
    Divides the given file into the data (X) and result (Y)
    :param file: DataFrame of the file
    :return: a tuple with the dataFrame divided in the Data and the Result
    """
    return file.drop(columns=config.CLASS_COLUMN), file[config.CLASS_COLUMN]


def encodeDataFrame(train, test):
    """
    Encodes both the train and test DataFrames by
    transforming the String columns into integers
    and splits them into the X_train, Y_train,  X_test and Y_test
    :param train: Train Dataframe
    :param test: Test Dataframe
    :return: the dataframes separated in X and Y for both training and testing

    """

    encoder = TargetEncoder()
    trainEncoder = targetEncoderDataFrame(train, encoder)
    train[config.ENCODER_COLUMNS] = trainEncoder

    testEncoder = targetEncoderDataFrame(test, encoder, test=True)
    test[config.ENCODER_COLUMNS] = testEncoder
    X_train, Y_train = separateDataBase(train)
    X_test, Y_test = separateDataBase(test)
    return X_train, X_test, Y_train, Y_test

def scaleDataFrame(X_train, X_test):
    """
    Scales the information of the DataFrames using RobustScaler
    to be used in the KNN module
    :param X_train: Training information
    :param X_test: Testing information
    :return: the information of the files scaled
    """
    scaler = RobustScaler()
    scaler_train = scaler.fit_transform(X_train)
    scaler_test = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(scaler_train, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler_test, columns=X_test.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled
