

def separate_Info(testData, trainData):
    X_train = trainData.drop(columns="class")
    X_test = testData.drop(columns="class")
    Y_train = trainData["class"]
    Y_test = testData["class"]
    return X_train, X_test, Y_train, Y_test