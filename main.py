import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, make_scorer
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

def separate_Info(testData, trainData):
    X_train = trainData.drop(columns="class")
    X_test = testData.drop(columns="class")
    Y_train = trainData["class"]
    Y_test = testData["class"]
    return X_train, X_test, Y_train, Y_test

def encode_DataFrame(DataFrame, encoder, test=False):
    X = DataFrame[["protocol_type", "service", "flag"]]
    Y = DataFrame["class"]
    if test:
        print("Test file")
        return encoder.transform(X)
    else:
        print("Train file")
        return encoder.fit_transform(X,Y)


if __name__ == '__main__':

    df_train = pd.read_csv("data/KDDTrain.arff", sep=',')
    df_train.drop(columns=["num_outbound_cmds"], inplace=True)

    encoder = TargetEncoder()
    le = LabelEncoder()
    mapping = {'normal': 0, 'anomaly': 1}
    one_hot_encoder = encode_DataFrame(df_train, encoder)
    df_train[["protocol_type", "service", "flag"]] = one_hot_encoder
    df_train["class"] = df_train["class"].map(mapping)

    df_test = pd.read_csv("data/KDDTest.arff", sep=',')
    df_test.drop(columns=["num_outbound_cmds"], inplace=True)
    one= encode_DataFrame(df_test, encoder, test=True)
    df_test[["protocol_type", "service", "flag"]] = one
    df_test['class'] = df_test["class"].map(mapping)
    df_numbers_test = df_test.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])

    X_train, X_test, Y_train, Y_test = separate_Info(df_test, df_train)


    """recall_class = make_scorer(recall_score, pos_label=0)
    param_grid = {
        'n_neighbors': [3, 5, 7, 11],  # Valores ímpares evitam empates
        'weights': ['uniform', 'distance'],  # 'distance' dá mais peso aos vizinhos mais próximos
        'metric': ['euclidean', 'manhattan']
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='recall')
    grid_search.fit(X_train, Y_train)

    print(f"Best Score: {grid_search.best_score_}")
    print(f"Best Parameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(classification_report(Y_test, y_pred))"""

    model_DecisionTree = DecisionTreeClassifier(min_samples_split=2, criterion="entropy", max_depth=30, class_weight='balanced')
    model_DecisionTree.fit(X_train, Y_train)
    y_prediction_Decision_tree = model_DecisionTree.predict(X_test)
    matrix_DecisionTree = confusion_matrix(Y_test, y_prediction_Decision_tree)
    report_DecisionTree = classification_report(Y_test, y_prediction_Decision_tree)
    print("matrix: \n", matrix_DecisionTree)
    print("report: \n", report_DecisionTree)

    print("============================================= RANDOM FOREST ===============================================\n")
    model_RF = RandomForestClassifier(random_state=100, min_samples_split=2)
    model_RF.fit(X_train, Y_train)
    y_prediction = model_RF.predict(X_test)
    features = model_RF.feature_importances_
    s = pd.Series(features, index=X_test.columns).sort_values(ascending=False)
    print(s)
    matrix_RandomForest = confusion_matrix(Y_test, y_prediction)
    report_RandomForest = classification_report(Y_test, y_prediction)
    print("matrix: \n", matrix_RandomForest)
    print("report: \n", report_RandomForest)

    print("============================================= XBB Classifier ==============================================\n")



    model_XBB = XGBClassifier(learning_rate=0.01, max_depth=7, n_estimators=100, scale_pos_weight=25, subsample=1.0)
    model_XBB.fit(X_train, Y_train)
    XBB_prediction = model_XBB.predict(X_test)
    matrix_XBB = confusion_matrix(Y_test, XBB_prediction)
    report_XBB = classification_report(Y_test, XBB_prediction)
    print("matrix: \n", matrix_XBB)
    print("report: \n", report_XBB)

    print("============================================= K Nearest Neighbours =========================================\n")
    scaler = RobustScaler()
    scaler_train = scaler.fit_transform(X_train)
    scaler_test = scaler.transform(X_test)
    df_train[X_train.columns] = scaler_train
    X_train = df_train[X_train.columns]
    df_test[X_test.columns] = scaler_test
    X_test = df_test[X_test.columns]
    classifier = KNeighborsClassifier(metric='manhattan', n_neighbors=3, weights='distance')
    classifier.fit(X_train, Y_train)
    KNN_Predictions = classifier.predict(X_test)
    matrix_KNN = confusion_matrix(Y_test, KNN_Predictions)
    report_KNN = classification_report(Y_test, KNN_Predictions)
    print("matrix: \n", matrix_KNN)
    print("report: \n", report_KNN)