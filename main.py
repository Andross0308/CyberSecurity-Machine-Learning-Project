import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from seaborn import heatmap
import matplotlib.pyplot as plt

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




    model_DecisionTree = DecisionTreeClassifier(min_samples_split=2, criterion="entropy", max_depth=30, class_weight='balanced')
    model_DecisionTree.fit(X_train, Y_train)
    y_prediction_Decision_tree = model_DecisionTree.predict(X_test)
    matrix_DecisionTree = confusion_matrix(Y_test, y_prediction_Decision_tree)
    report_DecisionTree = classification_report(Y_test, y_prediction_Decision_tree, output_dict=True)
    print("matrix: \n", matrix_DecisionTree)
    print("report: \n", report_DecisionTree)

    print("============================================= RANDOM FOREST ===============================================\n")
    model_RF = RandomForestClassifier(random_state=100, min_samples_split=2)
    model_RF.fit(X_train, Y_train)
    y_prediction = model_RF.predict(X_test)
    matrix_RandomForest = confusion_matrix(Y_test, y_prediction)
    report_RandomForest = classification_report(Y_test, y_prediction, output_dict=True)
    print("matrix: \n", matrix_RandomForest)
    print("report: \n", report_RandomForest)

    print("============================================= XBB Classifier ==============================================\n")

    model_XBB = XGBClassifier(learning_rate=0.01, max_depth=7, n_estimators=100, scale_pos_weight=25, subsample=1.0)
    model_XBB.fit(X_train, Y_train)
    XBB_prediction = model_XBB.predict(X_test)
    matrix_XBB = confusion_matrix(Y_test, XBB_prediction)
    report_XBB = classification_report(Y_test, XBB_prediction, output_dict=True)
    print("matrix: \n", matrix_XBB)
    print("report: \n", report_XBB)
    heatmap = heatmap(matrix_XBB, cmap="Greens", annot=True, xticklabels=["Normal", "Anomaly"],
                      yticklabels=["Normal", "Anomaly"], fmt='d', annot_kws={'size': 20})
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("XGB Classifier Confusion Matrix")
    plt.savefig("Results/Xbb_matrix.png")

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
    report_KNN = classification_report(Y_test, KNN_Predictions, output_dict=True)
    print("matrix: \n", matrix_KNN)
    print("report: \n", report_KNN)

    models = ['Decision Tree', 'Random Forest', 'XGBoost', 'KNN']
    recall_anomaly = [report_DecisionTree['1']['recall'],  report_RandomForest['1']['recall'],
                      report_XBB['1']['recall'], report_KNN['1']['recall']]
    precision_anomaly = [report_DecisionTree['1']['precision'],  report_RandomForest['1']['precision'],
                      report_XBB['1']['precision'], report_KNN['1']['precision']]
    recall_normal = [report_DecisionTree['0']['recall'],  report_RandomForest['0']['recall'],
                      report_XBB['0']['recall'], report_KNN['0']['recall']]
    precision_normal = [report_DecisionTree['0']['precision'],  report_RandomForest['0']['precision'],
                      report_XBB['0']['precision'], report_KNN['0']['precision']]

    w, x = 0.2, np.arange(len(models))

    fig, ax = plt.subplots()
    ax.bar(x - 0.4, recall_anomaly, width=w, label='recall_anomally')
    ax.bar(x - 0.2, precision_anomaly, width=w, label='precision_anomally')
    ax.bar(x + 0.2 , precision_normal, width=w, label='precision_normal')
    ax.bar(x, recall_normal, width=w, label='recall_normal')

    ax.set_xticks(x)
    ax.set_xticklabels(models)

    plt.savefig("Results/Graph.png")