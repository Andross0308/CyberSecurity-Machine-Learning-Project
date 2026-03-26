import numpy as np
import config
from data_loader import get_open_file
from PreProcessor import encodeDataFrame, scaleDataFrame
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from seaborn import heatmap
import matplotlib.pyplot as plt


if __name__ == '__main__':

    #Read First CSV
    df_train = get_open_file(config.TRAIN_FILE)
    df_test = get_open_file(config.TEST_FILE)

    #Split DataBase
    X_train, X_test, Y_train, Y_test = encodeDataFrame(df_train, df_test)

    #First model: Decision Tree
    model_DecisionTree = DecisionTreeClassifier(min_samples_split=config.MIN_SAMPLE_SPLIT,
                                                criterion=config.DECISION_TREE_CRITERION,
                                                max_depth=config.DECISION_TREE_MAX_DEPTH,
                                                class_weight=config.DECISION_TREE_CLASS_WEIGHT)
    model_DecisionTree.fit(X_train, Y_train)
    y_prediction_Decision_tree = model_DecisionTree.predict(X_test)
    matrix_DecisionTree = confusion_matrix(Y_test, y_prediction_Decision_tree)
    report_DecisionTree = classification_report(Y_test, y_prediction_Decision_tree, output_dict=True)
    print("matrix: \n", matrix_DecisionTree)
    print("report: \n", report_DecisionTree)

    #Second Model
    print("============================================= RANDOM FOREST ===============================================\n")
    model_RF = RandomForestClassifier(random_state=config.RANDOM_FOREST_RANDOM_STATE,
                                      min_samples_split=config.MIN_SAMPLE_SPLIT)
    model_RF.fit(X_train, Y_train)
    y_prediction = model_RF.predict(X_test)
    matrix_RandomForest = confusion_matrix(Y_test, y_prediction)
    report_RandomForest = classification_report(Y_test, y_prediction, output_dict=True)
    print("matrix: \n", matrix_RandomForest)
    print("report: \n", report_RandomForest)

    #Third Model
    print("============================================= XBB Classifier ==============================================\n")
    model_XBB = XGBClassifier(learning_rate=config.XGB_LEARNING_RATE, max_depth=config.XGB_MAX_DEPTH,
                              n_estimators=config.XGB_N_ESTIMATORS, scale_pos_weight=config.XGB_SCALE_POS_WEIGHT,
                              subsample=config.XGB_SUBSAMPLE)
    model_XBB.fit(X_train, Y_train)
    XBB_prediction = model_XBB.predict(X_test)
    matrix_XBB = confusion_matrix(Y_test, XBB_prediction)
    report_XBB = classification_report(Y_test, XBB_prediction, output_dict=True)
    print("matrix: \n", matrix_XBB)
    print("report: \n", report_XBB)

    #Create Heat Map with the confusion matrix
    heatmap = heatmap(matrix_XBB, cmap=config.COLOR_MAP, annot=True, xticklabels=config.TICK_LABELS,
                      yticklabels=config.TICK_LABELS, fmt=config.FORMAT, annot_kws=config.FONT_SIZE)
    plt.xlabel(config.HEATMAP_X_LABEL)
    plt.ylabel(config.HEATMAP_Y_LABEL)
    plt.title(config.HEATMAP_TITLE)
    plt.savefig(config.RESULT_FILE_PATH + config.MATRIX_IMAGE_FILE)
    plt.clf()

    #Fourth Model
    print("============================================= K Nearest Neighbours =========================================\n")
    X_train_scaled, X_test_scaled = scaleDataFrame(X_train, X_test)
    classifier = KNeighborsClassifier(metric=config.KNN_METRIC, n_neighbors=config.KNN_N_NEIGHBORS,
                                      weights=config.KNN_WEIGHTS)
    classifier.fit(X_train_scaled, Y_train)
    KNN_Predictions = classifier.predict(X_test_scaled)
    matrix_KNN = confusion_matrix(Y_test, KNN_Predictions)
    report_KNN = classification_report(Y_test, KNN_Predictions, output_dict=True)
    print("matrix: \n", matrix_KNN)
    print("report: \n", report_KNN)


    #Create graph to compare results of all modules
    recall_anomaly = [report_DecisionTree[config.ANOMALY_INTEGER][config.RECALL],
                      report_RandomForest[config.ANOMALY_INTEGER][config.RECALL],
                      report_XBB[config.ANOMALY_INTEGER][config.RECALL],
                      report_KNN[config.ANOMALY_INTEGER][config.RECALL]]
    precision_anomaly = [report_DecisionTree[config.ANOMALY_INTEGER][config.PRECISION],
                         report_RandomForest[config.ANOMALY_INTEGER][config.PRECISION],
                         report_XBB[config.ANOMALY_INTEGER][config.PRECISION],
                         report_KNN[config.ANOMALY_INTEGER][config.PRECISION]]
    recall_normal = [report_DecisionTree[config.NORMAL_INTEGER][config.RECALL],
                     report_RandomForest[config.NORMAL_INTEGER][config.RECALL],
                     report_XBB[config.NORMAL_INTEGER][config.RECALL],
                     report_KNN[config.NORMAL_INTEGER][config.RECALL]]
    precision_normal = [report_DecisionTree[config.NORMAL_INTEGER][config.PRECISION],
                        report_RandomForest[config.NORMAL_INTEGER][config.PRECISION],
                        report_XBB[config.NORMAL_INTEGER][config.PRECISION],
                        report_KNN[config.NORMAL_INTEGER][config.PRECISION]]


    x = np.arange(len(config.MODELS))
    plt.bar(x - (config.WIDTH + config.WIDTH), precision_normal, config.WIDTH)
    plt.bar(x-config.WIDTH, recall_normal, config.WIDTH)
    plt.bar(x, precision_anomaly, config.WIDTH)
    plt.bar(x + config.WIDTH, recall_anomaly, config.WIDTH)

    plt.xlabel(config.BAR_PLOT_X_LABEL)
    plt.ylabel(config.BAR_PLOT_Y_VALUES)
    plt.xticks(x, config.MODELS)
    plt.legend(config.BAR_PLOT_LEGEND)


    plt.savefig(config.RESULT_FILE_PATH + config.BAR_PLOT_FILE)