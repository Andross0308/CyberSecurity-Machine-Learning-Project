import numpy as np
import config
from data_loader import get_open_file
from ModelManager import ModelPipeline
from PreProcessor import encodeDataFrame, scaleDataFrame
from visualizer import visualizerGenerateHeatMap
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

model_DecisionTree = DecisionTreeClassifier(min_samples_split=config.MIN_SAMPLE_SPLIT,
                                            criterion=config.DECISION_TREE_CRITERION,
                                            max_depth=config.DECISION_TREE_MAX_DEPTH,
                                            class_weight=config.DECISION_TREE_CLASS_WEIGHT)

model_RF = RandomForestClassifier(random_state=config.RANDOM_FOREST_RANDOM_STATE,
                                  min_samples_split=config.MIN_SAMPLE_SPLIT)

model_XBB = XGBClassifier(learning_rate=config.XGB_LEARNING_RATE, max_depth=config.XGB_MAX_DEPTH,
                          n_estimators=config.XGB_N_ESTIMATORS, scale_pos_weight=config.XGB_SCALE_POS_WEIGHT,
                          subsample=config.XGB_SUBSAMPLE)

model_KNN = KNeighborsClassifier(metric=config.KNN_METRIC, n_neighbors=config.KNN_N_NEIGHBORS,
                                  weights=config.KNN_WEIGHTS)


if __name__ == '__main__':

    #Read First CSV
    df_train = get_open_file(config.TRAIN_FILE)
    df_test = get_open_file(config.TEST_FILE)

    #Split DataBase
    X_train, X_test, Y_train, Y_test = encodeDataFrame(df_train, df_test)

    matrixDCT, reportDCT = ModelPipeline(model_DecisionTree, X_train, Y_train, X_test, Y_test)

    print("Decision Tree Done")

    #Second Model
    matrixRFT, reportRFT = ModelPipeline(model_RF, X_train, Y_train, X_test, Y_test)
    print("Random Forest Done")

    #Third Model
    matrixXBB, reportXBB = ModelPipeline(model_XBB, X_train, Y_train, X_test, Y_test)
    print("XBB Classifier Done")

    X_train_scaled, X_test_scaled = scaleDataFrame(X_train, X_test)
    matrixKNN, reportKNN = ModelPipeline(model_KNN, X_train_scaled, Y_train, X_test_scaled, Y_test)
    print("KNN Classifier Done")

    visualizerGenerateHeatMap(matrixXBB)
    print("Matrix Done")

    #Create graph to compare results of all modules
    recall_anomaly = [reportDCT[config.ANOMALY_INTEGER][config.RECALL],
                      reportRFT[config.ANOMALY_INTEGER][config.RECALL],
                      reportXBB[config.ANOMALY_INTEGER][config.RECALL],
                      reportKNN[config.ANOMALY_INTEGER][config.RECALL]]
    precision_anomaly = [reportDCT[config.ANOMALY_INTEGER][config.PRECISION],
                         reportRFT[config.ANOMALY_INTEGER][config.PRECISION],
                         reportXBB[config.ANOMALY_INTEGER][config.PRECISION],
                         reportKNN[config.ANOMALY_INTEGER][config.PRECISION]]
    recall_normal = [reportDCT[config.NORMAL_INTEGER][config.RECALL],
                     reportRFT[config.NORMAL_INTEGER][config.RECALL],
                     reportXBB[config.NORMAL_INTEGER][config.RECALL],
                     reportKNN[config.NORMAL_INTEGER][config.RECALL]]
    precision_normal = [reportDCT[config.NORMAL_INTEGER][config.PRECISION],
                        reportRFT[config.NORMAL_INTEGER][config.PRECISION],
                        reportXBB[config.NORMAL_INTEGER][config.PRECISION],
                        reportKNN[config.NORMAL_INTEGER][config.PRECISION]]


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