import config
from data_loader import get_open_file
from ModelManager import get_trained_models
from PreProcessor import encodeDataFrame
from visualizer import visualizerGenerateHeatMap, GenerateBarChart


if __name__ == '__main__':

    #Read First CSV
    df_train = get_open_file(config.TRAIN_FILE)
    df_test = get_open_file(config.TEST_FILE)

    #Split DataBase
    X_train, X_test, Y_train, Y_test = encodeDataFrame(df_train, df_test)

    results = get_trained_models(X_train, Y_train, X_test, Y_test)
    print(results)

    visualizerGenerateHeatMap(results['XGBoost']['matrix'])
    GenerateBarChart(results)