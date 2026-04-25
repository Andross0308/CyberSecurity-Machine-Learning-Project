import config
from data_loader import get_open_file
from ModelManager import get_trained_models
from PreProcessor import encodeDataFrame
from visualizer import visualizer_generate_heat_map, generate_bar_chart


if __name__ == '__main__':

    #Read CSVs
    df_train = get_open_file(config.TRAIN_FILE)
    df_test = get_open_file(config.TEST_FILE)

    #Split DataBase
    X_train, X_test, Y_train, Y_test = encodeDataFrame(df_train, df_test)

    #Trains all the models
    results = get_trained_models(X_train, Y_train, X_test, Y_test)
    print(results)

    visualizer_generate_heat_map(results['XGBoost']['matrix'])
    generate_bar_chart(results)