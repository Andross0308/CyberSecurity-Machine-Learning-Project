import config

from PreProcessor import scaleDataFrame
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def model_pipeline(model, X_train, Y_train, X_test, Y_test):
    """
    Trains and tests the given model and gives the results
    :param model: Model that is used in the pipeline
    :param X_train: Training values for the model
    :param Y_train: Training class for the model
    :param X_test: Testing values for the model
    :param Y_test: Testing class for the model
    :return: a 2x2 matrix with the results and a dict with the precision and recall of the model
    """
    model.fit(X_train, Y_train)
    y_prediction = model.predict(X_test)
    matrix = confusion_matrix(Y_test, y_prediction)
    report = classification_report(Y_test, y_prediction, output_dict=True)
    return matrix, report


def get_trained_models(X_train, Y_train, X_test, Y_test):
    """
    Trains and tests all the models and gives the results
    :param X_train: Training values for the models
    :param Y_train: Training class for the models
    :param X_test: Testing values for the models
    :param Y_test: Testing class for the models
    :return: a dict with the results of the models
    """
    X_train_scaled, X_test_scaled = scaleDataFrame(X_train, X_test)
    models = {
        "Decision Tree": (DecisionTreeClassifier(min_samples_split=config.MIN_SAMPLE_SPLIT,
                                            criterion=config.DECISION_TREE_CRITERION,
                                            max_depth=config.DECISION_TREE_MAX_DEPTH,
                                            class_weight=config.DECISION_TREE_CLASS_WEIGHT), lambda: (X_train, X_test)),
        "Random Forest": (RandomForestClassifier(random_state=config.RANDOM_FOREST_RANDOM_STATE,
                                  min_samples_split=config.MIN_SAMPLE_SPLIT), lambda: (X_train, X_test)),
        "XGBoost": (XGBClassifier(learning_rate=config.XGB_LEARNING_RATE, max_depth=config.XGB_MAX_DEPTH,
                          n_estimators=config.XGB_N_ESTIMATORS, scale_pos_weight=config.XGB_SCALE_POS_WEIGHT,
                          subsample=config.XGB_SUBSAMPLE), lambda: (X_train, X_test)),
        "KNN": (KNeighborsClassifier(metric=config.KNN_METRIC, n_neighbors=config.KNN_N_NEIGHBORS,
                             weights=config.KNN_WEIGHTS), lambda: (X_train_scaled, X_test_scaled))
    }

    results = {}
    for name, (model, get_data) in models.items():
        current_X_train, current_X_test = get_data()
        matrix, report = model_pipeline(model, current_X_train, Y_train, current_X_test, Y_test)
        results[name] = {"matrix": matrix, "report": report}
        print(f"{name} Done")

    return results