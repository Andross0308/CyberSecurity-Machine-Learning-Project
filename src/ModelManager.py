import config

from PreProcessor import scaleDataFrame
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def ModelPipeline(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    y_prediction = model.predict(X_test)
    matrix = confusion_matrix(Y_test, y_prediction)
    report = classification_report(Y_test, y_prediction, output_dict=True)
    return matrix, report


def get_trained_models(X_train, Y_train, X_test, Y_test):
    models = {
        "Decision Tree": DecisionTreeClassifier(min_samples_split=config.MIN_SAMPLE_SPLIT,
                                            criterion=config.DECISION_TREE_CRITERION,
                                            max_depth=config.DECISION_TREE_MAX_DEPTH,
                                            class_weight=config.DECISION_TREE_CLASS_WEIGHT),
        "Random Forest": RandomForestClassifier(random_state=config.RANDOM_FOREST_RANDOM_STATE,
                                  min_samples_split=config.MIN_SAMPLE_SPLIT),
        "XGBoost": XGBClassifier(learning_rate=config.XGB_LEARNING_RATE, max_depth=config.XGB_MAX_DEPTH,
                          n_estimators=config.XGB_N_ESTIMATORS, scale_pos_weight=config.XGB_SCALE_POS_WEIGHT,
                          subsample=config.XGB_SUBSAMPLE)
    }

    results = {}
    for name, model in models.items():
        matrix, report = ModelPipeline(model, X_train, Y_train, X_test, Y_test)
        results[name] = {"matrix": matrix, "report": report}
        print(f"{name} Done")

    X_train_scaled, X_test_scaled = scaleDataFrame(X_train, X_test)
    model =  KNeighborsClassifier(metric=config.KNN_METRIC, n_neighbors=config.KNN_N_NEIGHBORS,
                                  weights=config.KNN_WEIGHTS)
    matrix, report = ModelPipeline(model, X_train, Y_train, X_test, Y_test)
    results["KNN"] = {"matrix": matrix, "report": report}
    print("KNN")

    return results