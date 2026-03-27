from sklearn.metrics import confusion_matrix, classification_report


def ModelPipeline(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    y_prediction = model.predict(X_test)
    matrix = confusion_matrix(Y_test, y_prediction)
    report = classification_report(Y_test, y_prediction, output_dict=True)
    return matrix, report
