
#Data Base Constants
CLASS_COLUMN = "class"
ENCODER_COLUMNS = ["protocol_type", "service", "flag"]
DROP_COLUMNS = ["num_outbound_cmds"]
CLASS_MAP = {'normal': 0, 'anomaly': 1}

# Path to open files
DATA_FILE_PATH= "../data"
TRAIN_FILE = "/KDDTrain.arff"
TEST_FILE = "/KDDTest.arff"
RESULT_FILE_PATH = "../Results"
MATRIX_IMAGE_FILE = "/Xbb_matrix.png"
BAR_PLOT_FILE = "/Graph.png"
COMMA = ","

#Models Constants
MIN_SAMPLE_SPLIT = 2
DECISION_TREE_CRITERION = "entropy"
DECISION_TREE_MAX_DEPTH = 30
DECISION_TREE_CLASS_WEIGHT = "balanced"
RANDOM_FOREST_RANDOM_STATE = 100
XGB_LEARNING_RATE = 0.01
XGB_MAX_DEPTH = 7
XGB_N_ESTIMATORS = 100
XGB_SCALE_POS_WEIGHT = 25
XGB_SUBSAMPLE = 1.0
KNN_METRIC = 'manhattan'
KNN_N_NEIGHBORS = 3
KNN_WEIGHTS = 'distance'

#HeatMap constants
COLOR_MAP = "Greens"
TICK_LABELS = ["Normal", "Anomaly"]
FORMAT = 'd'
FONT_SIZE = {'size': 20}
HEATMAP_X_LABEL = "Predicted"
HEATMAP_Y_LABEL = "Actual"
HEATMAP_TITLE = "XGB Classifier Confusion Matrix"

#Graph Constant
MODELS = ['Decision Tree', 'Random Forest', 'XGBoost', 'KNN']
NORMAL_INTEGER = '0'
ANOMALY_INTEGER = '1'
RECALL = 'recall'
PRECISION = 'precision'
WIDTH = 0.2
BAR_PLOT_X_LABEL = "Modules"
BAR_PLOT_Y_VALUES = "Values"
BAR_PLOT_LEGEND = ["precision Normal", "recall Normal", "precision Anomaly", "recall Anomaly"]

