from seaborn import heatmap
import numpy as np
import config
import matplotlib.pyplot as plt

def visualizer_generate_heat_map(matrix):
    """
    Creates a Heat Map with the XGBClassifier matrix(the model with best recall for anomalies)
    :param matrix: Results of the XGBClassifier
    :return: a png file of the matrix
    """
    heatmap(matrix, cmap=config.COLOR_MAP, annot=True, xticklabels=config.TICK_LABELS,
            yticklabels=config.TICK_LABELS, fmt=config.FORMAT, annot_kws=config.FONT_SIZE)
    plt.xlabel(config.HEATMAP_X_LABEL)
    plt.ylabel(config.HEATMAP_Y_LABEL)
    plt.title(config.HEATMAP_TITLE)
    plt.savefig(config.RESULT_FILE_PATH + config.MATRIX_IMAGE_FILE)
    plt.clf()

def generate_bar_chart(report):
    """
    Creates the chart to compare the values of all the moduls in their precision and recall
    :param report: A dict with the precision and recall of all models
    :return: a png file of the created graph
    """
    precision_normal = [report[m]['report'][config.NORMAL_INTEGER][config.PRECISION] for m in config.MODELS]
    recall_normal = [report[m]['report'][config.NORMAL_INTEGER][config.RECALL] for m in config.MODELS]
    precision_anomaly = [report[m]['report'][config.ANOMALY_INTEGER][config.PRECISION] for m in config.MODELS]
    recall_anomaly = [report[m]['report'][config.ANOMALY_INTEGER][config.RECALL] for m in config.MODELS]

    x = np.arange(len(config.MODELS))
    plt.bar(x - (config.WIDTH + config.WIDTH), precision_normal, config.WIDTH)
    plt.bar(x - config.WIDTH, recall_normal, config.WIDTH)
    plt.bar(x, precision_anomaly, config.WIDTH)
    plt.bar(x + config.WIDTH, recall_anomaly, config.WIDTH)

    plt.xlabel(config.BAR_PLOT_X_LABEL)
    plt.ylabel(config.BAR_PLOT_Y_VALUES)
    plt.xticks(x, config.MODELS)
    plt.legend(config.BAR_PLOT_LEGEND)
    plt.title(config.BAR_PLOT_TITLE)

    plt.savefig(config.RESULT_FILE_PATH + config.BAR_PLOT_FILE)