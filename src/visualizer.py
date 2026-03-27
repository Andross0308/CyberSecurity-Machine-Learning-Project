from seaborn import heatmap
import numpy as np
import config
import matplotlib.pyplot as plt

def visualizerGenerateHeatMap(matrix):
    XBBheatmap = heatmap(matrix, cmap=config.COLOR_MAP, annot=True, xticklabels=config.TICK_LABELS,
                      yticklabels=config.TICK_LABELS, fmt=config.FORMAT, annot_kws=config.FONT_SIZE)
    plt.xlabel(config.HEATMAP_X_LABEL)
    plt.ylabel(config.HEATMAP_Y_LABEL)
    plt.title(config.HEATMAP_TITLE)
    plt.savefig(config.RESULT_FILE_PATH + config.MATRIX_IMAGE_FILE)
    plt.clf()

def GenerateBarChart(report):

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

    plt.savefig(config.RESULT_FILE_PATH + config.BAR_PLOT_FILE)