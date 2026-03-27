from seaborn import heatmap
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