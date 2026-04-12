import pandas as pd
import config


def get_open_file(name):
    """
        Reads the file with the given name and cleans it
        by removing the "num_outbound_cmds" and
        mapping the values of the class column {normal, anomaly} with {0,1}, respectively

        :param name: Name of the file to be read in the data folder

        :return DataFrame preprocessed with  no "num_outbound_cmds" column and class column processed
    """
    df = pd.read_csv(config.DATA_FILE_PATH + name)
    df.drop(columns=config.DROP_COLUMNS, inplace=True)
    df[config.CLASS_COLUMN] = df[config.CLASS_COLUMN].map(config.CLASS_MAP)
    return df

