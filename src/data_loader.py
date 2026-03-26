import pandas as pd
import config

def get_open_file(name):
    df = pd.read_csv(config.DATA_FILE_PATH + name, sep=config.COMMA)
    df.drop(columns=config.DROP_COLUMNS, inplace=True)
    df[config.CLASS_COLUMN] = df[config.CLASS_COLUMN].map(config.CLASS_MAP)
    return df

