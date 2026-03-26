from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import LabelEncoder


def encodeDataFrame(DataFrame, test = False):
    targetEncoder = TargetEncoder()
    le = LabelEncoder()
    X = DataFrame[["protocol_type", "service", "flag"]]
    if test:
        DataFrame[["protocol_type", "service", "flag"]] = targetEncoder.transform(X)
        DataFrame['class'] = le.transform(DataFrame["class"])
    else:
        Y = DataFrame["class"]
        DataFrame[["protocol_type", "service", "flag"]] = targetEncoder.fit_transform(X,Y)
        DataFrame["class"] = le.fit_transform(DataFrame["class"])