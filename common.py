import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def get_index_average(series: pd.Series) -> pd.Series:
    index_log = np.log(series)

    x = np.array(range(len(index_log))).reshape(-1, 1)
    y = index_log

    reg = LinearRegression()
    reg.fit(x, y)

    average = reg.predict(x)

    return pd.Series(data=np.exp(average), index=series.index)
