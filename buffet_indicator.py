import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def parse_datetime(timestamp):
    return datetime.utcfromtimestamp(timestamp / 1000)


def get_index_average(series: pd.Series) -> pd.Series:
    index_log = np.log(series)

    x = np.array(range(len(index_log))).reshape(-1, 1)
    y = index_log

    reg = LinearRegression()
    reg.fit(x, y)

    average = reg.predict(x)

    return pd.Series(data=np.exp(average), index=series.index)


def get_gdp() -> pd.Series:
    with open('data/gdp.json', 'r') as fd:
        df = pd.DataFrame(json.load(fd), columns=['date', 'value'])
        df['date'] = df['date'].apply(parse_datetime)
        df = df.set_index('date').squeeze()
    return df


def get_wilshire5000() -> pd.Series:
    with open('data/wilshire5000.json', 'r') as fd:
        df = pd.DataFrame(json.load(fd), columns=['date', 'value'])
        df['date'] = df['date'].apply(parse_datetime)
        df = df.set_index('date').squeeze()
    return df


def get_buffet_indicator_norm() -> pd.Series:
    wilshire5000 = get_wilshire5000()
    gdp = get_gdp()
    buffet_indicator = wilshire5000 / gdp
    buffet_indicator_avg = get_index_average(buffet_indicator)
    buffet_indicator_norm = buffet_indicator / buffet_indicator_avg
    return buffet_indicator_norm


if __name__ == '__main__':
    buffet_indicator_norm = get_buffet_indicator_norm()
    plt.plot(buffet_indicator_norm.index, buffet_indicator_norm)
    plt.show()
