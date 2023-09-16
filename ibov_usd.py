from datetime import datetime
from typing import List

import pandas as pd
import numpy as np
import yfinance as yf
from benedict import benedict as bdict
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression


def _read_ibov_df(filename: str) -> pd.DataFrame:
    data_points = []
    xml = bdict.from_xml(filename)
    for data_point in xml['Result.Poster.Serie.barras.B']:
        data_points.append({
            'date': datetime.strptime(data_point['@d'], '%Y-%m-%d'),
            'close': data_point['@C']
        })
    df = pd.DataFrame(data_points)
    df['close'] = df['close'].astype(float)
    df = df.sort_values('date')
    return df


def _find_tops_and_bottom(series: pd.Series, type_: str) -> List:
    if type_ == 'top':
        peaks, _ = find_peaks(series, distance=120)
        peaks = peaks[:-1]
    elif type_ == 'bottom':
        peaks, _ = find_peaks(-series, distance=120)
    else:
        raise ValueError(f'Unknown type {type_}')
    return peaks


def _get_trend(x: List, series: pd.Series) -> np.ndarray:
    series_log = np.log(series)

    reg = LinearRegression()
    reg.fit(X=np.array(x).reshape((-1, 1)), y=series_log[x])

    trend = reg.predict(np.array(range(len(series_log))).reshape((len(series_log), 1)))

    return pd.Series(data=np.exp(trend), index=series.index)


def _get_trend_line(series: pd.Series, _type: str) -> pd.Series:
    if _type == 'middle':
        tops = _find_tops_and_bottom(series, type_='top')
        bottoms = _find_tops_and_bottom(series, type_='bottom')

        top_line_log = np.log(_get_trend(tops, series))
        bottom_line_log = np.log(_get_trend(bottoms, series))

        trend_middle_log = (top_line_log - bottom_line_log) / 2 + bottom_line_log

        return np.exp(trend_middle_log)
    else:
        points = _find_tops_and_bottom(series, type_=_type)

        trend_line = _get_trend(points, series)

        return trend_line


def get_ibovusd() -> pd.Series:
    ibov_df = pd.DataFrame(
        index=pd.date_range(
            start=datetime(1963, 1, 1),
            end=datetime.today(),
            freq='1D'
        )
    )

    usd_brl_df = yf.Ticker('USDBRL=X').history(period='max')
    usd_brl_df.columns = [f'USDBRL_{c}' for c in usd_brl_df.columns]
    ibov_df = ibov_df.merge(usd_brl_df, how='left', left_index=True, right_index=True)

    ibov_yahoo_df = yf.Ticker('^BVSP').history(period='max')
    ibov_yahoo_df.columns = [f'IBOV_YAHOO_{c}' for c in ibov_yahoo_df.columns]
    ibov_df = ibov_df.merge(ibov_yahoo_df, how='left', left_index=True, right_index=True)
    ibov_df['IBOV_YAHOO_Close'] = ibov_df['IBOV_YAHOO_Close'] / ibov_df['USDBRL_Close']
    ibov_df = ibov_df[['IBOV_YAHOO_Close']]

    ibov_enfoque_df = _read_ibov_df('data/ibovusd1963.xml')
    ibov_enfoque_df.set_index('date', inplace=True)
    ibov_enfoque_df.columns = [f'IBOV_ENFOQUE_{c}' for c in ibov_enfoque_df.columns]
    ibov_df = ibov_df.merge(ibov_enfoque_df, how='left', left_index=True, right_index=True)

    ibov_df['close'] = ibov_df['IBOV_ENFOQUE_close'].combine_first(ibov_df['IBOV_YAHOO_Close'])
    ibov_df = ibov_df[['close']]
    ibov_df = ibov_df.dropna()
    ibov_df = ibov_df.resample('M').last()
    ibov_df = ibov_df.reset_index()
    ibov_df = ibov_df.set_index('index')
    ibov_df = ibov_df.squeeze()

    return ibov_df


if __name__ == '__main__':
    ibovusd = get_ibovusd()
    trend_top = _get_trend_line(ibovusd, _type='top')
    trend_bottom = _get_trend_line(ibovusd, _type='bottom')
    trend_middle = _get_trend_line(ibovusd, _type='middle')

    plt.plot(ibovusd.index, np.log(ibovusd))
    plt.plot(trend_top.index, np.log(trend_top))
    plt.plot(trend_bottom.index, np.log(trend_bottom))
    plt.plot(trend_middle.index, np.log(trend_middle))

    plt.show()
