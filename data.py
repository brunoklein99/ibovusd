import json
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
import cpi
from dateutil.relativedelta import relativedelta
from scipy.signal import find_peaks
from benedict import benedict as bdict
from sklearn.linear_model import LinearRegression

from common import get_index_average


def get_cpi_index(update=False) -> pd.Series:
    df = pd.read_csv('data/cpi.csv', skiprows=1)
    df = df.drop(['HALF1', 'HALF2'], axis=1)
    df = df.melt(id_vars=['Year'], value_vars=df.columns[1:]).rename(columns={'variable': 'month'})

    df['Date'] = df.apply(lambda r: f'{r.Year}-{r.month}-01', axis=1)
    df['Date'] = pd.to_datetime(df['Date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    df = df.set_index('Date')
    df = df.sort_index()
    df = df.dropna()

    cpi_index = pd.Series(index=df.index, data=df['value'])
    cpi_index = cpi_index.reindex(index=pd.date_range(start=cpi_index.index[0], end=cpi_index.index[-1], freq='D'))
    cpi_index = cpi_index.ffill()

    return cpi_index


def get_sp500() -> pd.Series:
    sp500 = yf.Ticker('^GSPC').history('max')
    return pd.Series(data=sp500['Close'], index=sp500.index)


def get_sp500_total_return() -> pd.Series:
    df = pd.read_csv('data/sp500tr.csv')
    df['date'] = df.apply(
        lambda x: datetime(x['Year'], x['Month'], 1) + relativedelta(months=1) - relativedelta(days=1), axis=1)
    df = df.rename(columns={'Amount ($)': 'close'})
    df['close'] = df['close'].str.replace(',', '')
    df['close'] = pd.to_numeric(df['close'])
    df['close'] /= df['close'].iloc[0]
    df = df.set_index('date')

    sp500 = pd.Series(data=df['close'], index=df.index)

    return sp500


if __name__ == '__main__':
    sp500 = get_sp500_total_return()


def parse_datetime(timestamp):
    return datetime.utcfromtimestamp(timestamp / 1000)


def get_gdp() -> pd.Series:
    with open('data/gdp.json', 'r') as fd:
        df = pd.DataFrame(json.load(fd), columns=['date', 'value'])
        df['date'] = df['date'].apply(parse_datetime)
        df = df.set_index('date').squeeze()
    return df


def get_yield_inversion() -> pd.Series:
    with open('data/yield_inversion.json', 'r') as fd:
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


def get_gsci() -> pd.Series:
    with open('data/gsci.json') as fd:
        gsci = json.load(fd)
        dates = [x['date'] for x in gsci]
        values = [x['y'] for x in gsci]

    df = pd.DataFrame({'date': dates, 'value': values})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    gsci = pd.Series(df['value'], df.index)
    gsci = gsci.reindex(pd.date_range(gsci.index[0], gsci.index[-1], freq='D'))
    gsci = gsci.ffill()

    return gsci
