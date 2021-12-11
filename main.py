from datetime import datetime
from typing import List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from benedict import benedict as bdict
from dateutil.relativedelta import relativedelta
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression


def read_ibov_df(filename: str) -> pd.DataFrame:
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


def read_sp500_df(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, sep='\t', thousands=',')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'Adj Close**']]
    df = df.rename(columns={'Adj Close**': 'close', 'Date': 'date'})
    df = df.sort_values('date')
    return df


def read_sp_500_tr_df(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, usecols=['Current Portfolio Value'])
    data_points = []
    start_date = datetime(1963, 1, 1)
    for i, row in df.iterrows():
        data_points.append({
            'date': start_date,
            'close': row['Current Portfolio Value']
        })
        start_date += relativedelta(months=1)
    df = pd.DataFrame(data_points)
    return df


def find_tops_and_bottom(series: pd.Series, type_: str) -> List:
    if type_ == 'top':
        peaks, _ = find_peaks(series, distance=120)
        peaks = peaks[:-1]
    elif type_ == 'bottom':
        peaks, _ = find_peaks(-series, distance=120)
    else:
        raise ValueError(f'Unknown type {type_}')
    return peaks


def get_trend(tops_or_bottoms: List, series: pd.Series) -> np.ndarray:
    reg = LinearRegression().fit(
        X=np.array(tops_or_bottoms).reshape((-1, 1)),
        y=series[tops_or_bottoms]
    )
    trend = reg.predict(np.array(range(len(ibov_df))).reshape((len(ibov_df), 1)))
    return trend


def get_trends(series: pd.Series) -> Tuple[List, List]:
    tops = find_tops_and_bottom(series, type_='top')
    bottoms = find_tops_and_bottom(series, type_='bottom')

    series_log = np.log(series)

    trend_top_log = get_trend(tops, series_log)
    trend_bottom_log = get_trend(bottoms, series_log)

    trend_top = np.exp(trend_top_log)
    trend_bottom = np.exp(trend_bottom_log)

    return trend_top, trend_bottom


if __name__ == '__main__':
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

    ibov_enfoque_df = read_ibov_df('ibovusd1963.xml')
    ibov_enfoque_df.set_index('date', inplace=True)
    ibov_enfoque_df.columns = [f'IBOV_ENFOQUE_{c}' for c in ibov_enfoque_df.columns]
    ibov_df = ibov_df.merge(ibov_enfoque_df, how='left', left_index=True, right_index=True)

    ibov_df['close'] = ibov_df['IBOV_ENFOQUE_close'].combine_first(ibov_df['IBOV_YAHOO_Close'])
    ibov_df = ibov_df[['close']]
    ibov_df = ibov_df.dropna()
    ibov_df = ibov_df.resample('M').last()
    ibov_df = ibov_df.reset_index()
    ibov_df = ibov_df.rename(columns={'index': 'date'})

    fig, ax = plt.subplots(figsize=(20, 10))

    ibov_start_value = ibov_df['close'].iloc[0]
    ibov_df['close'] /= ibov_start_value

    ax.plot(ibov_df['date'], ibov_df['close'], label='IBOVESPA / USD')
    ibov_df['trend_top'], ibov_df['trend_bottom'] = get_trends(ibov_df['close'])
    ax.plot(ibov_df['date'], ibov_df['trend_top'])
    ax.plot(ibov_df['date'], ibov_df['trend_bottom'])
    ibov_df['close / trend_top'] = ibov_df['close'] / ibov_df['trend_top']
    ibov_df['upside'] = ibov_df['trend_top'] / ibov_df['close']
    print(ibov_df.tail())
    ax.text(0.75, 0.1, f'Upside = {ibov_df["upside"].iloc[-1]:.2f}x USD', transform=ax.transAxes, fontsize=14)

    sp500_df = read_sp_500_tr_df('sp500tr.csv')
    ax.plot(sp500_df['date'], sp500_df['close'], label='SP 500 TR')

    plt.title('IBOVESPA / USD vs SP 500 TR')

    plt.xticks([str(x) for x in ibov_df['date'].iloc[::12]])
    plt.xticks(rotation=70, ha='right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend()

    plt.yscale('log', base=10)
    plt.savefig('chart.png')
    plt.show()
