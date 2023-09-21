import matplotlib.pyplot as plt

from common import get_index_average
from data import get_ibovusd, get_buffet_indicator_norm, get_sp500, get_cpi_index, _get_trend_line, get_gsci

font_size = 16

if __name__ == '__main__':
    ibovusd = get_ibovusd()
    ibovusd_avg = get_index_average(ibovusd)
    buffet_indicator_norm = get_buffet_indicator_norm()

    fig, ax1 = plt.subplots(figsize=(20, 10))
    color = 'tab:green'
    text = 'IBOVESPA USD / average'
    ax1.plot(ibovusd.index, ibovusd / ibovusd_avg, color=color, label=text)
    ax1.set_ylabel(text, color=color, fontsize=font_size)

    color = 'tab:red'
    text = 'Normalized Buffet Indicator'
    ax2 = ax1.twinx()
    ax2.plot(buffet_indicator_norm.index, buffet_indicator_norm, color=color, label=text)
    ax2.set_ylabel(text, color=color, fontsize=font_size)

    plt.savefig('images/ibov_vs_buffet_indicator.jpg')

    plt.clf()
    fig, ax = plt.subplots(figsize=(20, 10))

    ibovusd_top_trend_line = _get_trend_line(ibovusd, _type='top')
    ibovusd_bottom_trend_line = _get_trend_line(ibovusd, _type='bottom')

    ax.plot(ibovusd.index, ibovusd)
    ax.plot(ibovusd_top_trend_line.index, ibovusd_top_trend_line)
    ax.plot(ibovusd_avg.index, ibovusd_avg)
    ax.plot(ibovusd_bottom_trend_line.index, ibovusd_bottom_trend_line)

    plt.yscale('log', base=10)
    plt.savefig('images/ibov_usd.jpg')

    plt.clf()
    fig, ax1 = plt.subplots(figsize=(20, 10))
    color = 'tab:green'
    text = 'SP500 / CPI'
    sp500 = get_sp500()
    cpi_index = get_cpi_index()
    cpi_index = cpi_index.reindex(sp500.index)
    ax1.plot(sp500.index, sp500 / cpi_index, color=color, label=text)
    ax1.set_ylabel(text, color=color, fontsize=font_size)
    ax1.set_yscale('log', base=2)

    color = 'tab:red'
    text = 'Normalized Buffet Indicator'
    ax2 = ax1.twinx()
    ax2.plot(buffet_indicator_norm.index, buffet_indicator_norm, color=color, label=text)
    ax2.set_ylabel(text, color=color, fontsize=font_size)

    plt.savefig('images/sp500_vs_buffet_indicator.jpg')

    #

    plt.clf()
    fig, ax1 = plt.subplots(figsize=(20, 10))
    color = 'tab:green'
    text = 'GSCI / CPI'
    gsci = get_gsci()
    cpi_index = get_cpi_index()
    cpi_index = cpi_index.reindex(gsci.index)
    ax1.plot(gsci.index, gsci / cpi_index, color=color, label=text)
    ax1.set_ylabel(text, color=color, fontsize=font_size)
    ax1.set_yscale('log', base=2)

    color = 'tab:red'
    text = 'IBOVESPA USD'
    ax2 = ax1.twinx()
    ax2.plot(ibovusd.index, ibovusd, color=color, label=text)
    ax2.set_ylabel(text, color=color, fontsize=font_size)
    ax2.set_yscale('log', base=2)

    plt.savefig('images/gsci_vs_ibov_usd.jpg')

