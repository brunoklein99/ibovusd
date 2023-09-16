import matplotlib.pyplot as plt

from buffet_indicator import get_buffet_indicator_norm, get_index_average
from ibov_usd import get_ibovusd, _get_trend_line

if __name__ == '__main__':
    ibovusd = get_ibovusd()
    ibovusd_avg = get_index_average(ibovusd)
    buffet_indicator_norm = get_buffet_indicator_norm()

    font_size = 16

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

