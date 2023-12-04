import math

import numpy as np
import yfinance as yf


def sigmoid(x):
    return 1/(1 + math.exp(-x))


def stocks_price_format(n):
    if (n < 0):
        return '- $ {0:2f}'.format(abs(n))
    else:
        return '+ $ {0:2f}'.format(abs(n))


def stocker_market(ticker='AAPL', period='max', interval='1mo'):
    dataset = yf.download(ticker, period=period, interval=interval)
    start_date = str(dataset.index[0]).split()[0]
    end_date = str(dataset.index[-1]).split()[0]
    close_stocker = dataset['Close']

    return close_stocker


def state_creator(data, timestep, window_size):
    starting_id = timestep - window_size + 1

    if starting_id >= 0:
        windowed_data = np.array(data[starting_id:timestep + 1])
    else:
        windowed_data = np.array(- starting_id *
                                 [data[0]] + list(data[0:timestep + 1]))

    state = []
    for i in range(window_size - 1):
        state.append(sigmoid(windowed_data[i + 1] - windowed_data[i]))

    return np.array([state]), windowed_data
