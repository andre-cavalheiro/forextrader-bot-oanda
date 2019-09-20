import pandas as pd
import numpy as np
import sys
import json
from oandapyV20.contrib.factories import InstrumentsCandlesFactory

def requestCandles(apiClient, gran, _from, _to, instr):
    params = {
        "granularity": gran,
        "from": _from,
        "to": _to
    }
    df = None
    for r in InstrumentsCandlesFactory(instrument=instr, params=params):
            print("REQUEST: {} {} {}".format(r, r.__class__.__name__, r.params))
            rv = apiClient.request(r)
            for candle in r.response.get('candles'):
                index = pd.to_datetime(candle.get('time')[0:19])
                data = np.array([[candle['mid']['o'],
                    candle['mid']['h'],
                    candle['mid']['l'],
                    candle['mid']['c']]])
                # print(data)
                df_ = pd.DataFrame({'Open': float(data[:, 0]), 'High': float(data[:, 1]),
                                    'Low': float(data[:, 2]), 'Close': float(data[:, 3])},
                                   index=[index])
                if df is None:
                    df = df_
                else:
                    df = df.append(df_)
    df = df.sort_index()
    return df
