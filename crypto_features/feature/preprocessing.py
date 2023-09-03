"""
Preprocessing module for feature engineering.
"""
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')


class PreprocessingBinance:
    """
    Preprocessing from binance historical data.
    """
    _BINANCE_KLINES_DIR = os.path.join("data", "futures", "um", "daily", "klines")
    _BINANCE_FUNDINGRATE_DIR = os.path.join("data", "futures", "um", "monthly", "fundingRate")

    def __init__(self, data_dir):
        self._data_dir = data_dir

    def _load_klines_data(self, symbol) -> pd.DataFrame:
        """
        Load klines data from csv files.
        :param symbol: symbol name
        :return: preprocessed klines data
        """
        # Load klines data
        # merge all csv files
        df = pd.DataFrame()
        for file in os.listdir(os.path.join(self._data_dir, self._BINANCE_KLINES_DIR, symbol, "1m")):
            df = pd.concat([df, pd.read_csv("/".join([self._data_dir, self._BINANCE_KLINES_DIR, symbol, "1m", file]))])

        df.columns = ["timestamp_open","open","high","low","close","volume","timestamp_close","volume_usdt","count","taker_buy_volume","taker_buy_quote_volume","ignore"]
        df['timestamp_open'] = pd.to_datetime(df['timestamp_open'], utc=True, unit='ms')
        df.set_index('timestamp_open', inplace=True)

        return df

    def _load_fundingrate_data(self, symbol) -> pd.DataFrame:
        """
        Load funding rate data from csv files.
        :param symbol: symbol name
        :return: preprocessed funding rate data
        """
        # Load funding rate data
        # merge all csv files
        df = pd.DataFrame()
        for file in os.listdir(os.path.join(self._data_dir, self._BINANCE_FUNDINGRATE_DIR, symbol)):
            df = pd.concat([df, pd.read_csv("/".join([self._data_dir, self._BINANCE_FUNDINGRATE_DIR, symbol, file]))])

        df.columns = ['timestamp_open', "interval_time", 'funding_rate']
        df['timestamp_open'] = pd.to_datetime(df['timestamp_open'], utc=True, unit='ms')
        df.set_index('timestamp_open', inplace=True)
        df = df.drop('interval_time', axis=1)

        return df


class PreprocessingBybit:
    """
    Preprocessing from bybit historical data.
    """
    _BYBIT_KLINES_DIR = os.path.join("bybit_data", "klines")
    _BYBIT_FUNDINGRATE_DIR = os.path.join("bybit_data", "fundingRate")

    def __init__(self, data_dir):
        self._data_dir = data_dir

    def _load_klines_data(self, symbol) -> pd.DataFrame:
        """
        Load klines data from csv files.
        :param symbol: symbol name
        :return: preprocessed klines data
        """
        # Load klines data
        df = pd.read_csv(os.path.join(self._data_dir, self._BYBIT_KLINES_DIR, symbol, '1m.csv'))
        df.columns = ["index", 'timestamp_open', 'open', 'high', 'low', 'close', 'volume', 'volume_usdt', "index2"]
        df = df.drop(['index', 'index2'], axis=1)
        df['timestamp_open'] = pd.to_datetime(df['timestamp_open'], utc=True)
        df.set_index('timestamp_open', inplace=True)

        return df

    def _load_fundingrate_data(self, symbol) -> pd.DataFrame:
        """
        Load funding rate data from csv files.
        :param symbol: symbol name
        :return: preprocessed funding rate data
        """
        # Load funding rate data
        df = pd.read_csv(os.path.join(self._data_dir, self._BYBIT_FUNDINGRATE_DIR, symbol + '.csv'))
        df.columns = ["index", 'funding_rate', 'timestamp_open', 'symbol']
        df = df.drop(['index', 'symbol'], axis=1)
        df['timestamp_open'] = pd.to_datetime(df['timestamp_open'], utc=True)
        df.set_index('timestamp_open', inplace=True)

        return df
