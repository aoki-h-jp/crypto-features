"""
Preprocessing module for feature engineering.
"""
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


class PreprocessingBinance:
    """
    Preprocessing from binance historical data.
    """

    _BINANCE_KLINES_DIR = os.path.join("data", "futures", "um", "daily", "klines")
    _BINANCE_AGGTRADES_DIR = os.path.join("data", "futures", "um", "daily", "aggTrades")
    _BINANCE_FUNDINGRATE_DIR = os.path.join(
        "data", "futures", "um", "monthly", "fundingRate"
    )
    _BINANCE_LIQUIDATIONSNAPSHOT_DIR = os.path.join(
        "data", "futures", "um", "daily", "liquidationSnapshot"
    )

    def __init__(self, data_dir):
        self._data_dir = data_dir

    def _load_klines_data(self, symbol) -> pd.DataFrame:
        """
        Load klines data from csv files.
        :param symbol: symbol name
        :return: preprocessed klines data
        """
        # Load klines data
        headers = [
            "timestamp_open",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timestamp_close",
            "volume_usdt",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]

        raw_headers = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]

        # merge all csv files
        df = pd.DataFrame(columns=headers)
        for file in os.listdir(
            os.path.join(self._data_dir, self._BINANCE_KLINES_DIR, symbol, "1m")
        ):
            # header check
            df_append_tmp = pd.read_csv(
                "/".join(
                    [
                        self._data_dir,
                        self._BINANCE_KLINES_DIR,
                        symbol,
                        "1m",
                        file,
                    ]
                ),
                nrows=1,
            )

            if list(df_append_tmp) != raw_headers:
                df_append = pd.read_csv(
                    "/".join(
                        [
                            self._data_dir,
                            self._BINANCE_KLINES_DIR,
                            symbol,
                            "1m",
                            file,
                        ]
                    ),
                    names=headers,
                )
            else:
                df_append = pd.read_csv(
                    "/".join(
                        [
                            self._data_dir,
                            self._BINANCE_KLINES_DIR,
                            symbol,
                            "1m",
                            file,
                        ]
                    ),
                    header=None,
                )
                df_append = df_append.drop(0, axis=0)
                df_append.columns = headers

            df = pd.concat([df, df_append])

        df.set_index("timestamp_open", inplace=True)
        df.index = pd.to_datetime(df.index, utc=True, unit="ms")
        df["close"] = df["close"].astype(float)

        return df

    def _load_aggtrades_data(self, symbol) -> pd.DataFrame:
        """
        Load aggTrades data from csv files.
        :param symbol: symbol name
        :return: preprocessed aggTrades data
        """
        # Load aggTrades data
        # merge all csv files
        raw_headers = [
            "agg_trade_id",
            "price",
            "quantity",
            "first_trade_id",
            "last_trade_id",
            "transact_time",
            "is_buyer_maker"
        ]

        headers = [
            "agg_trade_id",
            "price",
            "quantity",
            "first_trade_id",
            "last_trade_id",
            "timestamp_open",
            "is_buyer_maker"
        ]
        df = pd.DataFrame(columns=headers)
        for file in os.listdir(
                os.path.join(self._data_dir, self._BINANCE_AGGTRADES_DIR, symbol)
        ):
            # header check
            df_append_tmp = pd.read_csv(
                "/".join(
                    [
                        self._data_dir,
                        self._BINANCE_AGGTRADES_DIR,
                        symbol,
                        file,
                    ]
                ),
                nrows=1,
            )

            if list(df_append_tmp) != raw_headers:
                df_append = pd.read_csv(
                    "/".join(
                        [
                            self._data_dir,
                            self._BINANCE_AGGTRADES_DIR,
                            symbol,
                            file,
                        ]
                    ),
                    names=headers,
                )
            else:
                df_append = pd.read_csv(
                    "/".join(
                        [
                            self._data_dir,
                            self._BINANCE_AGGTRADES_DIR,
                            symbol,
                            file,
                        ]
                    ),
                    header=None,
                )
                df_append = df_append.drop(0, axis=0)
                df_append.columns = headers

            df = pd.concat([df, df_append])

        df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True, unit="ms")
        df.set_index("timestamp_open", inplace=True)
        df = df.drop_duplicates(keep="first")

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
        for file in os.listdir(
            os.path.join(self._data_dir, self._BINANCE_FUNDINGRATE_DIR, symbol)
        ):
            df = pd.concat(
                [
                    df,
                    pd.read_csv(
                        "/".join(
                            [
                                self._data_dir,
                                self._BINANCE_FUNDINGRATE_DIR,
                                symbol,
                                file,
                            ]
                        )
                    ),
                ]
            )

        df.columns = ["timestamp_open", "interval_time", "funding_rate"]
        df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True, unit="ms")
        df.set_index("timestamp_open", inplace=True)
        df.index = df.index.map(lambda x: x.replace(microsecond=0))
        df = df.drop("interval_time", axis=1)

        return df

    def _load_liquidationsnapshot_data(self, symbol) -> pd.DataFrame:
        """
        Load liquidation data from csv files.
        :param symbol: symbol name
        :return: preprocessed liquidation data
        """
        # Load liquidation data
        # merge all csv files
        df = pd.DataFrame()
        for file in os.listdir(
            os.path.join(self._data_dir, self._BINANCE_LIQUIDATIONSNAPSHOT_DIR, symbol)
        ):
            df = pd.concat(
                [
                    df,
                    pd.read_csv(
                        "/".join(
                            [
                                self._data_dir,
                                self._BINANCE_LIQUIDATIONSNAPSHOT_DIR,
                                symbol,
                                file,
                            ]
                        )
                    ),
                ]
            )

        df.columns = [
            "timestamp_open",
            "side",
            "order_type",
            "time_in_force",
            "original_quantity",
            "price",
            "average_price",
            "order_status",
            "last_fill_quantity",
            "accumulated_fill_quantity",
        ]

        # set index
        df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True, unit="ms")
        df.set_index("timestamp_open", inplace=True)
        df = df.drop_duplicates(keep="first")
        df["amount"] = df["average_price"] * df["original_quantity"]

        return df

    def load_klines_data(self, symbol):
        return self._load_klines_data(symbol)

    def load_aggtrades_data(self, symbol):
        return self._load_aggtrades_data(symbol)

    def load_fundingrate_data(self, symbol):
        return self._load_fundingrate_data(symbol)

    def load_liquidationsnapshot_data(self, symbol):
        return self._load_liquidationsnapshot_data(symbol)


class PreprocessingBybit:
    """
    Preprocessing from bybit historical data.
    """

    _BYBIT_KLINES_DIR = os.path.join("bybit_data", "klines")
    _BYBIT_AGGTRADES_DIR = os.path.join("bybit_data", "trades")
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
        df = pd.read_csv(
            os.path.join(self._data_dir, self._BYBIT_KLINES_DIR, symbol, "1m.csv")
        )
        df.columns = [
            "index",
            "timestamp_open",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "volume_usdt",
            "index2",
        ]
        df = df.drop(["index", "index2"], axis=1)
        df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True)
        df.set_index("timestamp_open", inplace=True)
        df["close"] = df["close"].astype(float)

        return df

    def _load_fundingrate_data(self, symbol) -> pd.DataFrame:
        """
        Load funding rate data from csv files.
        :param symbol: symbol name
        :return: preprocessed funding rate data
        """
        # Load funding rate data
        df = pd.read_csv(
            os.path.join(self._data_dir, self._BYBIT_FUNDINGRATE_DIR, symbol + ".csv")
        )
        df.columns = ["index", "funding_rate", "timestamp_open", "symbol"]
        df = df.drop(["index", "symbol"], axis=1)
        df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True)
        df.index = df.index.map(lambda x: x.replace(microsecond=0))
        df.set_index("timestamp_open", inplace=True)

        return df

    def load_klines_data(self, symbol):
        return self._load_klines_data(symbol)

    def load_fundingrate_data(self, symbol):
        return self._load_fundingrate_data(symbol)
