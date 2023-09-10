"""
Feature engineering module
"""
import datetime
import os

import numpy as np
import pandas as pd
import pytz

from .exceptions import InvalidParameterError


class FeatureEngineering:
    def __init__(self, **kwargs):
        """

        :param feature: feature data (pd.Series)
        :param klines: klines data (pd.DataFrame)
        :param aggtrades: aggTrades data (pd.DataFrame)
        :param liquidationsnapshot: preprocessed liquidationSnapshot data (pd.DataFrame)
        :param start_time: start time of feature data (datetime.datetime)
        :param end_time: end time of feature data (datetime.datetime)
        """
        self._feature: pd.Series = kwargs.get("feature", None)
        self._klines: pd.DataFrame = kwargs.get("klines", None)
        self._aggtrades: pd.DataFrame = kwargs.get("aggtrades", None)
        self._liquidationSnapshot: pd.DataFrame = kwargs.get(
            "liquidationsnapshot", None
        )
        self._start_time: datetime.datetime = kwargs.get("start_time", None)
        self._end_time: datetime.datetime = kwargs.get("end_time", None)
        self._set_utc_timezone()

    def _set_utc_timezone(self):
        """
        Set UTC timezone to start and end time
        """
        self._start_time = self._start_time.replace(tzinfo=pytz.UTC)
        self._end_time = self._end_time.replace(tzinfo=pytz.UTC)

    @staticmethod
    def save_dataframe(df: pd.DataFrame, filename: str):
        """
        Save dataframe to csv
        """
        if not os.path.exists(f"feature_engineering"):
            os.mkdir(f"feature_engineering")
        df.to_csv(f"feature_engineering/{filename}")

    def make_return(
        self, return_minutes: int, save_return=False, filename=None
    ) -> pd.Series:
        """
        Make return data
        # TODO: return_minutes is not used
        # TODO: this method is not working properly

        :param return_minutes: minutes to calculate return
        :param save_return: save return to csv
        :param filename: filename to save return
        """
        if self._check_start_end_time():
            aggtrades = self._parse_aggtrades()
        else:
            aggtrades = self._aggtrades
        index_range = pd.date_range(
            start=self._start_time, end=self._end_time, freq="1S", tz="UTC"
        )
        aggtrades["price"] = aggtrades["price"].astype(float)
        price = aggtrades["price"].resample("1S").last().fillna(method="ffill")
        empty_df = pd.DataFrame(index=index_range)
        empty_df["return"] = price
        empty_df["return"] = empty_df["return"].fillna(0)
        empty_df["return"] = empty_df["return"].astype(float)
        empty_df["return"] = empty_df["return"].round(4)

        save_name = f"return_{self._start_time.date()}_{self._end_time.date()}.csv"
        if filename is not None:
            save_name = save_name.replace(".csv", f"_{filename}.csv")
        if save_return:
            self.save_dataframe(empty_df["return"], save_name)

        return empty_df["return"]

    def make_volatility(self, save_volatility=False, filename=None) -> pd.Series:
        """
        Make volatility data over 24 hours period.

        :param save_volatility: save volatility to csv
        :param filename: filename to save volatility
        :return:
        """
        if self._check_start_end_time():
            aggtrades = self._parse_aggtrades()
        else:
            aggtrades = self._aggtrades
        index_range = pd.date_range(
            start=self._start_time, end=self._end_time, freq="1S", tz="UTC"
        )
        aggtrades["price"] = aggtrades["price"].astype(float)
        price = aggtrades["price"].resample("1S").last().fillna(method="ffill")
        empty_df = pd.DataFrame(index=index_range)
        empty_df["price"] = price
        empty_df["price"] = empty_df["price"].fillna(method="ffill")
        empty_df["price"] = empty_df["price"].astype(float)
        empty_df["price"] = empty_df["price"].round(4)

        def calc_volatility(ts):
            end_time = ts + pd.Timedelta(hours=24)
            subset = empty_df[(empty_df.index > ts) & (empty_df.index <= end_time)]
            min_price = subset["price"].min()
            if min_price == 0:
                return np.nan
            return (subset["price"].max() - subset["price"].min()) / subset[
                "price"
            ].min()

        empty_df["volatility"] = empty_df.index.to_series().apply(calc_volatility)

        save_name = f"volatility_{self._start_time.date()}_{self._end_time.date()}.csv"
        if filename is not None:
            save_name = save_name.replace(".csv", f"_{filename}.csv")
        if save_volatility:
            self.save_dataframe(empty_df["volatility"], save_name)

        return empty_df["volatility"]

    def _parse_feature(self) -> pd.Series:
        """
        Parse feature data
        """
        if self._feature is None:
            raise InvalidParameterError("The feature data is not given.")
        if self._start_time is None:
            raise InvalidParameterError("The start time is not given.")
        if self._end_time is None:
            raise InvalidParameterError("The end time is not given.")
        self._feature = self._feature.loc[self._start_time : self._end_time]
        return self._feature

    def _parse_klines(self) -> pd.DataFrame:
        """
        Parse klines data
        """
        if self._klines is None:
            raise InvalidParameterError("The klines data is not given.")
        if self._start_time is None:
            raise InvalidParameterError("The start time is not given.")
        if self._end_time is None:
            raise InvalidParameterError("The end time is not given.")
        self._klines = self._klines.loc[self._start_time : self._end_time]
        return self._klines

    def _parse_aggtrades(self) -> pd.DataFrame:
        """
        Parse aggtrades data
        """
        if self._aggtrades is None:
            raise InvalidParameterError("The aggtrades data is not given.")
        if self._start_time is None:
            raise InvalidParameterError("The start time is not given.")
        if self._end_time is None:
            raise InvalidParameterError("The end time is not given.")
        self._aggtrades = self._aggtrades.loc[self._start_time : self._end_time]
        return self._aggtrades

    def _parse_liquidationsnapshot(self) -> pd.DataFrame:
        """
        Parse liquidationSnapshot data
        """
        if self._liquidationSnapshot is None:
            raise InvalidParameterError("The liquidationSnapshot data is not given.")
        if self._start_time is None:
            raise InvalidParameterError("The start time is not given.")
        if self._end_time is None:
            raise InvalidParameterError("The end time is not given.")
        self._liquidationSnapshot = self._liquidationSnapshot.loc[
            self._start_time : self._end_time
        ]
        return self._liquidationSnapshot

    def _check_start_end_time(self):
        """
        Check start and end time
        :return: True if start and end time are given, False otherwise
        """
        return self._start_time is not None and self._end_time is not None

    def diff_feature(self) -> pd.Series:
        """
        Calculate difference of feature
        """
        return self._feature.diff()

    def square_feature(self) -> pd.Series:
        """
        Calculate square of feature
        """
        return self._feature**2

    def cube_feature(self) -> pd.Series:
        """
        Calculate cube of feature
        """
        return self._feature**3

    def exp_feature(self) -> pd.Series:
        """
        Calculate exp of feature
        """
        return np.exp(self._feature)

    def sin_feature(self) -> pd.Series:
        """
        Calculate sin of feature
        """
        return np.sin(self._feature)

    def cos_feature(self) -> pd.Series:
        """
        Calculate cos of feature
        """
        return np.cos(self._feature)

    def tan_feature(self) -> pd.Series:
        """
        Calculate tan of feature
        """
        return np.tan(self._feature)

    def tanh_feature(self) -> pd.Series:
        """
        Calculate tanh of feature
        """
        return np.tanh(self._feature)

    def sigmoid_feature(self) -> pd.Series:
        """
        Calculate sigmoid of feature
        """
        return 1 / (1 + np.exp(-self._feature))

    def softmax_feature(self) -> pd.Series:
        """
        Calculate softmax of feature
        """
        return np.exp(self._feature) / np.sum(np.exp(self._feature))

    def log_feature(self) -> pd.Series:
        """
        Calculate log of feature
        """
        return np.log(self._feature)

    def log10_feature(self) -> pd.Series:
        """
        Calculate log10 of feature
        """
        return np.log10(self._feature)

    def log2_feature(self) -> pd.Series:
        """
        Calculate log2 of feature
        """
        return np.log2(self._feature)

    def square_root_feature(self) -> pd.Series:
        """
        Calculate square root of feature
        """
        return np.sqrt(self._feature)

    def arctan_feature(self) -> pd.Series:
        """
        Calculate arctan of feature
        """
        return np.arctan(self._feature)

    def arcsin_feature(self) -> pd.Series:
        """
        Calculate arcsin of feature
        """
        return np.arcsin(self._feature)

    def arccos_feature(self) -> pd.Series:
        """
        Calculate arccos of feature
        """
        return np.arccos(self._feature)

    def arctanh_feature(self) -> pd.Series:
        """
        Calculate arctanh of feature
        """
        return np.arctanh(self._feature)

    def arcsinh_feature(self) -> pd.Series:
        """
        Calculate arcsinh of feature
        """
        return np.arcsinh(self._feature)

    def arccosh_feature(self) -> pd.Series:
        """
        Calculate arccosh of feature
        """
        return np.arccosh(self._feature)

    def absolute_feature(self) -> pd.Series:
        """
        Calculate absolute of feature
        """
        return np.absolute(self._feature)

    def reciprocal_feature(self) -> pd.Series:
        """
        Calculate reciprocal of feature
        """
        return np.reciprocal(self._feature)

    def negative_feature(self) -> pd.Series:
        """
        Calculate negative of feature
        """
        return np.negative(self._feature)

    def sign_feature(self) -> pd.Series:
        """
        Calculate sign of feature
        """
        return np.sign(self._feature)

    def ceil_feature(self) -> pd.Series:
        """
        Calculate ceil of feature
        """
        return np.ceil(self._feature)

    def floor_feature(self) -> pd.Series:
        """
        Calculate floor of feature
        """
        return np.floor(self._feature)

    def rint_feature(self) -> pd.Series:
        """
        Calculate rint of feature
        """
        return np.rint(self._feature)

    def trunc_feature(self) -> pd.Series:
        """
        Calculate trunc of feature
        """
        return np.trunc(self._feature)

    def count_liquidation(
        self, count_minutes: int, save_feature=False, filename=None
    ) -> pd.Series:
        """
        Count number of liquidation

        :param count_minutes: minutes to count liquidation
        :param save_feature: save feature to csv
        :param filename: filename to save feature
        """
        if self._check_start_end_time():
            parsed = self._parse_liquidationsnapshot()
        else:
            parsed = self._liquidationSnapshot

        parsed["trade_count"] = parsed["side"].map({"BUY": 1, "SELL": -1})
        index_range = pd.date_range(
            start=self._start_time, end=self._end_time, freq="1S", tz="UTC"
        )
        empty_df = pd.DataFrame(index=index_range)

        def sum_count(ts):
            start_time = ts - pd.Timedelta(minutes=count_minutes)
            subset = parsed[(parsed.index > start_time) & (parsed.index <= ts)]
            return subset["trade_count"].sum()

        empty_df["count"] = empty_df.index.to_series().apply(sum_count)
        empty_df["count"] = empty_df["count"].fillna(0)
        empty_df["count"] = empty_df["count"].astype(int)

        save_name = (
            f"count_liquidation_{self._start_time.date()}_{self._end_time.date()}.csv"
        )
        if filename is not None:
            save_name = save_name.replace(".csv", f"_{filename}.csv")
        if save_feature:
            self.save_dataframe(empty_df["count"], save_name)

        return empty_df["count"]

    def count_quote_liquidation(
        self, count_minutes: int, save_feature=False, filename=None
    ) -> pd.Series:
        """
        Count number of liquidation per minute

        :param count_minutes: minutes to count liquidation
        :param save_feature: save feature to csv
        :param filename: filename to save feature
        """
        if self._check_start_end_time():
            parsed = self._parse_liquidationsnapshot()
        else:
            parsed = self._liquidationSnapshot

        parsed["amount"] = (
            parsed["price"]
            * parsed["original_quantity"]
            * parsed["side"].map({"BUY": 1, "SELL": -1})
        )
        index_range = pd.date_range(
            start=self._start_time, end=self._end_time, freq="1S", tz="UTC"
        )
        empty_df = pd.DataFrame(index=index_range)

        def sum_amount(ts):
            start_time = ts - pd.Timedelta(minutes=count_minutes)
            subset = parsed[(parsed.index > start_time) & (parsed.index <= ts)]
            return subset["amount"].sum()

        empty_df["taker_buy_quote_volume"] = empty_df.index.to_series().apply(
            sum_amount
        )
        empty_df["taker_buy_quote_volume"] = empty_df["taker_buy_quote_volume"].fillna(
            0
        )
        empty_df["taker_buy_quote_volume"] = empty_df["taker_buy_quote_volume"].astype(
            float
        )

        save_name = f"count_quote_liquidation_{self._start_time.date()}_{self._end_time.date()}.csv"
        if filename is not None:
            save_name = save_name.replace(".csv", f"_{filename}.csv")
        if save_feature:
            self.save_dataframe(empty_df["taker_buy_quote_volume"], save_name)

        return empty_df["taker_buy_quote_volume"]

    def mean_liquidation(
        self, count_minutes: int, save_feature=False, filename=None
    ) -> pd.Series:
        """
        Calculate mean of liquidation

        :param count_minutes: minutes to calculate mean liquidation
        :param save_feature: save feature to csv
        :param filename: filename to save feature
        """
        df = self.count_quote_liquidation(count_minutes).abs() / self.count_liquidation(
            count_minutes
        )
        df = df.fillna(0)
        df = df.resample("1S").mean().fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        save_name = (
            f"mean_liquidation_{self._start_time.date()}_{self._end_time.date()}.csv"
        )
        if filename is not None:
            save_name = save_name.replace(".csv", f"_{filename}.csv")
        if save_feature:
            self.save_dataframe(df, save_name)

        return df

    def ratio_liquidation(
        self, count_minutes: int, save_feature=False, filename=None
    ) -> pd.Series:
        """
        Calculate ratio of liquidation
        count of number of liquidation / count of number of all trades

        :param count_minutes: minutes to calculate ratio liquidation
        :param save_feature: save feature to csv
        :param filename: filename to save feature
        """
        if self._check_start_end_time():
            aggtrades = self._parse_aggtrades()
        else:
            aggtrades = self._aggtrades

        index_range = pd.date_range(
            start=self._start_time, end=self._end_time, freq="1S", tz="UTC"
        )
        empty_df = pd.DataFrame(index=index_range)
        empty_df["aggtrade_count"] = empty_df.index.to_series().apply(
            lambda x: aggtrades[
                (aggtrades.index < x)
                & (aggtrades.index > x - pd.Timedelta(minutes=count_minutes))
            ].shape[0]
        )
        empty_df["aggtrade_count"] = empty_df["aggtrade_count"].fillna(0)
        empty_df["aggtrade_count"] = empty_df["aggtrade_count"].astype(int)
        df = self.count_liquidation(count_minutes) / empty_df["aggtrade_count"]
        df = df.fillna(0)

        save_name = (
            f"ratio_liquidation_{self._start_time.date()}_{self._end_time.date()}.csv"
        )
        if filename is not None:
            save_name = save_name.replace(".csv", f"_{filename}.csv")
        if save_feature:
            self.save_dataframe(df, save_name)

        return df

    def ratio_quote_liquidation(
        self, count_minutes: int, save_feature=False, filename=None
    ) -> pd.Series:
        """
        Calculate ratio of liquidation
        count of liquidation volume by quote / count of trades volume by quote

        :param count_minutes: minutes to calculate ratio liquidation
        :param save_feature: save feature to csv
        :param filename: filename to save feature
        """
        if self._check_start_end_time():
            aggtrades = self._parse_aggtrades()
        else:
            aggtrades = self._aggtrades

        aggtrades["price"] = aggtrades["price"].astype(float)
        aggtrades["quantity"] = aggtrades["quantity"].astype(float)
        aggtrades["amount"] = aggtrades["price"] * aggtrades["quantity"]
        index_range = pd.date_range(
            start=self._start_time, end=self._end_time, freq="1S", tz="UTC"
        )
        empty_df = pd.DataFrame(index=index_range)

        def sum_amount(ts):
            start_time = ts - pd.Timedelta(minutes=count_minutes)
            subset = aggtrades[(aggtrades.index > start_time) & (aggtrades.index <= ts)]
            return subset["amount"].sum()

        empty_df["aggtrade_amount"] = empty_df.index.to_series().apply(sum_amount)
        empty_df["aggtrade_amount"] = empty_df["aggtrade_amount"].fillna(0)
        empty_df["aggtrade_amount"] = empty_df["aggtrade_amount"].astype(float)
        df = (
            self.count_quote_liquidation(count_minutes).abs()
            / empty_df["aggtrade_amount"]
        )
        df = df.fillna(0)

        save_name = f"ratio_quote_liquidation_{self._start_time.date()}_{self._end_time.date()}.csv"
        if filename is not None:
            save_name = save_name.replace(".csv", f"_{filename}.csv")
        if save_feature:
            self.save_dataframe(df, save_name)

        return df
