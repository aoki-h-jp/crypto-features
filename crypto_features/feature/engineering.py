"""
Feature engineering module
"""
import datetime

import numpy as np
import pandas as pd
from .exceptions import DataNotFoundError, InsufficientDataError, InvalidParameterError


class FeatureEngineering:
    def __init__(self, **kwargs):
        """

        :param feature: feature data (pd.Series)
        :param klines: klines data (pd.DataFrame)
        :param liquidationsnapshot: preprocessed liquidationSnapshot data (pd.DataFrame)
        :param start_time: start time of feature data (datetime.datetime)
        :param end_time: end time of feature data (datetime.datetime)
        """
        self._feature: pd.Series = kwargs.get("feature", None)
        self._klines: pd.DataFrame = kwargs.get("klines", None)
        self._liquidationSnapshot: pd.DataFrame = kwargs.get("liquidationsnapshot", None)
        self._start_time: datetime.datetime = kwargs.get("start_time", None)
        self._end_time: datetime.datetime = kwargs.get("end_time", None)

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

    def count_liquidation(self, count_minutes: int) -> pd.Series:
        """
        Count number of liquidation

        :param count_minutes: minutes to count liquidation
        """
        if self._start_time is not None and self._end_time is not None:
            parsed = self._parse_liquidationsnapshot()
        else:
            parsed = self._liquidationSnapshot

        parsed['trade_count'] = parsed['side'].map({'BUY': 1, 'SELL': -1})
        index_range = pd.date_range(start=self._start_time, end=self._end_time, freq="1S", tz="UTC")
        empty_df = pd.DataFrame(index=index_range)

        def sum_count(ts):
            start_time = ts - pd.Timedelta(minutes=count_minutes)
            subset = parsed[(parsed.index > start_time) & (parsed.index <= ts)]
            return subset["trade_count"].sum()

        empty_df["count"] = empty_df.index.to_series().apply(sum_count)
        empty_df["count"] = empty_df["count"].fillna(0)
        empty_df["count"] = empty_df["count"].astype(int)

        return empty_df["count"]

    def count_quote_liquidation(self, count_minutes: int) -> pd.Series:
        """
        Count number of liquidation per minute

        :param count_minutes: minutes to count liquidation
        """
        if self._start_time is not None and self._end_time is not None:
            parsed = self._parse_liquidationsnapshot()
        else:
            parsed = self._liquidationSnapshot

        parsed["amount"] = parsed["price"] * parsed["original_quantity"] * parsed['side'].map({'BUY': 1, 'SELL': -1})
        index_range = pd.date_range(start=self._start_time, end=self._end_time, freq="1S", tz="UTC")
        empty_df = pd.DataFrame(index=index_range)

        def sum_amount(ts):
            start_time = ts - pd.Timedelta(minutes=count_minutes)
            subset = parsed[(parsed.index > start_time) & (parsed.index <= ts)]
            return subset["amount"].sum()

        empty_df["taker_buy_quote_volume"] = empty_df.index.to_series().apply(sum_amount)
        empty_df["taker_buy_quote_volume"] = empty_df["taker_buy_quote_volume"].fillna(0)
        empty_df["taker_buy_quote_volume"] = empty_df["taker_buy_quote_volume"].astype(float)

        return empty_df["taker_buy_quote_volume"]

    def mean_liquidation(self, count_minutes: int) -> pd.Series:
        """
        Calculate mean of liquidation

        :param count_minutes: minutes to calculate mean liquidation
        """
        df = self.count_quote_liquidation(count_minutes) / self.count_liquidation(
            count_minutes
        )
        df = df.fillna(0)
        return df.resample("1S").mean().fillna(0)

    def ratio_liquidation(self, count_minutes: int) -> pd.Series:
        """
        Calculate ratio of liquidation
        count of number of liquidation / count of number of all trades

        :param count_minutes: minutes to calculate ratio liquidation
        """
        self._klines["count"] = self._klines["count"].astype(int)
        se = self.count_liquidation(
            count_minutes
        ) / self._liquidationSnapshot.index.to_series().apply(
            lambda x: self._klines[
                (self._klines.index < x)
                & (self._klines.index > x - pd.Timedelta(minutes=count_minutes))
            ]["count"].sum()
        )
        se = se.fillna(0)
        se = se.replace([np.inf, -np.inf], 0)
        return se.resample("1S").mean().fillna(0)

    def ratio_quote_liquidation(self, count_minutes: int) -> pd.Series:
        """
        Calculate ratio of liquidation
        count of liquidation volume by quote / count of trades volume by quote

        :param count_minutes: minutes to calculate ratio liquidation
        """
        self._klines["taker_buy_quote_volume"] = self._klines[
            "taker_buy_quote_volume"
        ].astype(float)
        se = self.count_quote_liquidation(
            count_minutes
        ) / self._liquidationSnapshot.index.to_series().apply(
            lambda x: self._klines[
                (self._klines.index < x)
                & (self._klines.index > x - pd.Timedelta(minutes=count_minutes))
            ]["taker_buy_quote_volume"].sum()
        )
        se = se.fillna(0)
        se = se.replace([np.inf, -np.inf], 0)
        return se.resample("1S").mean().fillna(0)
