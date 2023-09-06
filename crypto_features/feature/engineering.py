"""
Feature engineering module
"""
import numpy as np
import pandas as pd


class FeatureEngineering:
    def __init__(self, **kwargs):
        """

        :param feature: feature data
        :param klines: klines data
        :param liquidationsnapshot: preprocessed liquidationSnapshot data (pd.DataFrame)
        """
        self._feature = kwargs.get("feature", None)
        self._klines = kwargs.get("klines", None)
        self._liquidationSnapshot = kwargs.get("liquidationsnapshot", None)

    def diff_feature(self) -> pd.Series:
        """
        Calculate difference of funding rate
        """
        return self._feature.diff()

    def square_feature(self) -> pd.Series:
        """
        Calculate square of funding rate
        """
        return self._feature**2

    def cube_feature(self) -> pd.Series:
        """
        Calculate cube of funding rate
        """
        return self._feature**3

    def exp_feature(self) -> pd.Series:
        """
        Calculate exp of funding rate
        """
        return np.exp(self._feature)

    def sin_feature(self) -> pd.Series:
        """
        Calculate sin of funding rate
        """
        return np.sin(self._feature)

    def cos_feature(self) -> pd.Series:
        """
        Calculate cos of funding rate
        """
        return np.cos(self._feature)

    def tan_feature(self) -> pd.Series:
        """
        Calculate tan of funding rate
        """
        return np.tan(self._feature)

    def tanh_feature(self) -> pd.Series:
        """
        Calculate tanh of funding rate
        """
        return np.tanh(self._feature)

    def sigmoid_feature(self) -> pd.Series:
        """
        Calculate sigmoid of funding rate
        """
        return 1 / (1 + np.exp(-self._feature))

    def softmax_feature(self) -> pd.Series:
        """
        Calculate softmax of funding rate
        """
        return np.exp(self._feature) / np.sum(np.exp(self._feature))

    def log_feature(self) -> pd.Series:
        """
        Calculate log of funding rate
        """
        return np.log(self._feature)

    def log10_feature(self) -> pd.Series:
        """
        Calculate log10 of funding rate
        """
        return np.log10(self._feature)

    def log2_feature(self) -> pd.Series:
        """
        Calculate log2 of funding rate
        """
        return np.log2(self._feature)

    def square_root_feature(self) -> pd.Series:
        """
        Calculate square root of funding rate
        """
        return np.sqrt(self._feature)

    def arctan_feature(self) -> pd.Series:
        """
        Calculate arctan of funding rate
        """
        return np.arctan(self._feature)

    def arcsin_feature(self) -> pd.Series:
        """
        Calculate arcsin of funding rate
        """
        return np.arcsin(self._feature)

    def arccos_feature(self) -> pd.Series:
        """
        Calculate arccos of funding rate
        """
        return np.arccos(self._feature)

    def arctanh_feature(self) -> pd.Series:
        """
        Calculate arctanh of funding rate
        """
        return np.arctanh(self._feature)

    def arcsinh_feature(self) -> pd.Series:
        """
        Calculate arcsinh of funding rate
        """
        return np.arcsinh(self._feature)

    def arccosh_feature(self) -> pd.Series:
        """
        Calculate arccosh of funding rate
        """
        return np.arccosh(self._feature)

    def absolute_feature(self) -> pd.Series:
        """
        Calculate absolute of funding rate
        """
        return np.absolute(self._feature)

    def reciprocal_feature(self) -> pd.Series:
        """
        Calculate reciprocal of funding rate
        """
        return np.reciprocal(self._feature)

    def negative_feature(self) -> pd.Series:
        """
        Calculate negative of funding rate
        """
        return np.negative(self._feature)

    def sign_feature(self) -> pd.Series:
        """
        Calculate sign of funding rate
        """
        return np.sign(self._feature)

    def ceil_feature(self) -> pd.Series:
        """
        Calculate ceil of funding rate
        """
        return np.ceil(self._feature)

    def floor_feature(self) -> pd.Series:
        """
        Calculate floor of funding rate
        """
        return np.floor(self._feature)

    def rint_feature(self) -> pd.Series:
        """
        Calculate rint of funding rate
        """
        return np.rint(self._feature)

    def trunc_feature(self) -> pd.Series:
        """
        Calculate trunc of funding rate
        """
        return np.trunc(self._feature)

    def count_liquidation(self, count_minutes: int) -> pd.Series:
        """
        Count number of liquidation

        :param count_minutes: minutes to count liquidation
        """

        def count_buy_sell_in_last_1min(current_time, df):
            subset = df[(df.index < current_time) & (df.index >= current_time - pd.Timedelta(minutes=count_minutes))]
            buy_count = subset[subset["side"] == "BUY"].shape[0]
            sell_count = subset[subset["side"] == "SELL"].shape[0]
            return buy_count - sell_count

        return self._liquidationSnapshot.index.to_series().apply(
            lambda x: count_buy_sell_in_last_1min(x, self._liquidationSnapshot)
        )

    def count_quote_liquidation(self, count_minutes: int) -> pd.Series:
        """
        Count number of liquidation per minute

        :param count_minutes: minutes to count liquidation
        """
        self._liquidationSnapshot["amount"] = (
            self._liquidationSnapshot["price"]
            * self._liquidationSnapshot["original_quantity"]
        )
        times = self._liquidationSnapshot.index.values
        amounts = self._liquidationSnapshot["amount"].values
        sides = self._liquidationSnapshot["side"].values

        def np_adjusted_sum_amount_in_last_1min(idx):
            current_time = times[idx]
            mask = (times < current_time) & (
                times >= current_time - pd.Timedelta(minutes=count_minutes)
            )
            adjusted_amounts = np.where(
                sides[mask] == "BUY", amounts[mask], -amounts[mask]
            )
            return adjusted_amounts.sum()

        se = pd.Series(
            np.array(
                [
                    np_adjusted_sum_amount_in_last_1min(i)
                    for i in range(len(self._liquidationSnapshot))
                ]
            ),
            index=times,
        )
        return se

    def mean_liquidation(self, count_minutes: int) -> pd.Series:
        """
        Calculate mean of liquidation

        :param count_minutes: minutes to calculate mean liquidation
        """
        df = self.count_quote_liquidation(count_minutes) / self.count_liquidation(
            count_minutes
        )
        df = df.fillna(0)
        return df

    def ratio_liquidation(self, count_minutes: int) -> pd.Series:
        """
        Calculate ratio of liquidation
        count of number of liquidation / count of number of all trades

        :param count_minutes: minutes to calculate ratio liquidation
        """
        self._klines["count"] = self._klines["count"].astype(int)
        return self.count_liquidation(
            count_minutes
        ) / self._liquidationSnapshot.index.to_series().apply(
            lambda x: self._klines[
                (self._klines.index < x)
                & (self._klines.index > x - pd.Timedelta(minutes=1))
            ]["count"].sum()
        )

    def ratio_quote_liquidation(self, count_minutes: int) -> pd.Series:
        """
        Calculate ratio of liquidation
        count of liquidation volume by quote / count of trades volume by quote

        :param count_minutes: minutes to calculate ratio liquidation
        """
        self._klines["taker_buy_quote_volume"] = self._klines[
            "taker_buy_quote_volume"
        ].astype(float)
        return self.count_quote_liquidation(
            count_minutes
        ) / self._liquidationSnapshot.index.to_series().apply(
            lambda x: self._klines[
                (self._klines.index < x)
                & (self._klines.index > x - pd.Timedelta(minutes=1))
            ]["taker_buy_quote_volume"].sum()
        )
