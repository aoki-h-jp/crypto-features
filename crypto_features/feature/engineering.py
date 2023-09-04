"""
Feature engineering module
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto_features.feature.information_correlation import \
    InformationCorrelation


class FeatureEngineering:
    def __init__(self, feature: pd.Series, klines=None):
        """

        :param feature: feature data
        :param klines: klines data
        """
        self._feature = feature
        self._klines = klines

    def set_feature(self, feature: pd.Series):
        """
        Set feature data
        """
        self._feature = feature

    def set_klines(self, klines: pd.DataFrame):
        """
        Set return data
        """
        self._klines = klines

    def _make_return(self, minutes: int) -> pd.Series:
        """
        Make return data

        :param minutes: minutes to calculate return
        """
        return self._klines["close"].pct_change(minutes)

    def visualize_histogram(self, return_minutes: int):
        """
        Visualize histogram of funding rate

        :param return_minutes: minutes to calculate return
        """
        # plot settings
        fig = plt.figure(figsize=(8, 8))
        grid = plt.GridSpec(5, 4, hspace=0.5, wspace=0.5)

        x, y = InformationCorrelation.format_array(
            self._klines, self._feature, return_minutes
        )
        main_ax = fig.add_subplot(grid[1:, 1:])
        main_ax.scatter(x, y, alpha=0.5)
        main_ax.set_xlabel("feature")
        main_ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

        x_hist = fig.add_subplot(grid[0, 1:], sharex=main_ax)
        x_hist.hist(x, bins=50, align="mid", rwidth=0.8)
        x_hist.set_title("feature vs return")
        x_hist.tick_params(bottom=True, labelbottom=True)

        y_hist = fig.add_subplot(grid[1:, 0], sharey=main_ax)
        y_hist.hist(y, bins=50, orientation="horizontal", align="mid", rwidth=0.8)
        y_hist.invert_xaxis()
        y_hist.tick_params(left=True, labelleft=True)
        y_hist.set_ylabel(f"return (after {return_minutes} minutes)")

        plt.tight_layout()
        plt.savefig(f"feature_vs_return_{return_minutes}.png")
        plt.close()

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
