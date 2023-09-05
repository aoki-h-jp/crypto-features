"""
Evaluation of the features
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


class EvaluationFeature:
    def __init__(self, **kwargs):
        self._feature = kwargs.get("feature", None)
        self._klines = kwargs.get("klines", None)

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

    def format_array(self, return_minutes=1):
        """
        Format the array.
        :param return_minutes: The return minutes.
        :return: formatted feature and return array.
        """
        klines = self._klines
        feature = self._feature

        close_chg_pct_header = f"close_chg_pct_after_{return_minutes}min"
        klines["close"] = klines["close"].astype(float)
        klines[close_chg_pct_header] = klines["close"].pct_change(
            return_minutes, fill_method="bfill"
        )
        klines[close_chg_pct_header] = klines[close_chg_pct_header].shift(
            -return_minutes
        )
        klines[close_chg_pct_header] = klines[close_chg_pct_header].fillna(0)
        klines[close_chg_pct_header] = klines[close_chg_pct_header].replace(
            [np.inf, -np.inf, np.nan, -np.nan], 0
        )
        klines[close_chg_pct_header] = klines[close_chg_pct_header].astype(float)
        klines[close_chg_pct_header] = klines[close_chg_pct_header].round(4)

        feature_arr = feature[feature.index.isin(klines.index)].values
        return_arr = klines[klines.index.isin(feature.index)][
            close_chg_pct_header
        ].values

        assert len(feature_arr) == len(
            return_arr
        ), f"len(feature_arr)={len(feature_arr)}, len(return_arr)={len(return_arr)}"

        return feature_arr, return_arr

    def information_correlation(self, return_minutes=1, **kwargs):
        """
        Calculate and visualize the information correlation.

        :param return_minutes: The return minutes.
        """
        if not os.path.exists("information_correlation"):
            os.mkdir("information_correlation")

        klines = self._klines
        feature = self._feature

        feature_arr, klines_arr = InformationCorrelation.format_array(
            klines, feature, return_minutes
        )
        print("[green] Start calculating the information correlation... [/green]")

        # Pearson's correlation coefficient
        rho, pval = pearsonr(feature_arr, klines_arr)
        print(f"rho={rho}, pval={pval}")

        lr = LinearRegression()
        lr.fit(feature_arr.reshape(-1, 1), klines_arr.reshape(-1, 1))
        print(f"coef={lr.coef_[0][0]}, intercept={lr.intercept_[0]}")

        # Visualize
        plt.scatter(feature_arr, klines_arr * 100)
        plt.plot(
            feature_arr,
            lr.predict(feature_arr.reshape(-1, 1)) * 100,
            color="red",
            linewidth=1,
            linestyle="-.",
        )
        plt.xlabel(feature.name)
        plt.ylabel(f"close_chg_pct_after_{return_minutes}min [%]")
        plt.title(
            f"rho={round(rho, 3)}, pval={round(pval, 3)}\ncoef={round(lr.coef_[0][0], 3)}, intercept={round(lr.intercept_[0], 3)}\n{feature.name} vs close_chg_pct_after_{return_minutes}min"
        )
        plt.tight_layout()
        save_dir = f"information_correlation/{feature.name}_vs_close_chg_pct_after_{return_minutes}min.png"
        if kwargs.get("save_name", False):
            save_dir = save_dir.replace(".png", f"_{kwargs['save_name']}.png")
        plt.savefig(save_dir)
        print(f"Saved: {save_dir}")
        plt.close()
