"""
Calculate the information correlation.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


class InformationCorrelation:
    """
    Calculate the information correlation.
    """

    def __init__(self):
        pass

    @staticmethod
    def format_array(klines: pd.DataFrame, feature: pd.Series, return_minutes=1):
        """
        Format the array.
        :param klines: The klines data.
        :param feature: The feature data.
        :param return_minutes: The return minutes.
        :return: formatted feature and return array.
        """
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

    @staticmethod
    def run_calculate(
        klines: pd.DataFrame, feature: pd.Series, return_minutes=1, **kwargs
    ):
        """
        Calculate and visualize the information correlation.

        :param klines: The klines data.
        :param feature: The feature data.
        :param return_minutes: The return minutes.
        """
        if not os.path.exists("information_correlation"):
            os.mkdir("information_correlation")

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
