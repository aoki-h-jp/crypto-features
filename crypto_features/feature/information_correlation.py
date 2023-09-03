"""
Calculate the information correlation.
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
from rich import print


class InformationCorrelation:
    """
    Calculate the information correlation.
    """

    def __init__(self):
        pass

    @staticmethod
    def run_calculate(klines: pd.DataFrame, feature: pd.Series, return_minutes=1, **kwargs):
        """
        Calculate and visualize the information correlation.

        :param klines: The klines data.
        :param feature: The feature data.
        :param return_minutes: The return minutes.
        """
        if not os.path.exists("information_correlation"):
            os.mkdir("information_correlation")

        klines = klines.copy()
        feature = feature.copy()

        close_chg_pct_header = f"close_chg_pct_after_{return_minutes}min"
        klines[close_chg_pct_header] = klines["close"].pct_change(return_minutes)
        klines[close_chg_pct_header] = klines[close_chg_pct_header].shift(-return_minutes)
        klines[close_chg_pct_header] = klines[close_chg_pct_header].fillna(0)
        klines[close_chg_pct_header] = klines[close_chg_pct_header].replace([np.inf, -np.inf, np.nan, -np.nan], 0)
        klines[close_chg_pct_header] = klines[close_chg_pct_header].astype(float)
        klines[close_chg_pct_header] = klines[close_chg_pct_header].round(4)

        feature.index = feature.index.map(lambda x: x.replace(microsecond=0))
        feature_arr = feature[feature.index.isin(klines.index)].values
        klines_arr = klines[klines.index.isin(feature.index)][close_chg_pct_header].values

        assert len(feature_arr) == len(klines_arr), f"len(feature_arr)={len(feature_arr)}, len(klines_arr)={len(klines_arr)}"

        print("[green] Start calculating the information correlation... [/green]")

        # Pearson's correlation coefficient
        rho, pval = pearsonr(feature_arr, klines_arr)
        print(f"rho={rho}, pval={pval}")

        lr = LinearRegression()
        lr.fit(feature_arr.reshape(-1, 1), klines_arr.reshape(-1, 1))
        print(f"coef={lr.coef_[0][0]}, intercept={lr.intercept_[0]}")

        # Visualize
        plt.scatter(feature_arr, klines_arr * 100)
        plt.plot(feature_arr, lr.predict(feature_arr.reshape(-1, 1)) * 100, color='red', linewidth=1, linestyle='-.')
        plt.xlabel(feature.name)
        plt.ylabel("close_chg_pct [%]")
        plt.title(f"rho={round(rho, 3)}, pval={round(pval, 3)}\ncoef={round(lr.coef_[0][0], 3)}, intercept={round(lr.intercept_[0], 3)}\n{feature.name} vs close_chg_pct_after_{return_minutes}min")
        plt.tight_layout()
        save_dir = f"information_correlation/{feature.name}_vs_close_chg_pct_after_{return_minutes}min.png"
        if kwargs.get("save_name", False):
            save_dir = save_dir.replace(".png", f"_{kwargs['save_name']}.png")
        plt.savefig(save_dir)
        print(f"Saved: {save_dir}")
        plt.close()
