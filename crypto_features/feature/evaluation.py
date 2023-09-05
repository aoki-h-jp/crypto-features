"""
Evaluation of the features
"""
import matplotlib.pyplot as plt
import pandas as pd
from .information_correlation import InformationCorrelation


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
