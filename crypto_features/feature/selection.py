"""
Feature selection module.
"""
import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from math import sqrt


class FeatureSelection:
    def __init__(self, **kwargs):
        """
        Feature selection class.
        """
        self._features_df = kwargs.get("features_df", None)
        self._return_arr = kwargs.get("return_arr", None)

    def make_train_test_by_period(self, train_start: datetime.datetime, train_end: datetime.datetime, test_start: datetime.datetime, test_end: datetime.datetime) -> (pd.DataFrame, pd.DataFrame):
        """
        Make train and test sets.

        :param train_start: train start date
        :param train_end: train end date
        :param test_start: test start date
        :param test_end: test end date
        """
        train = self._features_df.loc[train_start:train_end]
        test = self._features_df.loc[test_start:test_end]
        return train, test

    @staticmethod
    def feature_selection_by_lasso(train_x: pd.DataFrame, train_y: np.array, test_x: pd.DataFrame, test_y: np.array, alpha=0.001) -> (list, float):
        """
        Lasso feature selection (Embedded method).

        :param train_x: train x
        :param train_y: train y
        :param test_x: test x
        :param test_y: test y
        :param alpha: alpha (LASSO parameter)
        """
        lasso = Lasso(alpha=alpha)
        lasso.fit(test_x.values, train_y)
        coef = pd.Series(lasso.coef_, index=train_x.columns)
        feature_selection = coef[coef != 0].index.tolist()
        predict = lasso.predict(test_x.values)
        mse = mean_squared_error(test_y, predict)
        rmse = sqrt(mse)
        return feature_selection, rmse
