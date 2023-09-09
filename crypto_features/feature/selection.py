"""
Feature selection module.
"""
import datetime
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


class FeatureSelection:
    def __init__(self, **kwargs):
        """
        Feature selection class.
        """
        self._features_df: pd.DataFrame = kwargs.get("features_df", None)
        self._return_df: pd.DataFrame = kwargs.get("return_df", None)

    def make_train_test_by_period(
        self,
        train_start: datetime.datetime,
        test_start: datetime.datetime,
        test_end: datetime.datetime,
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Make train and test sets.

        :param train_start: train start date
        :param test_start: test start date
        :param test_end: test end date
        """
        train_x = self._features_df.loc[train_start:test_start]
        test_x = self._features_df.loc[test_start:test_end]
        train_y = self._return_df.loc[train_start:test_start]
        test_y = self._return_df.loc[test_start:test_end]
        return train_x, test_x, train_y, test_y

    @staticmethod
    def feature_selection_by_lasso(
        train_x: pd.DataFrame,
        train_y: pd.DataFrame,
        test_x: pd.DataFrame,
        test_y: pd.DataFrame,
        alpha=0.001,
    ) -> (list, float):
        """
        Lasso feature selection (Embedded method).

        :param train_x: train x
        :param train_y: train y
        :param test_x: test x
        :param test_y: test y
        :param alpha: alpha (LASSO parameter)
        """
        sc = StandardScaler()
        train_x_arr = sc.fit_transform(train_x.values)
        test_x_arr = sc.fit_transform(test_x.values)
        lasso = Lasso(alpha=alpha)
        lasso.fit(train_x_arr, train_y.values)
        coef = pd.Series(lasso.coef_, index=train_x.columns)
        print(coef)
        feature_selection = coef[coef != 0].index.tolist()
        predict = lasso.predict(test_x_arr)
        mse = mean_squared_error(test_y.values, predict)
        rmse = sqrt(mse)
        return feature_selection, rmse
