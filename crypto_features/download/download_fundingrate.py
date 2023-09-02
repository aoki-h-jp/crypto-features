"""
Download funding rate data from CEXs
"""
import os
from datetime import datetime, timedelta

import pandas as pd
from binance_bulk_downloader.downloader import BinanceBulkDownloader
from pybit.unified_trading import HTTP
from rich.progress import track


class BinanceFundingRateDownload(BinanceBulkDownloader):
    """
    Download funding rate data from Binance
    """

    def __init__(self):
        super().__init__(
            destination_dir=".", data_type="fundingRate", timeperiod_per_file="monthly"
        )
        self.super = super()

    def run_download(self):
        self.super.run_download()


class BybitFundingRateDownload:
    """
    Download funding rate data from Bybit
    """

    def __init__(self):
        if not os.path.exists("bybit_fundingrate"):
            os.mkdir("bybit_fundingrate")

        self.session = HTTP()

    @staticmethod
    def generate_dates_until_today(start_year, start_month) -> list:
        """
        Generate dates until today
        :param start_year:
        :param start_month:
        :return: list of dates
        """
        start_date = datetime(start_year, start_month, 1)
        end_date = datetime.today()

        output = []
        while start_date <= end_date:
            next_date = start_date + timedelta(days=60)  # Roughly two months
            if next_date > end_date:
                next_date = end_date
            output.append(
                f"{start_date.strftime('%Y-%m-%d')} {next_date.strftime('%Y-%m-%d')}"
            )
            start_date = next_date + timedelta(days=1)

        return output

    def run_download(self):
        """
        Download funding rate data from Bybit
        """
        s_list = [
            d["symbol"]
            for d in self.session.get_tickers(category="linear")["result"]["list"]
            if d["symbol"][-4:] == "USDT"
        ]
        # Get all available symbols
        for sym in track(
            s_list, description="Downloading funding rate data from Bybit"
        ):
            # Get funding rate history
            df = pd.DataFrame(columns=["fundingRate", "fundingRateTimestamp", "symbol"])
            for dt in self.generate_dates_until_today(2021, 1):
                start_time, end_time = dt.split(" ")
                # Convert to timestamp (ms)
                start_time = int(
                    datetime.strptime(start_time, "%Y-%m-%d").timestamp() * 1000
                )
                end_time = int(
                    datetime.strptime(end_time, "%Y-%m-%d").timestamp() * 1000
                )
                for d in self.session.get_funding_rate_history(
                    category="linear",
                    symbol=sym,
                    limit=200,
                    startTime=start_time,
                    endTime=end_time,
                )["result"]["list"]:
                    df.loc[len(df)] = d

            df["fundingRateTimestamp"] = pd.to_datetime(
                df["fundingRateTimestamp"].astype(float) * 1000000
            )
            df["fundingRate"] = df["fundingRate"].astype(float)
            df = df.sort_values("fundingRateTimestamp")

            # Save to csv
            df.to_csv(f"bybit_fundingrate/{sym}.csv")
