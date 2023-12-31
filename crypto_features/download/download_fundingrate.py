"""
Download funding rate data from CEXs
"""
from binance_bulk_downloader.downloader import BinanceBulkDownloader
from bybit_bulk_downloader.downloader import BybitBulkDownloader


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


class BybitFundingRateDownload(BybitBulkDownloader):
    """
    Download funding rate data from Bybit
    """

    def __init__(self):
        super().__init__(destination_dir=".", data_type="fundingRate")
        self.super = super()

    def run_download(self):
        self.super.run_download()
