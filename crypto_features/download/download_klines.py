"""
Download klines from CEXs
"""
from binance_bulk_downloader.downloader import BinanceBulkDownloader
from bybit_bulk_downloader.downloader import BybitBulkDownloader


class BinanceKlinesDownload(BinanceBulkDownloader):
    """
    Download klines from Binance
    """

    def __init__(self):
        super().__init__(
            destination_dir=".", data_type="klines", timeperiod_per_file="daily"
        )
        self.super = super()

    def run_download(self):
        self.super.run_download()


class BybitKlinesDownload(BybitBulkDownloader):
    """
    Download klines from Bybit
    """
    def __init__(self, symbol="BTCUSDT"):
        super().__init__(
            destination_dir=".", data_type="klines"
        )
        self.super = super()
        self._symbol = symbol

    def run_download(self):
        self.super.run_download()

    def run_download_single_symbol(self):
        self.super.download_klines(self._symbol)
