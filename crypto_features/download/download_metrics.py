"""
Download metrics data from binance
"""
from binance_bulk_downloader.downloader import BinanceBulkDownloader


class BinanceMetricsDownload(BinanceBulkDownloader):
    """
    Download metrics data from Binance
    """

    def __init__(self):
        super().__init__(
            destination_dir=".", data_type="metrics", timeperiod_per_file="daily"
        )
        self.super = super()

    def run_download(self):
        self.super.run_download()
