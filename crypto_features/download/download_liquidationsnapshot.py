"""
Download liquidationSnapshot data from binance
"""
from binance_bulk_downloader.downloader import BinanceBulkDownloader


class BinanceLiquidationSnapshotDownload(BinanceBulkDownloader):
    """
    Download liquidationSnapshot data from Binance
    """

    def __init__(self):
        super().__init__(
            destination_dir=".",
            data_type="liquidationSnapshot",
            timeperiod_per_file="daily",
        )
        self.super = super()

    def run_download(self):
        self.super.run_download()
