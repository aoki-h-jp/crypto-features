from crypto_features.download.download_klines import (BinanceKlinesDownload,
                                                      BybitKlinesDownload)

# download klines data from Binance and Bybit
BinanceKlinesDownload().run_download()
BybitKlinesDownload().run_download()
