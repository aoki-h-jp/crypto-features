# crypto-features
Consistent from downloading to feature engineering from historical and raw data of cryptocurrency CEXs.

## /download
Download the following data from binance (USDT-M), bybit (linear) in one go.
|              | binance | bybit |
|--------------|---------|-------|
| klines       |      ✅  |   ✅  |
| aggTrades |      ✅  |   TBD  |
| fundingRate  |    ✅    |   ✅  |
| liquidationSnapshot |    ✅    |  ❌  |

## /feature
### preprocessing.py
Perform preprocessing for downloaded data.

### engineering.py
Perform feature engineering for preprocessed data.

### evaluation.py
Evaluate the performance of the engineered features.
