# crypto-features
Consistent from downloading to feature engineering from historical and raw data of cryptocurrency CEXs.

## /download
Download the following data from binance (USDT-M), bybit (linear) in one go.
|              | binance | bybit |
|--------------|---------|-------|
| klines       |      ✅  |   ✅  |
| fundingRate  |    ✅    |   ✅  |
| liquidationSnapshot |    ✅    |  ❌  |

## /feature
### preprocessing
Perform preprocessing for downloaded data.

### information_correlation
Calculate and Visualize information correlation (IC). High IC (>0.05) indicates an important feature.

### engineering
Perform feature engineering for preprocessed data.

### evaluation
Evaluate the performance of the engineered features.
