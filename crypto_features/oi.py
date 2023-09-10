from feature.preprocessing import PreprocessingBinance

p = PreprocessingBinance("E:")
df = p.load_openinterest_data("1000FLOKIUSDT")
print(df)