import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv('Dataset/train.csv')
profile = ProfileReport(df)
profile.to_file(output_file="train.html")

df = pd.read_csv('Dataset/test.csv')
profile = ProfileReport(df)
profile.to_file(output_file="test.html")

df = pd.read_csv('Dataset/customtest.csv')
profile = ProfileReport(df)
profile.to_file(output_file="customtest.html")
