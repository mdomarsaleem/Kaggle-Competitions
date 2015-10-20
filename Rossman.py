import pandas as pd
import numpy as np
import matplotlib as plt
import pylab as P

df = pd.read_csv("C:/Users/javed/Documents/ROSSMAN/train.csv")
df.head(10)
df.dtypes
df.describe()
df['StateHoliday'].unique()
df.boxplot(column='Sales')
df['Open']
df['Open'].hist()
P.show()

df['Open'].dropna().hist(bins=16, range=(0,2), alpha = .5)
P.show()

print df