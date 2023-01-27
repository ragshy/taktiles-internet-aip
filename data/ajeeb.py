import numpy as np
import pandas as pd

df1 = pd.read_csv(r'/Users/noor/taktiles-internet-aip/data/cam_data_raw/sc4_data_person1_True.csv', )
df2 = pd.read_csv(r'/Users/noor/taktiles-internet-aip/data/cam_data_raw/sc4_data_person2_True.csv')

dim = len(df1)
df1 = df1.drop(0)
df2 = df2.drop(0)

df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
print(df1)
print(df2)