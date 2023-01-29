import numpy as np
import pandas as pd
from datetime import datetime

df1 = pd.read_csv(r'/Users/noor/taktiles-internet-aip/data/sc4_data_person1_preprocessed.csv')
df2 = pd.read_csv(r'/Users/noor/taktiles-internet-aip/data/sc4_data_person2_preprocessed.csv')

# Start our database with timestamp of optitrack
index1 = df1.loc[df1["Time"] == "2023-01-18 19:25:34.080938"]
index1 = int(index1.iloc[:,0])
print(index1)

df1 = df1.iloc[index1:, :]
df2 = df2.iloc[index1:, :]

df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
print(df1)
print(df2)


df1.to_csv('sc4_data_person1_preprocessed.csv')
df2.to_csv('sc4_data_person2_preprocessed.csv')



""" df1 = df1.rename(columns={"Time": "Seconds"})
df2 = df2.rename(columns={"Time": "Seconds"})

start = "18:48:03"

ts1 = df1.iloc[:,0] 
print(ts1)
 """