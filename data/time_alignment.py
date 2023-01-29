import numpy as np
import pandas as pd
from datetime import datetime

df1 = pd.read_csv(r'/Users/noor/taktiles-internet-aip/data/sc1_data_person1_optitrackprocessed.csv')
df2 = pd.read_csv(r'/Users/noor/taktiles-internet-aip/data/sc1_data_person2_optitrackprocessed.csv')

""" df1 = df1.rename(columns={"Time": "Seconds"})
df2 = df2.rename(columns={"Time": "Seconds"})

start = "18:48:03"

ts1 = df1.iloc[:,0] 
print(ts1)
 """