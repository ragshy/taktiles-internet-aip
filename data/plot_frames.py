import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv(r'/Users/noor/taktiles-internet-aip/data/sc1_data_person1_preprocessed.csv')
df2 = pd.read_csv(r'/Users/noor/taktiles-internet-aip/data/sc1_data_person2_preprocessed.csv')



plt.plot(df1['Seconds'], df1['Pitch'])
plt.show()
