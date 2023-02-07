import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv(r'/Users/noor/taktiles-internet-aip/data/sc1_data_person1_preprocessed.csv')
df2 = pd.read_csv(r'/Users/noor/taktiles-internet-aip/data/sc3_data_person2_preprocessed.csv')
dfcam = pd.read_csv(r'/Users/noor/taktiles-internet-aip/data/sc1_data_person1_optitrackprocessed.csv')

# sc1 person1 für Pitch
time_cam =  df1.loc[11:110,'Seconds'] - 29.42
time_opti = dfcam.loc[444:5672,'Time'] - 1.233
action_cam = df1.iloc[11:111,4]             
action_opti = dfcam.iloc[444:5673,1] - 74

# sc2 person2 für rechts links
""" time_cam =  df2.loc[237:277,'Seconds'] - 67.04
time_opti = dfcam.loc[12187:14239,'Time'] 
action_cam = df2.iloc[237:278,7]                     # X-Pos
action_opti = dfcam.iloc[12187:14240,4] /10          # X-Pos """

# sc3 person2 für vor hinter
""" time_cam =  df2.loc[165:229,'Seconds'] - 25.93
time_opti = dfcam.loc[8117:11675,'Time'] 
action_cam = df2.iloc[165:230,9]                         # Z-Pos
#action_opti = (dfcam.iloc[8117:11676,5] -60)  /10       # Y-Pos 
action_opti = dfcam.iloc[8117:11676,5]   """


print(action_cam)
print(action_opti)
print(time_cam)
print(time_opti)


plt.plot(time_opti, action_opti, label = 'Optitrack')
plt.plot(time_cam, action_cam, label = 'Webcam-Setup')
plt.xlabel("Time [s]")
plt.ylabel("Position [cm]")
plt.title('Speaker 2 Back and Forth Comparison')
plt.legend()
plt.show()

