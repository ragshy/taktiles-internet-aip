import numpy as np
import pandas as pd
import datetime

df1 = pd.read_csv(r'/Users/noor/taktiles-internet-aip/data/cam_data_raw/sc4_data_person1_True.csv')
df2 = pd.read_csv(r'/Users/noor/taktiles-internet-aip/data/cam_data_raw/sc4_data_person2_True.csv')

# Scene 4 weird data error, drop first sample/row
dim = len(df1)
df1 = df1.drop(0)
df2 = df2.drop(0)

df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)

sec1 = df1.iloc[:,1]
sec1 = sec1.rename("Seconds")
df1.insert(1,"Seconds",sec1)
vec1 = df1.iloc[:,3]

sec2 = df2.iloc[:,1]
sec2 = sec2.rename("Seconds")
df2.insert(1,"Seconds",sec2)
vec2 = df2.iloc[:,3]

counter = 0

#print(type(vec))

for i in range(len(vec1)): 
    vec_row = vec1[i]
    vec_row = vec_row[1:-1]
    vec_split1 = vec_row.split()
    vec1[i] = vec_split1
    counter += 1

for i in range(len(vec2)): 
    vec_row = vec2[i]
    vec_row = vec_row[1:-1]
    vec_split2 = vec_row.split()
    vec2[i] = vec_split2
    counter += 1

df_angle_pos1 = pd.DataFrame(data=vec1.to_list(), columns=['Pitch','Roll', 'Yaw', 'X-Pos', 'Y-Pos', 'Z-Pos'])
df_angle_pos2 = pd.DataFrame(data=vec2.to_list(), columns=['Pitch','Roll', 'Yaw', 'X-Pos', 'Y-Pos', 'Z-Pos'])

df1[['Pitch','Roll', 'Yaw', 'X-Pos', 'Y-Pos', 'Z-Pos']] = df_angle_pos1
df2[['Pitch','Roll', 'Yaw', 'X-Pos', 'Y-Pos', 'Z-Pos']] = df_angle_pos2

#print(df1)
#print(df2)

df1.to_csv('sc4_data_person1_preprocessed.csv')
df2.to_csv('sc4_data_person2_preprocessed.csv')

""" #timestamp = df1.loc[5,"Time"]
timestamp = '2023-01-18 19:25:15.713018'
ts_conv = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S:%f')
print(type(timestamp))
print(timestamp) """