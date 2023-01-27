import numpy as np
import pandas as pd


df2 = pd.read_csv(r'/Users/noor/taktiles-internet-aip/data/sc1_data_person1_True.csv')

sec = df2.iloc[:,1]
sec = sec.rename("Seconds")
df2.insert(1,"Seconds",sec)
vec = df2.iloc[:,3]

counter = 0

#print(type(vec))

for i in range(len(vec)): 
    vec_row = vec[i]
    vec_row = vec_row[1:-1]
    vec_split = vec_row.split()
    vec[i] = vec_split
    counter += 1

print(vec[58])
df_angle_pos = pd.DataFrame(data=vec.to_list(), columns=['Pitch','Roll', 'Yaw', 'X-Pos', 'Y-Pos', 'Z-Pos'])

df2[['Pitch','Roll', 'Yaw', 'X-Pos', 'Y-Pos', 'Z-Pos']] = df_angle_pos
print(df2.iloc[58])