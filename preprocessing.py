import pandas as pd
from datetime import timedelta
import numpy as np
df = pd.read_csv('./data/0.125_noise.csv')
df = df.drop(columns=['Variant', 'Variant index', 'lifecycle:transition'])

df = df.rename(columns={'Complete Timestamp':'Timestamp'})
print(df.columns.values)


df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values(by='Timestamp')

groups = df.groupby('Case ID')
concating = []
for _, group in groups:
    group = group.reset_index(drop=True)
    group.loc[-1] = [_, 'Start', list(group['Timestamp'])[0] - timedelta(seconds=1), 'Start']
    group.index = group.index+1
    group.sort_index(inplace=True)
    group.loc[-1] = [_, 'End', list(group['Timestamp'])[-1] + timedelta(seconds=1), 'End']

    concating.append(group)

dfs = pd.concat(concating)
dfs = dfs.sort_values(by='Timestamp')
dfs.to_csv('./data/0.125_noise.csv', index=False)