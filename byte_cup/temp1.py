import pandas as pd
import ipdb

path1 = '/Users/pjs/byte_play/data/bytecup2018/test_small.csv'
df = pd.read_csv(path1)
for i, row in df.iterrows():
    df.loc[i, 'source'] = ','.join([str(x) for x in eval(row['source'])])
    df.loc[i, 'target'] = ','.join([str(x) for x in eval(row['target'])])
df = df[['source','target','uid']]
df.to_csv(path1, index=False)