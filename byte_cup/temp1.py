import pandas as pd
import ipdb

path1 = '/Users/pjs/byte_play/data/bytecup2018/test.csv'
df = pd.read_csv(path1)
df['source'] = df['source'].apply(lambda x: ','.join([str(x_) for x_ in eval(x)]))
df['target'] = df['target'].apply(lambda x: ','.join([str(x_) for x_ in eval(x)]))
df.to_csv('/Users/pjs/byte_play/data/bytecup2018/test_.csv', index=False)
print("Test done!")

path1 = '/Users/pjs/byte_play/data/bytecup2018/train.csv'
df = pd.read_csv(path1)
df['source'] = df['source'].apply(lambda x: ','.join([str(x_) for x_ in eval(x)]))
df['target'] = df['target'].apply(lambda x: ','.join([str(x_) for x_ in eval(x)]))
df.to_csv('/Users/pjs/byte_play/data/bytecup2018/train_.csv', index=False)
print("Train done!")