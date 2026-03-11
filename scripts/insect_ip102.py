from pathlib import Path
import pandas as pd

DATA_PATH = Path('data/ip102')

def read_split(path, split):
    rows=[]
    with open(path ,'r') as f:
        for line in f:
            img, label= line.strip().split()
            rows.append({'image': img, 'label': int(label), 'split': split})
    return pd.DataFrame(rows)

train =read_split(DATA_PATH / 'train.txt', 'train')
test =read_split(DATA_PATH / 'test.txt', 'test')
val =read_split(DATA_PATH / 'val.txt', 'val')

df = pd.concat([train, val, test], ignore_index=True)

print('Total images:', len(df))
print(df['split'].value_counts())
print(df.tail())
print('Unique classes:', df['label'].nunique())

print("\nSplit distribution:")
print(df.split.value_counts())

counts = df.groupby("label").size().sort_values()

print("\nSmallest classes:")
print(counts.head())

print("\nLargest classes:")
print(counts.tail())


# df.to_csv(DATA_PATH/"metadata.csv",index=False)
# print("\nSaved metadata.csv")