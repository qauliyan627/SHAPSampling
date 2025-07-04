import os

import pandas as pd
import numpy as np

DATANAME = "Trilobite"
target = "survived"
SPLIT = 0
SAVE = 0

DATAPATH = "SHAPSampling\\Datasets\\Origin\\"
SAVEPATH = f"SHAPSampling\\Datasets\\{DATANAME}\\"

df = pd.read_csv(DATAPATH + DATANAME + ".csv")
print(df.info())
print(df.head())

df.replace('?', np.nan, inplace=True)
df_dropna_columns = df.dropna(how='any')
# df_dropna_columns = df_dropna_columns[df_dropna_columns.columns[df_dropna_columns.columns != "Id"]]
print(df_dropna_columns.info())
print(df_dropna_columns.head())

if SPLIT:
    X = df_dropna_columns[df_dropna_columns.columns[df_dropna_columns.columns != target]]
    y = df_dropna_columns[df_dropna_columns.columns[df_dropna_columns.columns == target]]
    print()
    print(X.head())
    print(y.head())

if SAVE:
    if not os.path.exists(SAVEPATH): os.makedirs(SAVEPATH)
    X.to_csv(SAVEPATH + 'X.csv', index=False)
    y.to_csv(SAVEPATH + 'y.csv', index=False)
    print()
    print("saved")
