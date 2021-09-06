import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .types import *


def data_preprocessing(data_path, scaling=0) -> Dataframe:

    df = pd.read_csv(data_path, header=0)
    df_t = df.T
    df_t.columns = df['Unnamed: 0']
    df_t = df_t.drop(['Unnamed: 0'])
    df_t.reset_index(drop=True, inplace=True)
    df_t_float = df_t.astype(float, copy=True)  # as the data come from Seurat, the dtype is object

    if scaling==1:
        print("Scaling the data...")
        scaler = StandardScaler()
        scaler.fit(df_t_float)
        df_t_float = scaler.transform(df_t_float)

    train_data, test_data = train_test_split(df_t_float)

    return train_data, test_data
