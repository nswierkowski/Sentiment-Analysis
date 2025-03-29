import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List
import sys

def read_processed_data(path_to_processed_data) -> pd.DataFrame:
    return pd.read_csv(path_to_processed_data, dtype=str)

def split_dataset(path_to_processed_data, test_size, stratify) -> List[pd.DataFrame]:
    processed_data = read_processed_data(path_to_processed_data)
    return train_test_split(
        processed_data.drop(columns=['LABEL-rating']),
        y:=processed_data['LABEL-rating'],
        test_size=test_size,
        stratify=(y if stratify else None)
    )
    
def create_and_save_train_test_df(path_to_processed_data, path_to_split_data, test_size, stratify):
    X_train, X_test, y_train, y_test = split_dataset(path_to_processed_data, test_size, stratify)
    X_train.to_csv("".join([path_to_split_data, '/X_train.csv']), index=False)
    X_test.to_csv("".join([path_to_split_data, '/X_test.csv']), index=False)
    y_train.to_csv("".join([path_to_split_data, '/y_train.csv']), index=False)
    y_test.to_csv("".join([path_to_split_data, '/y_test.csv']), index=False)

if __name__=='__main__':
    path_to_processed_data = sys.argv[1]
    path_to_split_data = sys.argv[2]
    test_size = float(sys.argv[3])
    stratify = sys.argv[4].lower() == "true"
    create_and_save_train_test_df(path_to_processed_data, path_to_split_data, test_size, stratify)