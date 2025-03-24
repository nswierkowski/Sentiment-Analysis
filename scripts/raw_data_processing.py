import pandas as pd
import os
import numpy as np
import sys

def read_products(path_to_raw_data) -> pd.DataFrame:
    return pd.read_csv("".join([path_to_raw_data, '/product_info.csv']), dtype=str)\
        [['product_id', 'product_name', 'brand_name', 'loves_count']]

def read_all_reviews(path_to_raw_data) -> pd.DataFrame:
    return pd.concat([
        pd.read_csv("".join([path_to_raw_data, '/', file]), lineterminator='\n', dtype=str)\
            [['LABEL-rating', 'helpfulness', 'review_text', 'review_title', 'product_id']]\
            .replace("", np.nan)
            .dropna(subset=['LABEL-rating'])
        for file in os.listdir(path_to_raw_data) 
        if file.startswith('reviews')
    ])
        
def join_products_and_reviews(path_to_raw_data) -> pd.DataFrame:
    return pd.merge(
        read_products(path_to_raw_data), 
        read_all_reviews(path_to_raw_data), 
        how="right", 
        on=["product_id"]
    )

def preprocess_and_save(path_to_raw_data, path_to_processed_data):
    join_products_and_reviews(path_to_raw_data)\
        .to_csv(path_to_processed_data)

if __name__=='__main__':
    path_to_raw_data = sys.argv[1]
    path_to_processed_data = sys.argv[2]
    preprocess_and_save(path_to_raw_data, path_to_processed_data)