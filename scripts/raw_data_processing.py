import pandas as pd
import os
import numpy as np
import sys
from typing import List
import yaml

def read_products(path_to_raw_data) -> pd.DataFrame:
    return pd.read_csv("".join([path_to_raw_data, '/product_info.csv']), dtype=str)

def read_all_reviews(path_to_raw_data) -> pd.DataFrame:
    return pd.concat([
        pd.read_csv("".join([path_to_raw_data, '/', file]), lineterminator='\n', dtype=str)\
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
    
def add_new_columns_base_on_knowledge_domain(df: pd.DataFrame) -> pd.DataFrame:
    df['text_len'] = df['review_text'].str.len()
    df['contains_capslock'] = df['review_text'].str.contains(r'[A-Z]{2,}')
    df['contains_exclamation_point'] = df['review_title'].str.count(r'!')
    df['contains_emoticon'] = df['review_text'].str.contains(r'[:;<][)(3]')
    return df
    
def handle_existing_columns(df: pd.DataFrame, columns_not_to_keep: List[str]) -> pd.DataFrame:
    return add_new_columns_base_on_knowledge_domain(
        df.drop(columns=columns_not_to_keep)
    )
    
    
def preprocess_and_save(path_to_raw_data, path_to_processed_data, columns_not_to_keep) -> None:
    handle_existing_columns(join_products_and_reviews(path_to_raw_data), columns_not_to_keep)\
        .to_csv(path_to_processed_data, index=False)

if __name__=='__main__':
    path_to_raw_data = sys.argv[1]
    path_to_processed_data = sys.argv[2]
    
    with open('params.yaml', "r") as file:
        params = yaml.safe_load(file)

    columns_not_to_keep = params['raw_data_processing']['colums_not_to_keep']
    preprocess_and_save(path_to_raw_data, path_to_processed_data, columns_not_to_keep)