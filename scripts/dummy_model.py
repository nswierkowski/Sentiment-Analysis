import json
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier
import sys

import yaml

def read_datasets(path_to_split_data):
    return (
        pd.read_csv("".join([path_to_split_data, '/X_train.csv'])),
        pd.read_csv("".join([path_to_split_data, '/X_test.csv'])),
        pd.read_csv("".join([path_to_split_data, '/y_train.csv'])),
        pd.read_csv("".join([path_to_split_data, '/y_test.csv']))
    )
    
def use_dummy_model(path_to_split_data, train_model):
    X_train, X_test, y_train, y_test = read_datasets(path_to_split_data)
    dummyClassifier = DummyClassifier(strategy=train_model)
    dummyClassifier.fit(X_train, y_train)
    y_pred = dummyClassifier.predict(X_test)
    
    return f1_score(y_test, y_pred, average="weighted")  

def count_f1_for_dummy_model(path_to_split_data, train_model):
    f1_score = use_dummy_model(path_to_split_data, train_model)
    
    print(f"Dummy Classifier F1 Score: {f1_score:.4f}")
    return {"dummy_f1_score": f1_score}
        
def save_f1_dummy_model_score(path_to_split_data, train_model, metrics_path):
    results = count_f1_for_dummy_model(path_to_split_data, train_model)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
        
if __name__=='__main__':
    path_to_split_data = sys.argv[1]
    
    with open(sys.argv[2], "r") as file:
        params = yaml.safe_load(file)
    
    train_model = params['train']['model']
    metrics_path = sys.argv[3]
    save_f1_dummy_model_score(path_to_split_data, train_model, metrics_path)
    