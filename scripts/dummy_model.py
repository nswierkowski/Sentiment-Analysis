import json
import os
import mlflow
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import yaml

def read_datasets(path_to_split_data):
    return (
        pd.read_csv("".join([path_to_split_data, '/X_train.csv'])),
        pd.read_csv("".join([path_to_split_data, '/X_test.csv'])),
        pd.read_csv("".join([path_to_split_data, '/y_train.csv'])),
        pd.read_csv("".join([path_to_split_data, '/y_test.csv']))
    )
    
def plot_conf_matrix(mlflow, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.savefig(f"/app/data/mlflow/conf_matrix_{title}.png")
    mlflow.log_artifact(f"/app/data/mlflow/conf_matrix_{title}.png")
    
def for_mlflow(mlflow, model, y_pred_test, y_true_test, y_pred_train, y_true_train):
    mlflow.log_params({"model_type": "DummyClassifier"})
    
    mlflow.log_metric("train_accuracy", accuracy_score(y_true_train, y_pred_train))
    mlflow.log_metric("train_f1_score", f1_score(y_true_train, y_pred_train, average="weighted"))    
    mlflow.log_metric("test_accuracy", accuracy_score(y_true_test, y_pred_test))
    mlflow.log_metric("test_f1_score", f1_score(y_true_test, y_pred_test, average="weighted"))

    plot_conf_matrix(mlflow, y_true_test, y_pred_test, "Test")
    plot_conf_matrix(mlflow, y_true_train, y_pred_train, "Train")

    mlflow.sklearn.log_model(model, "model")
    
def use_dummy_model(path_to_split_data, train_model):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("train_model")
    X_train, X_test, y_train, y_test = read_datasets(path_to_split_data)
    dummyClassifier = DummyClassifier(strategy=train_model)
    dummyClassifier.fit(X_train, y_train)
    y_pred = dummyClassifier.predict(X_test)
    
    with mlflow.start_run():
        for_mlflow(mlflow, dummyClassifier, y_pred, y_test, dummyClassifier.predict(X_train), y_train)
    
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
    