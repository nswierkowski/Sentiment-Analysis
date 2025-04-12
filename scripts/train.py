import json
import os
import mlflow
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml
from src.utils import PCAFor, create_feature_pipeline

def read_datasets(path_to_split_data):
    return (
        pd.read_csv(os.path.join(path_to_split_data, 'X_train_cleaned.csv')),
        pd.read_csv(os.path.join(path_to_split_data, 'X_test_cleaned.csv')),
        pd.read_csv(os.path.join(path_to_split_data, 'y_train_cleaned.csv')).squeeze(),
        pd.read_csv(os.path.join(path_to_split_data, 'y_test_cleaned.csv')).squeeze()
    )

def plot_conf_matrix(mlflow, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.savefig(f"conf_matrix_{title}.png")
    mlflow.log_artifact(f"conf_matrix_{title}.png")
    plt.close()

def log_metrics(mlflow, model, y_pred_test, y_true_test, y_pred_train, y_true_train, model_name, config):
    mlflow.log_params({
        "model_type": model_name,
        "mode": config.get('mode', 'all'),
        "selectKBest": config.get('selectKBest', False),
        "pca": config.get('pca', False)
    })
    
    metrics = {
        "train_accuracy": accuracy_score(y_true_train, y_pred_train),
        "train_f1_score": f1_score(y_true_train, y_pred_train, average="weighted"),
        "test_accuracy": accuracy_score(y_true_test, y_pred_test),
        "test_f1_score": f1_score(y_true_test, y_pred_test, average="weighted")
    }
    
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
    
    plot_conf_matrix(mlflow, y_true_test, y_pred_test, f"Test_{model_name}")
    plot_conf_matrix(mlflow, y_true_train, y_pred_train, f"Train_{model_name}")
    mlflow.sklearn.log_model(model, "model")
    
    return metrics

def create_model_pipeline(model, text_cols, mode, use_selectkbest=False, use_pca=False):
    preprocessor = create_feature_pipeline(
        text_cols=text_cols if mode in ["text", "all"] else []
    )
    
    pipeline_steps = [('preprocessor', preprocessor)]
    
    if use_selectkbest:
        pipeline_steps.append(('feature_selection', SelectKBest(f_classif, k=10)))
    
    if use_pca:
        pipeline_steps.append(('pca', PCAFor()))
    
    pipeline_steps.append(('model', model))
    return Pipeline(pipeline_steps)

def find_best_configs(cv_metrics_path):
    try:
        cv_results = pd.read_csv(cv_metrics_path)
        best_configs = {}
        
        for model in ['SVM', 'RandomForest']:
            model_results = cv_results[cv_results['model'] == model]
            if not model_results.empty:
                best_run = model_results.loc[model_results['f1'].idxmax()]
                best_configs[model] = {
                    'mode': best_run['mode'],
                    'selectKBest': best_run['selectKBest'],
                    'pca': best_run['pca'],
                    'f1_score': best_run['f1']
                }
        return best_configs
    except Exception as e:
        print(f"Error reading best configs: {e}")
        return None

def train_dummy_model(X_train, X_test, y_train, y_test, strategy):
    dummy = DummyClassifier(strategy=strategy)
    dummy.fit(X_train, y_train)
    
    y_pred_test = dummy.predict(X_test)
    y_pred_train = dummy.predict(X_train)
    
    with mlflow.start_run(run_name="Dummy"):
        metrics = log_metrics(
            mlflow, dummy, 
            y_pred_test, y_test, 
            y_pred_train, y_train,
            "Dummy", {}
        )
    
    return {"dummy": metrics}

def train_model_with_config(X_train, X_test, y_train, y_test, model_name, config, text_cols):
    if model_name == "SVM":
        model = SVC(kernel="linear", random_state=42)
    else:
        model = RandomForestClassifier(random_state=42)
    
    pipeline = create_model_pipeline(
        model, text_cols,
        mode=config['mode'],
        use_selectkbest=config['selectKBest'],
        use_pca=config['pca']
    )
    
    pipeline.fit(X_train, y_train)
    
    y_pred_test = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)
    
    with mlflow.start_run(run_name=f"{model_name}_BestConfig"):
        metrics = log_metrics(
            mlflow, pipeline,
            y_pred_test, y_test,
            y_pred_train, y_train,
            model_name, config
        )
    
    return metrics

def main():
    path_to_split_data, path_val_metrics, params_yaml, metrics_path = sys.argv[1:5]
    
    with open(params_yaml, 'r') as file:
        params = yaml.safe_load(file)
    
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("model_training")
    
    X_train, X_test, y_train, y_test = read_datasets(path_to_split_data)
    text_cols = params.get('text_cols', [X_train.columns[0]])
    
    results = train_dummy_model(X_train, X_test, y_train, y_test, params['train']['model'])
    
    best_configs = find_best_configs(path_val_metrics)
    
    if best_configs:
        for model_name, config in best_configs.items():
            model_results = train_model_with_config(
                X_train, X_test, y_train, y_test,
                model_name, config, text_cols
            )
            results[f"{model_name.lower()}_best"] = {
                "config": config,
                "metrics": model_results
            }
    
    for model_result in results.values():
        if isinstance(model_result, dict) and 'config' in model_result:
            config = model_result['config']
            for key in ['selectKBest', 'pca']:
                if key in config:
                    config[key] = int(config[key])
    
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()