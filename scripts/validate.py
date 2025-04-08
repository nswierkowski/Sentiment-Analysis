import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from src.utils import create_feature_pipeline, PCAFor
import yaml
import itertools
import os
    
    
def evaluate_models_with_cv(X, y, text_cols, mode, cv, selectKBest=False, pca=False):
    if mode == "text":
        X = X[text_cols]
        preprocessor = create_feature_pipeline(text_cols=text_cols)
    elif mode == "non-text":
        X = X.drop(columns=text_cols)
        preprocessor = create_feature_pipeline(text_cols=[])  
    elif mode == "all":
        preprocessor = create_feature_pipeline(text_cols=text_cols)
    else:
        raise ValueError("mode must be 'text', 'non-text', or 'all'")

    models = {
        "SVM": SVC(kernel="linear"),
        "RandomForest": RandomForestClassifier(),
        "Dummy": DummyClassifier(strategy="most_frequent")
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0),
        'f1': make_scorer(f1_score, average='macro', zero_division=0)
    }

    results = []
    for name, model in models.items():
        pipeline_steps = [('preprocessor', preprocessor)]
        
        if selectKBest:
            pipeline_steps.append(('feature_selection', SelectKBest(f_classif, k=10)))
        
        if pca:
            pipeline_steps.append(('pca', PCAFor()))
        
        pipeline_steps.append(('model', model))
        
        pipeline = Pipeline(pipeline_steps)

        scores = cross_validate(
            pipeline,
            X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=False
        )

        avg_scores = {
            'model': name,
            'mode': mode,
            'selectKBest': selectKBest,
            'pca': pca,
            **{metric: np.mean(scores[f'test_{metric}']) for metric in scoring}
        }
        results.append(avg_scores)

    return pd.DataFrame(results)


if __name__ == "__main__":
    in_dir, out_dir, params_path = sys.argv[1:4]

    X = pd.read_csv("".join([in_dir, "/X_train_cleaned.csv"]))
    y = pd.read_csv("".join([in_dir, "/y_train_cleaned.csv"]))#.values.ravel()

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    text_cols = [X.columns[0]]
    print(f'text_cols {text_cols}')
    cv = int(params['validate']['cv'])

    all_results = []
    for mode, selectKBest, pca in itertools.product(["text", "non-text", "all"], [False, True], [False, True]):
        print(f"Evaluating mode: {mode}")
        df_results = evaluate_models_with_cv(X, y, text_cols, mode, cv, selectKBest, pca)
        all_results.append(df_results)

    pd.concat(all_results).to_csv("".join([out_dir, "/cv_metrics_all_modes.csv"]), index=False)
