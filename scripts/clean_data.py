import sys
import pandas as pd
import scipy.sparse
from src.utils import create_cleaning_pipeline, get_feature_names_clean
import yaml

def copy_y_datasets(in_path, out_path):
    pd.read_csv(in_path, nrows=1000).to_csv(out_path, index=False)

def clean_data(input_file, output_file, missing_threshold, text_cols=[], preprocessor=None):
    data = pd.read_csv(input_file, nrows=1000)
    print(f'data columns: {data.columns}')

    y = None

    if not preprocessor:
        preprocessor = create_cleaning_pipeline(text_cols, missing_threshold=missing_threshold)
        X = preprocessor.fit_transform(data)
    else:
        X = preprocessor.transform(data)

    feature_names = get_feature_names_clean(preprocessor, data)


    if scipy.sparse.issparse(X):
        scipy.sparse.save_npz(output_file.replace('.csv', '.npz'), X)
    else:
        if len(feature_names) != X.shape[1]:
            print(f"Warning: Mismatch between number of columns in feature_names ({len(feature_names)}) and X ({X.shape[1]})")

        X_df = pd.DataFrame(X)
        X_df.to_csv(output_file, index=False)

    return preprocessor


if __name__ == "__main__":
    in_dir, out_dir, params_yaml = sys.argv[1:4]
    X_train_out = "".join([out_dir, "/X_train_cleaned.csv"])
    y_train_out = "".join([out_dir, "/y_train_cleaned.csv"])
    X_test_out = "".join([out_dir, "/X_test_cleaned.csv"])
    y_test_out = "".join([out_dir, "/y_test_cleaned.csv"])

    with open(params_yaml, "r") as file:
        params = yaml.safe_load(file)

    text_cols = params['clean']['text_col']
    missing_threshold = float(params['clean']['missing_threshold'])

    preprocessor = clean_data("".join([in_dir, "/X_train.csv"]), X_train_out, missing_threshold, text_cols=text_cols)
    clean_data("".join([in_dir, "/X_test.csv"]), X_test_out, missing_threshold, text_cols=text_cols, preprocessor=preprocessor)

    copy_y_datasets("".join([in_dir, "/y_train.csv"]), y_train_out)
    copy_y_datasets("".join([in_dir, "/y_test.csv"]), y_test_out)
