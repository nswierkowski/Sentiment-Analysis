import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.data.path.append("/app/data/nltk_data")

nltk.download('punkt_tab', download_dir="/app/data/nltk_data")
nltk.download('punkt', download_dir="/app/data/nltk_data")
nltk.download('stopwords', download_dir="/app/data/nltk_data")
nltk.download('wordnet', download_dir="/app/data/nltk_data")

class PCAFor(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold=0.95):
        self.variance_threshold = variance_threshold
        self.pca_ = None
        self.n_components_ = None
    
    def fit(self, X, y=None):
        pca_full = PCA()
        pca_full.fit(X)
        
        explained_variance = np.cumsum(pca_full.explained_variance_ratio_)
        self.n_components_ = np.argmax(explained_variance >= self.variance_threshold) + 1
        
        self.pca_ = PCA(n_components=self.n_components_)
        self.pca_.fit(X)
        return self
    
    def transform(self, X):
        return self.pca_.transform(X)

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = re.sub(r'[^\w\s]', '', re.sub(r'\s+', ' ', str(text).lower()))
        tokens = nltk.word_tokenize(text)
        return ' '.join(self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.apply(lambda col: col.apply(self.clean_text))
        else:
            df = X.apply(self.clean_text)
            
        print(f'cols: {df.columns}')
        return df
        


class DropMostlyMissingColumns(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.columns_to_keep_ = []

    def fit(self, X, y=None):
        missing_ratio = X.isnull().mean()
        self.columns_to_keep_ = missing_ratio[missing_ratio <= self.threshold].index.tolist()
        return self

    def transform(self, X):
        return X[self.columns_to_keep_]


def get_feature_names(preprocessor):
    feature_names = []

    if hasattr(preprocessor, 'named_steps') and 'text_cleaning' in preprocessor.named_steps:
        preprocessor = preprocessor.named_steps['text_cleaning']
    
    if not hasattr(preprocessor, 'transformers_'):
        raise ValueError("Provided preprocessor does not contain transformers_")

    for name, trans, cols in preprocessor.transformers_:
        if hasattr(trans, 'named_steps'):
            if 'vectorizer' in trans.named_steps:
                fn = trans.named_steps['vectorizer'].get_feature_names_out()
            elif 'encoder' in trans.named_steps:
                fn = trans.named_steps['encoder'].get_feature_names_out(cols)
            else:
                fn = cols
        else:
            fn = cols

        feature_names.extend(fn)

    return feature_names

def get_feature_names_clean(preprocessor, data):
    feature_names = []

    if hasattr(preprocessor, 'named_steps') and 'text_cleaning' in preprocessor.named_steps:
        text_cleaning_pipeline = preprocessor.named_steps['text_cleaning']
        if hasattr(text_cleaning_pipeline, 'transformers_'):
            for name, trans, cols in text_cleaning_pipeline.transformers_:
                if isinstance(trans, Pipeline):
                    if hasattr(trans.named_steps['combine_text'], 'get_feature_names_out'):
                        feature_names.append("cleaned_text")  
                else:
                    print(f'text_cleaning_pipeline.transformers_: {cols}')
                    feature_names.extend(cols)  

    feature_names.append(len(feature_names))
    return feature_names

def create_preprocessing_pipeline(text_cols):
    dropper = DropMostlyMissingColumns(threshold=0.5)

    text_pipeline = Pipeline([
        ('combine_text', FunctionTransformer(lambda X: X.astype(str).agg(' '.join, axis=1), validate=False)),
        ('cleaner', TextCleaner()),
        ('vectorizer', TfidfVectorizer(
            max_features=30,    
            dtype=np.float32,    
            stop_words='english' 
        ))
    ])
    
    categorical_pipeline = Pipeline([
        ('to_str', FunctionTransformer(lambda x: x.astype(str))),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))  
    ])

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False)) 
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_pipeline, text_cols),
            ('cat', categorical_pipeline, make_column_selector(dtype_include='object')),
            ('num', numerical_pipeline, make_column_selector(dtype_include=np.number))
        ],
        remainder='drop'
    )

    full_pipeline = Pipeline([
        ('drop_mostly_missing', dropper),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline


def create_cleaning_pipeline(text_cols, missing_threshold=0.5):
    dropper = DropMostlyMissingColumns(threshold=missing_threshold)

    text_cleaning_pipeline = Pipeline([
        ('combine_text', FunctionTransformer(
            lambda X: pd.DataFrame(X.astype(str).agg(' '.join, axis=1), columns=["cleaned_text"]),
            validate=False
        )),
        ('cleaner', TextCleaner()),
    ])

    cleaning = ColumnTransformer(
        transformers=[
            ('text_cleaning', text_cleaning_pipeline, text_cols)
        ],
        remainder='passthrough' 
    )

    return Pipeline([
        ('drop_mostly_missing', dropper),
        ('text_cleaning', cleaning)
    ])


def create_feature_pipeline(text_cols):
    text_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=100,
            dtype=np.float32,
            stop_words='english'
        ))
    ])

    categorical_pipeline = Pipeline([
        ('to_str', FunctionTransformer(lambda x: x.astype(str))),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_pipeline, text_cols),
            ('cat', categorical_pipeline, make_column_selector(dtype_include='object')),
            ('num', numerical_pipeline, make_column_selector(dtype_include=np.number))
        ],
        remainder='drop'
    )

    return preprocessor
