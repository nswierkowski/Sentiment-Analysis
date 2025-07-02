# Sentiment Analysis

This project focuses on sentiment analysis using a dataset of Sephora product reviews:  
[Sephora Products and Skincare Reviews on Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)

The project compares classical machine learning models and evaluates their explainability.

- Exploratory analysis notebook: `notebooks/exploratory_analysis.ipynb`
- Hyperparameter tuning and model explainability: `notebooks/tuning_hiperparameters.ipynb`

---

## 🚀 How to Run the Project

### 1. Prepare Data Directory Structure

Ensure the `data/` folder has the following structure:
```
data/
    cleaned/
        rt-polaritydata/
        sephora/
    mlflow/
    nltk_data/
    processed/
    raw/
        rt-polaritydata/
        sephora/
    splitted/
        rt-polaritydata/
        sephora/
    validate/
        rt-polaritydata/
        sephora/
    vectorization/
        rt-polaritydata/
        sephora/
metrics/
```

> Place the raw datasets into their respective subdirectories under `data/raw/`.

---

### 2. Create a Python Environment and Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Build the Docker Image
```bash
make build
```

4. Run the Project
```bash
make run
```


Final results:
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Dataset</th>
      <th>Model</th>
      <th>Mode (only text, no text, all)</th>
      <th>Params</th>
      <th>SelectKBest</th>
      <th>PCA</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>sephora</td>
      <td>Dummy</td>
      <td>-</td>
      <td>-</td>
      <td>False</td>
      <td>False</td>
      <td>0.632</td>
      <td>0.399424</td>
      <td>0.632</td>
      <td>0.48949019607843136</td>
    </tr>
    <tr>
      <td>sephora</td>
      <td>SVM Best</td>
      <td>all</td>
      <td>{'model__kernel': 'rbf', 'model__C': 10.0}</td>
      <td>True</td>
      <td>True</td>
      <td>0.549</td>
      <td>0.492071911439409</td>
      <td>0.549</td>
      <td>0.5134903908832975</td>
    </tr>
    <tr>
      <td>sephora</td>
      <td>RandomForest Best</td>
      <td>all</td>
      <td>{'model__n_estimators': 100, 'model__min_samples_split': 5, 'model__max_depth': 10}</td>
      <td>True</td>
      <td>True</td>
      <td>0.537</td>
      <td>0.49399435252665835</td>
      <td>0.537</td>
      <td>0.5107971696675814</td>
    </tr>
    <tr>
      <td>polarity</td>
      <td>Dummy</td>
      <td>-</td>
      <td>-</td>
      <td>False</td>
      <td>False</td>
      <td>0.496</td>
      <td>0.24601599999999998</td>
      <td>0.496</td>
      <td>0.3288983957219251</td>
    </tr>
    <tr>
      <td>polarity</td>
      <td>SVM Best</td>
      <td>all</td>
      <td>{'model__kernel': 'linear', 'model__C': 10.0}</td>
      <td>True</td>
      <td>True</td>
      <td>0.513</td>
      <td>0.5164174107447141</td>
      <td>0.513</td>
      <td>0.49800407025042226</td>
    </tr>
    <tr>
      <td>polarity</td>
      <td>RandomForest Best</td>
      <td>all</td>
      <td>{'model__n_estimators': 200, 'model__min_samples_split': 10, 'model__max_depth': 20}</td>
      <td>True</td>
      <td>True</td>
      <td>0.512</td>
      <td>0.5155883361921098</td>
      <td>0.512</td>
      <td>0.49531463695735695</td>
    </tr>
    <tr>
      <td>sephora</td>
      <td>Bag-of-Words</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.635</td>
      <td>0.5868962201872834</td>
      <td>0.635</td>
      <td>0.600742194930262</td>
    </tr>
    <tr>
      <td>sephora</td>
      <td>TF-IDF</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.644</td>
      <td>0.4917197180528308</td>
      <td>0.644</td>
      <td>0.5289205670110072</td>
    </tr>
    <tr>
      <td>sephora</td>
      <td>Word2Vec</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.638</td>
      <td>0.4831018329938901</td>
      <td>0.638</td>
      <td>0.5069993804213135</td>
    </tr>
    <tr>
      <td>polarity</td>
      <td>Bag-of-Words</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.655</td>
      <td>0.6559563763105377</td>
      <td>0.655</td>
      <td>0.6546737371543856</td>
    </tr>
    <tr>
      <td>polarity</td>
      <td>TF-IDF</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.638</td>
      <td>0.646828441714064</td>
      <td>0.638</td>
      <td>0.6332436886324526</td>
    </tr>
    <tr>
      <td>polarity</td>
      <td>Word2Vec</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.732</td>
      <td>0.7328495734171532</td>
      <td>0.732</td>
      <td>0.7318498599439777</td>
    </tr>
  </tbody>
</table>
