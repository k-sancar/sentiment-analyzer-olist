# To run this experiment, first download the language model:
# python -m spacy download pt_core_news_lg

import os
import spacy
import numpy as np
import pandas as pd
import xgboost as xgb
from google.cloud import bigquery
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone

if __name__ == '__main__':
    load_dotenv()

    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        raise ValueError("Error: GCP key not found!")

    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise ValueError("Error: GCP_PROJECT_ID not found in .env!")

    client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{project_id}.shop_data.shop_data_review`"
    df = client.query(query).to_dataframe(create_bqstorage_client=False)

    nlp = spacy.load("pt_core_news_lg")
    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=3)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=94)

    text_column = 'review_comment_message'
    label_column = 'review_score'
    df = df.dropna(subset=[text_column]).reset_index(drop=True)
    texts = df[text_column].tolist()

    vectors = []

    for doc in nlp.pipe(texts, disable=["parser", "ner", "textcat"], batch_size=2000):
        vectors.append(doc.vector)
        
    X, y = np.array(vectors), df[label_column].astype(int) - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    results = {}

    for train_idx, test_idx in cv.split(X_train, y_train):
        X_train_cv, X_test_cv = X_train[train_idx], X_train[test_idx]
        y_train_cv, y_test_cv = y_train.iloc[train_idx], y_train.iloc[test_idx]
        weights_cv = weights[train_idx]

        estimator = clone(clf)
        
        est = estimator.fit(X_train_cv, y_train_cv, eval_set=[(X_test_cv, y_test_cv)], sample_weight=weights_cv, verbose=False)
        train_score = f1_score(y_train_cv, estimator.predict(X_train_cv), average='macro')
        test_score = f1_score(y_test_cv, estimator.predict(X_test_cv), average='macro')

        results[est] = (train_score, test_score)

    for i, (model_instance, scores) in enumerate(results.items()):
        print(f"Iteration {i+1}: Train Macro F1: {scores[0]:.4f} | Test Macro F1: {scores[1]:.4f}")

    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    weights_final = compute_sample_weight(class_weight='balanced', y=y_train_final)

    clf.fit(X_train_final, y_train_final, eval_set=[(X_val, y_val)], sample_weight=weights_final, verbose=False)

    y_pred = clf.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nMatrix of Confusions:")
    print(confusion_matrix(y_test, y_pred))