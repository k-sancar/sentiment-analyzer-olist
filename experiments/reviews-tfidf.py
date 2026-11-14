# To run this experiment, first download the language model:
# python -m spacy download pt_core_news_sm

import os
import spacy
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone
import xgboost as xgb

def setup_environment():
    load_dotenv() 
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        raise ValueError("Error: GCP key not found!")
    
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise ValueError("Error: GCP_PROJECT_ID not found in .env!")
        
    return project_id

def main():
    project_id = setup_environment()
    
    client = bigquery.Client(project=project_id) 
    query = f"SELECT * FROM `{project_id}.shop_data.shop_data_review`"
    print("Setting connection with BigQuery and downloading view...")
    df = client.query(query).to_dataframe(create_bqstorage_client=False)
    print(f"Success! Data loaded. Table shape: {df.shape}")

    df = df.dropna(subset=['review_comment_message']).reset_index(drop=True)
    df = df[df['review_comment_message'].str.len() >= 5]
    reviews = df['review_comment_message'].tolist()

    nlp = spacy.load("pt_core_news_sm")

    print("Cleaning and lemmatizing reviews...")
    cleaned_reviews = []
    for doc in nlp.pipe(reviews, disable=["parser", "ner", "textcat"], batch_size=2000):
        tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop]
        cleaned_reviews.append(" ".join(tokens))

    df['cleaned_review'] = cleaned_reviews

    def map_sentiment(score):
            score = int(score)
            if score <= 2: return 0
            elif score == 3: return 1
            else: return 2

    X = df['cleaned_review']
    y = df['review_score'].apply(map_sentiment)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42
    )

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer()

    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=94)
    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=3)
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    print("Running 5-Fold Cross-Validation...")

    for i, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train), start = 1):

        X_train_cv, y_train_cv = X_train[train_idx], y_train.iloc[train_idx]
        X_test_cv, y_test_cv = X_train[test_idx], y_train.iloc[test_idx]
        weights_cv = compute_sample_weight(class_weight='balanced', y=y_train_cv)

        estimator = clone(clf)

        estimator.fit(
                    X_train_cv, y_train_cv,
                    eval_set=[(X_val, y_val)],
                    sample_weight=weights_cv,
                    verbose=False
                )

        train_score = f1_score(
                    y_train_cv, 
                    estimator.predict(X_train_cv), 
                    average='macro'
                )
        
        test_score = f1_score(
                    y_test_cv, 
                    estimator.predict(X_test_cv), 
                    average='macro'
                )

        print(f"Iteration {i}: Train Macro F1: {train_score:.4f} | Test Macro F1: {test_score:.4f}")

    clf.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=False,
        sample_weight=weights 
    )

    y_pred = clf.predict(X_test)

    target_names = ['Negative (1-2)', 'Neutral (3)', 'Positive (4-5)']

    print(classification_report(y_test, y_pred, target_names=target_names, digits=3))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
if __name__ == "__main__":
    main()