# To run this experiment, first download the language model:
# python -m spacy download pt_core_news_sm

import os
import spacy
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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
    df = client.query(query).to_dataframe(create_bqstorage_client=False)
    print(f"Data loaded. Table shape: {df.shape}")

    reviews = df['review_comment_message'].fillna("").tolist()
    nlp = spacy.load("pt_core_news_sm")

    print("Cleaning and lemmatizing reviews...")
    cleaned_reviews = []
    for doc in nlp.pipe(reviews, disable=["parser", "ner", "textcat"], batch_size=2000):
        tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_stop]
        cleaned_reviews.append(" ".join(tokens))

    df['cleaned_review'] = cleaned_reviews

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['review_score'] - 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=94)
    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=3)
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    print("Training XGBoost model...")
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], sample_weight=weights, verbose=False)
    
    y_pred = clf.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nMatrix of Confusions:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()