import os
import pandas as pd
import numpy as np
from google.cloud import bigquery
from dotenv import load_dotenv
import xgboost as xgb
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight

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
    query = f"SELECT * FROM `{project_id}.shop_data.shop_data_view`"
    print("Setting connection with BigQuery and downloading view...")
    df = client.query(query).to_dataframe(create_bqstorage_client=False)
    print(f"Success! Data loaded. Table shape: {df.shape}")

    # Konwersja cech kategorialnych pod XGBoost
    text_columns = ['customer_state', 'main_seller_state', 'main_product_category', 'order_status']
    df[text_columns] = df[text_columns].astype("category")

    # Mapowanie ocen na 3 kategorie (1-2 -> 0, 3 -> 1, 4-5 -> 2)
    def map_sentiment(score):
        score = int(score)
        if score <= 2: return 0
        elif score == 3: return 1
        else: return 2

    X = df.drop('review_score', axis=1)
    y = df['review_score'].apply(map_sentiment)

    # 1. Podział na zbiór treningowy i nietykalny testowy (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Wydzielenie osobnego zbioru walidacyjnego z puli treningowej pod early stopping (zapobiega wyciekowi danych)
    X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=94)
    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=3, enable_categorical=True)

    weights = compute_sample_weight(class_weight='balanced', y=y_train_final)
    results = {}

    print("Running 5-Fold Cross-Validation...")
    for train_idx, test_idx in cv.split(X_train_final, y_train_final):
        X_train_cv, X_test_cv = X_train_final.iloc[train_idx], X_train_final.iloc[test_idx]
        y_train_cv, y_test_cv = y_train_final.iloc[train_idx], y_train_final.iloc[test_idx]
        weights_cv = compute_sample_weight(class_weight='balanced', y=y_train_cv)

        estimator = clone(clf)
        estimator.fit(
            X_train_cv, y_train_cv, 
            eval_set=[(X_test_cv, y_test_cv)], 
            verbose=False, 
            sample_weight=weights_cv
        )
        
        train_score = f1_score(y_train_cv, estimator.predict(X_train_cv), average='macro')
        test_score = f1_score(y_test_cv, estimator.predict(X_test_cv), average='macro')
        results[estimator] = (train_score, test_score)

    for i, (model_instance, scores) in enumerate(results.items()):
        print(f"Iteration {i+1}: Train Macro F1: {scores[0]:.4f} | Test Macro F1: {scores[1]:.4f}")

    # Finalne trenowanie na całej puli treningowej z użyciem oddzielnego zbioru walidacyjnego do zatrzymania
    print("\nTraining final XGBoost model on tabular features...")
    weights_final = compute_sample_weight(class_weight='balanced', y=y_train_final)
    clf.fit(
        X_train_final, y_train_final, 
        eval_set=[(X_val, y_val)], 
        verbose=False, 
        sample_weight=weights_final
    )

    y_pred = clf.predict(X_test)

    target_names = ['Negative (1-2)', 'Neutral (3)', 'Positive (4-5)']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()