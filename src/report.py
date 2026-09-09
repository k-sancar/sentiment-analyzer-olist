import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import classification_report

def setup_environment():
    load_dotenv() 
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        raise ValueError("Error: GCP key not found!")
    
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise ValueError("Error: GCP_PROJECT_ID not found in .env!")
        
    return project_id

def map_sentiment(score):
    score = int(score)
    if score <= 2: return 0 
    elif score == 3: return 1  
    else: return 2 

def main():
    project_id = setup_environment()
    
    client = bigquery.Client(project=project_id) 
    query = f"SELECT review_comment_message, review_score FROM `{project_id}.shop_data.shop_data_review`"
    df = client.query(query).to_dataframe(create_bqstorage_client=False)

    df = df.dropna(subset=['review_comment_message'])
    df = df[df['review_comment_message'].str.len() >= 5]

    df['labels'] = df['review_score'].apply(map_sentiment)

    dataset = Dataset.from_pandas(df[['review_comment_message', 'labels']])
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    model_path = "./olist_bert_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def tokenize_function(examples):
        return tokenizer(examples["review_comment_message"], padding="max_length", truncation=True, max_length=128)

    tokenized_test = dataset["test"].map(tokenize_function, batched=True)

    trainer = Trainer(model=model)
    predictions_output = trainer.predict(tokenized_test)

    preds = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids

    print("\n--- DETAILED REPORT ---")
    print(classification_report(
        labels, 
        preds, 
        target_names=['Negative (1-2)', 'Neutral (3)', 'Positive (4-5)'],
        digits=3
    ))

if __name__ == "__main__":
    main()