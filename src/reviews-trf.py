import os
import torch
import pandas as pd
import numpy as np
from torch import nn
from dotenv import load_dotenv
from google.cloud import bigquery
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weight = self.class_weights.to(model.device)
        
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def setup_environment():
    load_dotenv() 
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        raise ValueError("Error: GCP key not found!")
    
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        raise ValueError("Error: GCP_PROJECT_ID not found in .env!")
        
    return project_id

def load_and_preprocess_data(project_id: str) -> pd.DataFrame:
    client = bigquery.Client(project=project_id) 
    query = f"SELECT review_comment_message, review_score FROM `{project_id}.shop_data.shop_data_review`"
    df = client.query(query).to_dataframe(create_bqstorage_client=False)

    df = df.dropna(subset=['review_comment_message'])
    df = df[df['review_comment_message'].str.len() >= 5]

    def map_sentiment(score):
        score = int(score)
        if score <= 2: return 0 
        elif score == 3: return 1  
        else: return 2 

    df['labels'] = df['review_score'].apply(map_sentiment)
    return df

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision": precision,
        "recall": recall
    }

def main():
    project_id = setup_environment()
    
    df = load_and_preprocess_data(project_id)
    dataset = Dataset.from_pandas(df[['review_comment_message', 'labels']])
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    train_labels = np.array(dataset["train"]["labels"])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    model_name = "neuralmind/bert-base-portuguese-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    def tokenize_function(examples):
        return tokenizer(
            examples["review_comment_message"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        warmup_steps=500
    )

    trainer = CustomTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    trainer.train()
    model.save_pretrained("./olist_bert_model")
    tokenizer.save_pretrained("./olist_bert_model")

if __name__ == "__main__":
    main()