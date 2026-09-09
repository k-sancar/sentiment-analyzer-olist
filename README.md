# GCP E-commerce NLP Pipeline

## Overview
This repository contains a Machine Learning pipeline analyzing customer satisfaction using the Olist dataset. The project utilizes Google Cloud Platform for data warehousing and model training workflows.

The project was developed in two main phases:
1. **Initial Approach (Tabular Feature Analysis):** The project began by predicting customer satisfaction based on logistical and behavioral features (e.g., delivery time ratios, freight values, product weight) extracted via SQL queries and evaluated using XGBoost.
2. **Final Outcome (NLP Pipeline):** Since the initial approach failed to deliver satisfactory results, the ultimate solution delivered in this project relies entirely on Natural Language Processing. The final pipeline extracts raw Portuguese review texts and utilizes Deep Learning (Hugging Face BERT) to predict sentiment directly from user comments.

## Architecture & Data Flow
1. **Data Source:** [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
2. **Database (GCP BigQuery):** Raw data is stored and queried using SQL to extract tabular features and review texts.
3. **Data Fetching:** Python scripts authenticate with GCP via a Service Account to download the datasets into memory.
4. **NLP Model:** A Python script fine-tunes the `neuralmind/bert-base-portuguese-cased` model to classify reviews as Negative, Neutral, or Positive.

## Repository Structure
*   `src/`: The final, production-ready NLP pipeline.
    *   `reviews-trf.py`: Trains the BERT model, connects to GCP, and implements a `CustomTrainer` to handle imbalanced review classes.
    *   `report.py`: Evaluation script generating a classification_report on the test set.
    *   `inference.py`: Inference script to test the trained model on new, raw sentences.
*   `experiments/`: Analytical baselines and earlier iterations.
    *   `customer_behavior.ipynb`: Jupyter notebook exploring the initial approach—building a baseline model on non-linguistic tabular features using XGBoost.
    *   `reviews-tfidf.py`: An NLP baseline utilizing Spacy lemmatization and TF-IDF vectorization with XGBoost.
    *   `reviews-lg.py`: An NLP baseline utilizing Spacy vectorization.
*   `sql/`: DDL/DML code for BigQuery.
    *   `feature-extraction.sql`: Complex query extracting logistical, geographical, and sales features (used in the initial tabular approach).
    *   `review-extraction.sql`: Query isolating raw text data for the final NLP pipeline.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/gcp-ecommerce-nlp-pipeline.git](https://github.com/YOUR_USERNAME/gcp-ecommerce-nlp-pipeline.git)
   cd gcp-ecommerce-nlp-pipeline
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv312
   # On Windows:
   venv312\Scripts\activate
   # On macOS/Linux:
   source venv312/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note for GPU utilization: Install the appropriate PyTorch version with CUDA support manually from the PyTorch website before running the training script.*

4. **Configure Google Cloud Credentials:**
   * Create a `.env` file in the root directory based on `.env.example`.
   * Provide your specific credentials:
     ```text
     GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account.json
     GCP_PROJECT_ID=your-project-id
     ```

## Usage

**To run the final NLP training pipeline:**
```bash
python src/reviews-trf.py
```

**To test the model with your own sentences:**
```bash
python src/inference.py
```