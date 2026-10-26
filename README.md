# GCP E-commerce NLP Pipeline

## Overview
This repository contains a Machine Learning pipeline analyzing customer satisfaction using the Olist dataset. The project utilizes Google Cloud Platform for data warehousing and model training workflows.

The project was developed in two main phases:
1. **Initial Approach (Tabular Feature Analysis):** The project began by predicting customer satisfaction based on logistical and behavioral features (e.g., delivery time ratios, freight values, product weight) extracted via SQL queries and evaluated using XGBoost.
2. **Final Outcome (NLP Pipeline):** Since the initial approach failed to deliver satisfactory results, the ultimate solution delivered in this project relies entirely on Natural Language Processing. The final pipeline extracts raw Portuguese review texts and utilizes Deep Learning (Hugging Face BERT) to predict sentiment directly from user comments.

## Architecture & Data Flow
1. **Data Source:** [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
2. **Database & SQL Views (GCP BigQuery):** Raw tables are stored in BigQuery, where custom SQL views handle data aggregation, cleaning, and feature extraction directly on the warehouse level.
3. **Data Fetching:** Python scripts authenticate with GCP via a Service Account and query the pre-processed BigQuery views directly into memory, avoiding local raw-table joins.
4. **NLP Model:** A Python script fine-tunes the `neuralmind/bert-base-portuguese-cased` model to classify reviews as Negative, Neutral, or Positive.
5. **Deployment:** The trained model is served via a Dockerized container using a lightweight, CPU-optimized environment for memory-efficient inference.

## Repository Structure
-   `src/`: The final, production-ready NLP pipeline.
    *   `reviews-trf.py`: Trains the BERT model, connects to GCP, and implements a `CustomTrainer` to handle imbalanced review classes.
    *   `report.py`: Evaluation script generating a classification_report on the test set.
    *   `inference.py`: Inference script to test the trained model on new, raw sentences.
    *   `app.py`: Web server endpoint for Docker deployment.
-   `experiments/`: Analytical baselines and earlier iterations.
    *   `customer_behavior.ipynb`: Jupyter notebook exploring the initial approach using XGBoost on tabular features.
    *   `reviews-tfidf.py`: An NLP baseline utilizing Spacy lemmatization and TF-IDF vectorization.
    *   `reviews-lg.py`: An NLP baseline utilizing Spacy vectorization.
-   `sql/`: DDL/DML scripts generating BigQuery views.
    * `feature-extraction.sql`: View definition aggregating logistical, geographical, and sales features.
    * `review-extraction.sql`: View definition isolating text data from reviews for the NLP pipeline.
-   `dev-requirements.txt`: Full dependency list for local GPU training and data acquisition.
-   `api-requirements.txt`: Lightweight dependency list strictly for Dockerized API inference.
-   `Dockerfile`: Instructions to build the inference container.

## Installation & Setup

**Prerequisite:** This project strictly requires **Python 3.12**.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/k-sancar/sentiment-analyzer-olist.git
   cd sentiment-analyzer-olist
   ```

2. **Create and activate a Python 3.12 virtual environment:**
   ```bash
   # On Windows (explicitly target 3.12 via Python Launcher):
   py -3.12 -m venv venv312
   venv312\Scripts\activate

   # On macOS/Linux:
   python3.12 -m venv venv312
   source venv312/bin/activate
   ```

3. **Install dependencies (Local Training):**
   ```bash
   # Option A: Windows/Linux with NVIDIA GPU (recommended for training)
   pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
   pip install -r dev-requirements.txt

   # Option B: macOS or CPU-only machines
   pip install -r dev-requirements.txt
   ```

4. **Configure Google Cloud Credentials:**
   * Create a `.env` file in the root directory based on `.env.example`.
   * Provide your specific credentials:
     ```text
     GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account.json
     GCP_PROJECT_ID=your-project-id
     ```

## Usage

**1. Train the final NLP model locally:**
```bash
python src/reviews-trf.py
```

**2. Test the model locally:**
```bash
python src/inference.py
```

**3. Deploy via Docker:**
```bash
docker build -t olist-sentiment-api .
docker run -p 8000:8000 olist-sentiment-api
```