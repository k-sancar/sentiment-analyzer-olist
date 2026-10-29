# GCP E-commerce NLP Pipeline

## Overview
This repository contains a Machine Learning pipeline analyzing customer satisfaction using the Olist dataset. The project utilizes Google Cloud Platform for data warehousing and local GPU acceleration for model training workflows.

The project was developed in two main phases:
1. **Initial Approach (Tabular Feature Analysis):** The project began by predicting customer satisfaction based on logistical and behavioral features (e.g., delivery time ratios, freight values, product weight) extracted via SQL queries and evaluated using XGBoost.
2. **Final Outcome (NLP Pipeline):** Since the initial tabular approach proved insufficient for predicting customer ratings, the project pivoted to Natural Language Processing. The final pipeline extracts raw Portuguese review texts and utilizes Deep Learning (Hugging Face BERTimbau) to predict sentiment directly from user comments.

## Architecture & Data Flow
1. **Data Source:** [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
2. **Database & SQL Views (GCP BigQuery):** Raw tables are stored in BigQuery, where custom SQL views (queries available in '/sql' directory) handle data aggregation, cleaning, and feature extraction directly on the warehouse level.
3. **Data Fetching:** Python scripts authenticate with GCP via a Service Account and extracts preprocessed BigQuery views directly into memory, avoiding local raw-table joins.
4. **NLP Model:** A Python script fine-tunes the `neuralmind/bert-base-portuguese-cased` model to classify reviews as Negative, Neutral, or Positive.
5. **Deployment:** The model is served via a Dockerized FastAPI (app.py) or tested locally using inference.py.

## Methodology & Experimental Design
As the project evolved from initial exploratory scripts into a structured pipeline, the codebase was refactored to enforce a unified evaluation protocol. This ensured a fair comparison across all iterations:
* **3-Class Sentiment Mapping:** 5-star ratings are mapped into three distinct categories: **Negative** (1–2 stars), **Neutral** (3 stars), and **Positive** (4–5 stars).
* **Validation Leakage Prevention:** Every experiment strictly separates an untouched **20% Test set** (`seed=42`) and carves out an additional **10% Validation set** from the training pool strictly for early stopping (`early_stopping_rounds` in XGBoost / `EarlyStoppingCallback` in BERT).
* **Class Imbalance Handling:** Because positive reviews heavily dominate the dataset while neutral reviews represent a small minority (~8.5%), all models apply inverse class weighting (`compute_sample_weight('balanced')` in XGBoost and a custom weighted `CrossEntropyLoss` inside `CustomTrainer` for BERT).
* **Primary Metric (Macro F1):** Models are evaluated and selected using **Macro F1** rather than Accuracy, penalizing models that ignore the minority `Neutral` class.

## Results

| Approach | Primary Metric: Macro F1 | Accuracy | Weighted F1 | F1: Negative (1–2) | F1: Neutral (3) | F1: Positive (4–5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Majority Class Baseline (All Positive)**| **0.262** | 0.648 | 0.509 | 0.000 | 0.000 | 0.786 |
| **1. Tabular + XGBoost** (`customer-behavior.py`) | **0.447** | 0.634 | 0.677 | 0.438 | 0.132 | 0.770 |
| **2. TF-IDF + XGBoost** (`reviews-tfidf.py`) | **0.612** | 0.725 | 0.761 | 0.735 | 0.264 | 0.838 |
| **3. SpaCy Vec (300d) + XGBoost** (`reviews-lg.py`) | **0.628** | 0.805 | 0.799 | 0.770 | 0.225 | 0.888 |
| **4. Fine-tuned BERTimbau** (`reviews-trf.py`) | **0.705** | **0.841** | **0.854** | **0.831** | **0.357** | **0.928** |

### Key Analytical Findings

1. **Why Tabular Features Failed (`customer-behavior.py` — Macro F1: `0.447`):**
   Logistical metrics (delivery delays, freight costs, product dimensions, installment counts) seem to not reliably capture subjective customer satisfaction. While the model achieved a `0.68` Macro F1 on training folds, it dropped to `0.447` on the test set and failed on the `Neutral` class (`Precision: 0.093`, `F1: 0.132`). Customers frequently leave 1-star reviews due to product defects despite fast delivery, or 5-star reviews despite shipping delays, justifying the pivot to text analysis.
2. **Classical NLP Baseline (`reviews-tfidf.py` — Macro F1: `0.612`):**
   Switching from tabular data to lemmatized Portuguese text vectorized via TF-IDF increased Macro F1 by **+16.5 percentage points**. Due to balanced sample weights, the model aggressively targeted the `Neutral` class, achieving a high recall (`0.457`) but suffering from low precision (`0.186`), as it misclassified 905 positive and 467 negative reviews as neutral.
3. **The Static Embedding Averaging Trap (`reviews-lg.py` — Macro F1: `0.628`):**
   Replacing TF-IDF with 300-dimensional static word embeddings (`pt_core_news_lg`) improved overall Accuracy from `0.725` to `0.805`, but **degraded performance on the `Neutral` class** (`F1: 0.225` vs `0.264` in TF-IDF, with Recall dropping from `45.7%` to `19.0%`). Additionally, 5-fold cross-validation revealed severe overfitting on dense vectors (`~0.955` Train Macro F1 vs `~0.610` Validation Macro F1).
4. **Contextual Deep Learning Superiority (`reviews-trf.py` — Macro F1: `0.705`):**
   Fine-tuning BERTimbau (`neuralmind/bert-base-portuguese-cased`) with mixed precision (`fp16`) and custom class weights outperformed all baselines across every metric (+7.7 pp Macro F1 over SpaCy embeddings, +25.8 pp over tabular features):
   * **Nuance & Contrast Detection:** Self-attention mechanisms allowed the model to capture mixed sentiment in 3-star reviews, achieving the highest `Neutral` F1 score (`0.357`, combining `0.449` Recall with `0.296` Precision).
   * **Minimal Polarity Errors:** On the 8,012-sample test set, BERT misclassified a genuinely negative review as positive only **61 times** (<2.8% of negative cases), compared to 250 times in SpaCy+XGBoost and 848 times in the tabular model.

## Repository Structure
- `src/`: The final, production-ready NLP pipeline.
  * `reviews-trf.py`: Fine-tunes the BERTimbau model using mixed precision (`fp16`), connects to GCP BigQuery, and implements a weighted `CustomTrainer` to handle class imbalance.
  * `report.py`: Evaluation script generating a detailed classification report and confusion matrix on the untouched test set.
  * `inference.py`: Inference script to test the trained model on new, raw sentences.
  * `app.py`: Web server endpoint for Docker deployment.
- `experiments/`: Analytical baselines and earlier iterations.
  * `customer-behavior.py`: Initial approach evaluating XGBoost on logistical and behavioral tabular features.
  * `reviews-tfidf.py`: NLP baseline utilizing SpaCy lemmatization and TF-IDF vectorization with XGBoost.
  * `reviews-lg.py`: NLP baseline utilizing 300-dimensional SpaCy word embeddings (`pt_core_news_lg`) with XGBoost.
- `sql/`: DDL/DML scripts generating BigQuery views.
  * `feature-extraction.sql`: View definition aggregating logistical, geographical, and sales features.
  * `review-extraction.sql`: View definition isolating text data from reviews for the NLP pipeline.
- `dev-requirements.txt`: Full dependency list for local GPU training and data acquisition.
- `api-requirements.txt`: Lightweight dependency list strictly for Dockerized API inference.
- `Dockerfile`: Instructions to build the inference container.

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

3. **Install dependencies:**
   * **For Model Training & Development (full setup):**
     ```bash
     # With NVIDIA GPU (strongly recommended):
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

     # CPU-only / macOS (strongly discouraged — BERT fine-tuning on CPU is prohibitively slow):
     pip install -r dev-requirements.txt
     ```
   * **For inference from existing model and API Only (lightweight setup):**
     ```bash
     pip install -r api-requirements.txt
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

**2. Evaluate the trained model on the test set:**
```bash
python src/report.py
```

**3. Test inference locally on sample sentences:**
```bash
python src/inference.py
```

**4. Deploy via Docker:**
```bash
docker build -t olist-sentiment-api .
docker run -p 8000:8000 olist-sentiment-api
```