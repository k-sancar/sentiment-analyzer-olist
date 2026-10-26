from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.inference import predict_sentiment

MODEL_PATH = Path(__file__).parent / "olist_bert_model"

SENTIMENT_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_PATH)
    ml["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    yield
    ml.clear()


app = FastAPI(title="E-commerce Sentiment API", lifespan=lifespan)


class ReviewRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(request: ReviewRequest):
    prediction = predict_sentiment(
        request.text, ml["model"], ml["tokenizer"], SENTIMENT_MAP
    )
    return {"review": request.text, "sentiment": prediction}