import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_sentiment(text: str, model, tokenizer, sentiment_map: dict) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return sentiment_map[predicted_class]

def main():
    model_path = "./olist_bert_model"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    sentiment_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    test_sentences = [
        "Parabéns aos envolvidos, o celular parou de funcionar na primeira semana. Um peso de papel muito caro.",
        "Eu adoraria dizer que é bom e recomendar para todos, mas infelizmente veio faltando peças importantes.",
        "O produto cumpre o que promete, mas achei o acabamento um pouco frágil pelo preço cobrado.",
        "Comprei para a minha mãe, ela ainda está testando. O tamanho é o que diz na descrição.",
        "Não esperava muito, mas quebra um galho enorme no dia a dia. Vale o custo-benefício.",
        "Entrega no prazo. Tudo nos conformes."
    ]

    for sentence in test_sentences:
        print(f"Text: {sentence}")
        print(f"Sentiment: {predict_sentiment(sentence, model, tokenizer, sentiment_map)}\n")

if __name__ == "__main__":
    main()