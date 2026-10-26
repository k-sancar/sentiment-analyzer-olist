FROM python:3.12-slim

WORKDIR /app

COPY api-requirements.txt .

RUN pip install --no-cache-dir -r api-requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]