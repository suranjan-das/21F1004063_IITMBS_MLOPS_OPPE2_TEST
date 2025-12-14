FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8200
ENV MODEL_PATH=/app/heart_model.joblib

CMD ["uvicorn", "model_log:app", "--host", "0.0.0.0", "--port", "8200"]
