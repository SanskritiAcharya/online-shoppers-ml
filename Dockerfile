# One image, two services (API and UI). docker-compose.yml picks the command.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what serving needs: the code, the saved pipeline and its metadata, and the result tables
# the Streamlit app reads. Training data is not needed to serve.
COPY src/ src/
COPY app/ app/
COPY .streamlit/ .streamlit/
COPY models/ models/
COPY reports/*.csv reports/

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
