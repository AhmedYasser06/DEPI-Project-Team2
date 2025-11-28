FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY app/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app /app

# Copy model artifacts
COPY artifacts /app/artifacts

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
