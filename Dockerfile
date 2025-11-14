FROM python:3.10-slim

WORKDIR /app

# Install Tesseract OCR dependencies
RUN apt-get update && \
    apt-get install -y tesseract-ocr libleptonica-dev libtesseract-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Default command to run FastAPI
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
