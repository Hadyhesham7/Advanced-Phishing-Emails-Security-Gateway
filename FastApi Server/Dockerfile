FROM python:3.13-slim

WORKDIR /app

# Install system dependencies required by OCR and ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
# Note: Assuming this Dockerfile is run with the build context at "Last_phishing" (the parent dir)
COPY ["FastApi Server/requirements.txt", "./requirements.txt"]
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire pipeline and FastApi Server code
COPY . /app

# Expose port 8002 for the Orchestrator
EXPOSE 8002

# Run the API server
CMD ["python", "FastApi Server/api_server.py"]
