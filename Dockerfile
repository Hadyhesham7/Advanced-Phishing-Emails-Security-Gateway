FROM python:3.13-slim

# Set environment variables to optimize Python performance
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (e.g., build tools for packages if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first to leverage Docker layer caching
COPY requirements.txt /app/

# We also need FastAPI and Uvicorn which aren't in the base requirements.txt
RUN pip install --no-cache-dir fastapi>=0.100.0 uvicorn>=0.22.0 python-multipart>=0.0.6
# Install lightweight CPU-only PyTorch first to avoid 2.5GB CUDA blobs
RUN pip install --no-cache-dir torch>=2.2.0 --index-url https://download.pytorch.org/whl/cpu
# Install remaining Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model during build
RUN python -m spacy download en_core_web_sm

# Copy the entire project code into the container
COPY . /app/

# Expose the API server port
EXPOSE 8002

# Run the API server using python
CMD ["python", "api_server.py"]
