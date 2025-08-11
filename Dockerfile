FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /app/logs /app/outputs /app/models /app/loras
# Add this to your Dockerfile to fix permissions
RUN mkdir -p /app/models /app/loras /app/outputs /app/logs
RUN chmod -R 777 /app/models /app/loras /app/outputs /app/logs

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables to disable progress bars and reduce logging
ENV DIFFUSERS_PROGRESS_BAR=0
ENV TRANSFORMERS_PROGRESS_BAR=0
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Expose port for API
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "text2img:app", "--host", "0.0.0.0", "--port", "8000"]