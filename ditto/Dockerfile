# Use the official Python 3.9 slim image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "uvicorn", "Ditto.main:app", "--host", "0.0.0.0", "--port", "8000"]