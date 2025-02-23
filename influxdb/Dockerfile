# Use the official Python 3.9 slim image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy the requirements file 
COPY requirements.txt .

# Install Python dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the InfluxDB-related code into the container
COPY influx /app/influx

# Copy the secrets directory to the container 
COPY secrets /app/secrets

# Set Python path
ENV PYTHONPATH=/app

# Expose the port that the FastAPI application will run on
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "uvicorn", "influx.main:app", "--host", "0.0.0.0", "--port", "8000"]
