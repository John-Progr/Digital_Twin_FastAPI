# Use the official Python 3.9 slim image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if any needed, e.g., build tools, libraries)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (including main.py in the root folder)
COPY . .

# Set the Python path (for easy import of app modules)
ENV PYTHONPATH=/app

# Expose the port for FastAPI to communicate
EXPOSE 8000

# Command to run the FastAPI application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
