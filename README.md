# Project Name

This project is designed to run multiple services using Docker Compose, including **FastAPI**, **Ditto**, **ML Algorithm**, and **InfluxDB**. All services are set up with version **3.9** of Docker Compose.

## Table of Contents

1. [Overview](#overview)
2. [Services](#services)
   - [Ditto](#ditto)
   - [ML Algorithm](#ml-algorithm)
   - [InfluxDB](#influxdb)
   - [FastAPI](#fastapi)
3. [Getting Started](#getting-started)
   - [Building the Services](#building-the-services)
   - [Running the Services](#running-the-services)
4. [Ports & Volumes](#ports--volumes)
5. [Secrets & Environment Variables](#secrets--environment-variables)
6. [Contributing](#contributing)

## Overview

This project uses Docker Compose to define and run multi-container Docker applications. The services running in this project are:

- **Ditto**: A custom service that interacts with external APIs.
- **ML Algorithm**: A service that runs machine learning algorithms.
- **InfluxDB**: A time-series database for storing and querying metrics data.
- **FastAPI**: A web framework for building APIs, used to interact with the other services.

All services are linked together, and Docker handles the orchestration, allowing you to focus on the application logic.

## Services

### Ditto

Ditto is a service built from the `./ditto` directory. It interacts with external systems through APIs.

- **Ports:** 
  - Host: `8001` → Container: `8000`
- **Volumes:** 
  - Mounts `./ditto` to `/app/ditto` inside the container.
- **Environment Variables:**
  - `DITTO_API_BASE_URL`
  - `DITTO_USERNAME`
  - `DITTO_PASSWORD`

### ML Algorithm

This service is responsible for running machine learning algorithms. It is built from the `./ml_algorithm` directory.

- **Ports:**
  - Host: `8002` → Container: `8000`
- **Volumes:**
  - Mounts `./ml_algorithm` to `/app/ml_algorithm`
  - Mounts `./models/plots` to `/app/models/plots`
- **Environment Variables:** Loaded from `.env` file.

### InfluxDB

InfluxDB is a time-series database used to store and query metrics. This service uses the InfluxDB Docker image (`influxdb:2.7.5`).

- **Ports:** 
  - Host: `8086` → Container: `8086`
- **Environment Variables:** 
  - `DOCKER_INFLUXDB_INIT_MODE: setup`
  - `DOCKER_INFLUXDB_INIT_USERNAME_FILE`
  - `DOCKER_INFLUXDB_INIT_PASSWORD_FILE`
  - `DOCKER_INFLUXDB_INIT_ADMIN_TOKEN_FILE`
  - `DOCKER_INFLUXDB_INIT_ORG`
  - `DOCKER_INFLUXDB_INIT_BUCKET`
- **Secrets:**
  - Loaded from the `./influxdb/secrets/` directory.
- **Volumes:**
  - `influxdb2-data`: For data persistence.
  - `influxdb2-config`: For InfluxDB configuration.

### FastAPI

FastAPI is the main web service that interacts with the other services (Ditto, ML Algorithm, and InfluxDB).

- **Ports:**
  - Host: `8000` → Container: `8000`
- **Volumes:**
  - Mounts the entire project folder to `/app`.
- **Command:**
  - Starts FastAPI using `uvicorn`: 
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
- **Dependencies:** 
  - Depends on `Ditto`, `ML Algorithm`, and `InfluxDB`.

## Getting Started

### Building the Services

Before running the services, you need to build them using the following command:

```bash
docker-compose build
