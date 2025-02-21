import os

# Configuration settings for Ditto
DITTO_API_BASE_URL = os.getenv("DITTO_API_BASE_URL", "http://192.168.0.120:8080")
DITTO_USERNAME = os.getenv("DITTO_USERNAME", "ditto")
DITTO_PASSWORD = os.getenv("DITTO_PASSWORD", "ditto")