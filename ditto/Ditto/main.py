from fastapi import FastAPI
import logging

# Set up logging at the start of the file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Simple FastAPI endpoint
@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Hello, World!"}