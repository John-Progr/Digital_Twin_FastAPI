from fastapi import Depends
from contextlib import asynccontextmanager
import httpx
from ditto.Ditto.config import DITTO_API_BASE_URL, DITTO_USERNAME, DITTO_PASSWORD
from typing import AsyncIterator

# Dependency for HTTP Client
@asynccontextmanager
async def get_http_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:  # Using async with to manage client lifecycle
        yield client  # Yield client for use
    # No need to manually close the client, async with will do this automatically

# Dependency for Base URL
def get_base_url() -> str:
    return DITTO_API_BASE_URL

# Dependency for Authentication
def get_ditto_credentials() -> tuple:
    return DITTO_USERNAME, DITTO_PASSWORD
