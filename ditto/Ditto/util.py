import httpx
from ditto.Ditto.config import DITTO_API_BASE_URL, DITTO_USERNAME, DITTO_PASSWORD

# Common HTTP client to be reused
async def get_httpx_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=DITTO_API_BASE_URL,
        auth=(DITTO_USERNAME, DITTO_PASSWORD),
        timeout=None  # or set an appropriate timeout
    )