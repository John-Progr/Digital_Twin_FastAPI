import json
from typing import AsyncIterator
from fastapi import Depends
from ditto.Ditto.schemas import DigitalTwin
from ditto.Ditto.exceptions import DittoAPIException, NetworkPropertiesValidationError, EventStreamParseException
from ditto.Ditto.config import DITTO_API_BASE_URL
from pydantic import ValidationError
import httpx
from ditto.Ditto.util import get_httpx_client  # Import the utility function
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

async def get_digital_twin_state() -> AsyncIterator[dict]:
    url = f"{DITTO_API_BASE_URL}/api/2/things"
    print(f"Connecting to Ditto at: {url}")
    
    async with await get_httpx_client() as client:
        async with client.stream(
            "GET",
            url,
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            print(f"Connected with status: {response.status_code}")
            async for line in response.aiter_lines():
                print(f"Raw line received: {line}")
                if line:
                    yield parse_event_stream(line)



def parse_event_stream(line: str) -> dict:
    """
    Parse the event stream line.
    """
      # Check if the line is empty or just contains "data:"
    if not line or line.strip() == "data:":
        return {}  # Return an empty dictionary or None, depending on your preference
    if line.startswith("data:"):
        line = line[len("data:"):].strip()  # Remove 'data:' and any leading whitespace

    try:
        network_data = json.loads(line)
        logger.info(f"Raw line received: {network_data}")
        return network_data
    except json.JSONDecodeError as e:
        raise EventStreamParseException(f"Failed to parse event stream line: {line}")
    except ValidationError as e:
        raise NetworkPropertiesValidationError(f"Invalid network data structure: {e.errors()}")


async def send_message_to_ditto(
    payload: dict, 
    twin_id: str, 
) -> dict:
    """
    Send a message to the digital twin's inbox to reconfigure OLSRd.
    """
    url = f"{DITTO_API_BASE_URL}/api/2/things/{twin_id}/inbox/messages/olsr_reconfigure?timeout=0s"

    logger.info(f"Sending message to Ditto twin with ID {twin_id} at: {url}")
    
    
    async with await get_httpx_client() as client:
        response = await client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            auth=("ditto", "ditto"),
            timeout=None  # No timeout
        )
        