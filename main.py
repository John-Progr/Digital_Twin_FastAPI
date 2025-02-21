import asyncio
from fastapi import FastAPI, HTTPException
from ditto.Ditto.service import get_digital_twin_state, send_message_to_ditto
from influxdb.influx.database import write_to_influx
from ml_algorithm.OLSR_algorithm.service import MLService
from typing import Dict
import numpy as np
from pydantic_settings import BaseSettings
import logging
from loguru import logger
from ditto.Ditto.dependencies import get_http_client,get_ditto_credentials,get_base_url
from typing import AsyncIterator
import httpx
import time
from fastapi import Body
from typing import Optional
from influxdb.influx.models import DataPoint
import random



# Initialize FastAPI app
app = FastAPI()

# Buffer to temporarily store states
buffer: Dict[str, dict] = {}

# Expected 4 states for testing
EXPECTED_STATES = 4

ml_service = MLService()

# Previous state variables to detect changes
previous_avg_d = None
previous_std_d = None

def generate_test_states():
    # Generate random hl_int and tc_int values once
    hl_int = random.randint(1, 10)  # Random integer between 1 and 10 for hl_int
    tc_int = random.randint(1, 10)  # Random integer between 1 and 10 for tc_int
    
    return {
        f"thing_{i}": {
            "thingId": f"thing_{i}",
            "features": {
                "network": {
                    "properties": {
                        "neighbors": random.sample(range(EXPECTED_STATES), random.randint(1, EXPECTED_STATES)),  # Random number of neighbors
                        "hl_int": hl_int,  # Same hl_int for all states
                        "tc_int": tc_int   # Same tc_int for all states
                    }
                }
            }
        } for i in range(EXPECTED_STATES)
    }


# Define a helper function for retry logic
async def retry_request(func, retries=5, delay=1, backoff=2, *args, **kwargs):
    """Retries an async function call with exponential backoff."""
    attempt = 0
    while attempt < retries:
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTPError on attempt {attempt + 1}: {e}")
        except httpx.RequestError as e:
            logger.error(f"RequestError on attempt {attempt + 1}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
        
        attempt += 1
        if attempt < retries:
            # Exponential backoff
            sleep_time = delay * (backoff ** attempt)
            logger.info(f"Retrying in {sleep_time} seconds...")
            await asyncio.sleep(sleep_time)
    # After all retries, raise an exception
    logger.error(f"Failed after {retries} retries.")
    raise Exception("Maximum retries reached")

# Function to fetch the selected algorithm
"""
async def fetch_selected_algorithm():
    async def request_algorithm():
        async with httpx.AsyncClient() as client:
            response = await client.get("http://ml_algorithm:8000/ml/selected-algorithm")
            response.raise_for_status()  # Will raise an HTTPStatusError for non-2xx responses
            data = response.json()  # Parse the JSON response
            algorithm_name = data.get("message", "").split(":")[1].strip()  # Extract algorithm name
            return algorithm_name

    return await retry_request(request_algorithm)
"""

"""
# Function to fetch the selected threshold
async def fetch_selected_threshold():
    async def request_threshold():
        async with httpx.AsyncClient() as client:
            response = await client.get("http://ml_algorithm:8000/ml/selected_threshold")
            response.raise_for_status()  # Will raise an HTTPStatusError for non-2xx responses
            data = response.json()  # Parse the JSON response
            threshold_value = data.get("value", 0.5)  # Extract the threshold value
            return threshold_value

    return await retry_request(request_threshold)
"""
async def get_optimal_parameters(avg_d: float, std_d: float):
    async def request_optimization():
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://ml_algorithm:8000/ml/optimize-parameters",
                params={
                    "avg_neighbors": avg_d,
                    "std_neighbors": std_d
                }
            )
            response.raise_for_status()  # Will raise an HTTPStatusError for non-2xx responses
            data = response.json()  # Parse the JSON response
            return data  # Return the complete response data

    return await retry_request(request_optimization)

def calculate_avg_and_std(states):
    degrees = [
        len(state.get("features", {}).get("network", {}).get("properties", {}).get("neighbors", []))
        for state in states.values()
    ]
    return {"avg_neighbors": np.mean(degrees), "std_neighbors": np.std(degrees)}


async def process_states_batch(buffer):
    global previous_avg_d, previous_std_d
    stats = calculate_avg_and_std(buffer)
    avg_neighbors, std_neighbors = stats["avg_neighbors"], stats["std_neighbors"]
    logger.info(f"avg_neighbors: {avg_neighbors}, std_neighbors: {std_neighbors}")

    hl_values = set()
    tc_values = set()

    # Collect hl_int and tc_int values from all states
    for thing_id, state in buffer.items():
        logger.info(f"thing_id {thing_id}")
        features = state.get("features", {}).get("network", {}).get("properties", {})
        hl_int = features.get("hl_int")
        tc_int = features.get("tc_int")

         # Only add non-None values to the sets
        if hl_int is not None:
          hl_values.add(hl_int)
        if tc_int is not None:
          tc_values.add(tc_int)

    logger.info(f"hl_values: {hl_values}, tc_values: {tc_values}")

    # Check if all values are the same
    if len(hl_values) == 1 and len(tc_values) == 1:
        # Extract the single consistent values
        hl_int = next(iter(hl_values))  # Get the only value in the set
        tc_int = next(iter(tc_values))  # Get the only value in the set

        # Create a single DataPoint and write once
        data_point = DataPoint(
            hl_int=hl_int,
            tc_int=tc_int,
            avg_d=avg_neighbors,
            std_d=std_neighbors
        )

        logger.info(f"Writing single DataPoint to DB: {data_point}")
        write_to_influx(data_point)  # Single write

    else:
        logger.info("Skipped writing to database: hl_int or tc_int values are inconsistent.")
   
        
       
    
    if avg_neighbors != previous_avg_d or std_neighbors != previous_std_d:

        previous_avg_d, previous_std_d = avg_neighbors, std_neighbors
        logger.info(f"CURRENT AVERAGE_NEIGHBOR: {avg_neighbors}")
        logger.info(f"PREVIOUS AVERAGE_NEIGHBOR: {previous_avg_d}")
        #selected_algorithm = await fetch_selected_algorithm()
        #print("SELECTED_ALGORITHM")
        #print(selected_algorithm)
        #print("SELECTED_THRESHOLD")
       
        #threshold = await fetch_selected_threshold()
        #print(threshold)
        print("GETTING THE RESPONSE")
        response = await get_optimal_parameters(avg_d=avg_neighbors, std_d=std_neighbors)
       
    
        # Extract the optimization result
        #optimal_params = response["optimization_result"]
        # Extract optimization result correctly
        optimization_result = response.get("optimization_result", {})
        optimal_parameters = optimization_result.get("optimal_parameters", {})

        # Build the payload
        payload = {
             "hl_int": float(optimal_parameters.get("hello_interval", 0)),  # Default to 0 if key is missing
             "tc_int": float(optimal_parameters.get("tc_interval", 0))
        }
        
        
        
        logger.info(f"Printing payload instead of sending:")
        
        #for thing_id in filter(None, buffer):
        #logger.info(f"thing_id: {thing_id}, payload: {payload}")
        
        # Send the updated parameters back to Ditto
        for thing_id in filter(None, buffer):
            logger.info(f"thing_id: {thing_id}, payload: {payload}")
            await send_message_to_ditto(payload, thing_id)
            logger.info(f"FINISHED SENDING THE MESSAGES")
     


@app.get("/start_task/")
async def start_task():
    buffer = generate_test_states()  # Generate the test states
    await process_states_batch(buffer)  # Process the generated states
    return {"message": "Test states generated and processed!"}




@app.on_event("startup")
async def start_background_task():
    """Start the infinite background task on startup."""
    asyncio.create_task(fetch_and_process_data())


async def fetch_and_process_data():
    """Fetch data from Ditto, wait for all expected states, and process them."""
    global buffer

    async for state in get_digital_twin_state():
        #logger.info(f"These are the states")
        #logger.info(f"{state}")
        thing_id = state.get("thingId")



        # Add the state to the buffer
        buffer[thing_id] = state
        #logger.info(f"This is the buffer")
        #logger.info(f"{buffer}")

        # Wait for a short timeout to gather all states if they're not fully synchronized
        await asyncio.sleep(0.3)

        # Process the batch once all expected states are gathered
        if len(buffer) > EXPECTED_STATES:
            await process_states_batch(buffer)
            buffer.clear()