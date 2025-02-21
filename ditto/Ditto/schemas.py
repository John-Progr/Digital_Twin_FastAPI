from pydantic import BaseModel
from typing import Optional, List, Dict


# Define a model for a Neighbor's IP address
class Neighbor(BaseModel):
    ipv4Address: str  # IP address of the neighboring device

# Define NetworkProperties for the network-related features
class NetworkProperties(BaseModel):
    neighbors: List[Neighbor]  # List of Neighbor instances
    hl_int: Optional[float]  # High-level interval (e.g., 10.0)
    tc_int: Optional[float]  # TC interval (e.g., 5.0)
    error: Optional[int]  # Error state or message, if applicable (assuming it's an integer based on provided data)

# Define Features to encapsulate network-related features
class Features(BaseModel):
    network: NetworkProperties  # The network properties

# Define the Digital Twin structure
class DigitalTwin(BaseModel):
    thingId: str  # Unique identifier for the digital twin object
    features: Features  # The features (network) of the digital twin

