from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Literal
from enum import Enum

class OptimizationMethod(str, Enum):
    MINIMIZE_RCL = "minimize_rcl"
    MINIMIZE_OVERHEAD = "minimize_overhead"

class PredictionRequest(BaseModel):
    hello_interval: int = Field(..., ge=1, le=7)
    tc_interval: int = Field(..., ge=3, le=11)
    window_size: int = Field(..., ge=8, le=32)
    avg_neighbors: float = Field(default=10.08333333)
    std_neighbors: float = Field(default=1.815792237)

class PredictionResponse(BaseModel):
    overhead: float
    rcl: float
    parameters: dict

class OptimalParametersResponse(BaseModel):
    optimization_method: OptimizationMethod
    optimal_parameters: dict
    overhead: float
    rcl: float


class AlgorithmResponse(BaseModel):
    message: str = Field(..., description="Confirmation message with selected algorithm")

class ThresholdResponse(BaseModel):
    type: Literal["rcl", "overhead"] = Field(..., description="Type of threshold")
    value: float = Field(..., description="Current threshold value")

class ThresholdUpdateResponse(BaseModel):
    message: str = Field(..., description="Confirmation message with updated threshold values")