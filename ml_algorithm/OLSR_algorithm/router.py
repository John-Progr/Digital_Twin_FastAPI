from fastapi import APIRouter, HTTPException, Body, Query
from typing import Dict, Optional, Any
import logging
from .service import MLService
from .schemas import AlgorithmResponse, ThresholdResponse, ThresholdUpdateResponse


# Configure logging
logger = logging.getLogger(__name__)

# Initialize router and service
router = APIRouter(
    prefix="/ml",
    tags=["ml"],
    responses={
        500: {"description": "Internal Server Error"},
        400: {"description": "Bad Request"}
    }
)

ml_service = MLService()

@router.post(
    "/select_algorithm",
    response_model=AlgorithmResponse,
    summary="Select ML algorithm",
    response_description="Returns the selected algorithm confirmation"
)
async def select_algorithm(
    algorithm: str = Body(..., embed=True, description="Algorithm to select (minimize_rcl or minimize_overhead)")
) -> AlgorithmResponse:
    """
    Endpoint to select the algorithm for ML processing.
    
    Args:
        algorithm (str): The algorithm to use for processing. Must be either 'minimize_rcl' or 'minimize_overhead'.
        
    Returns:
        AlgorithmResponse: Confirmation message with the selected algorithm.
        
    Raises:
        HTTPException: 400 if the algorithm is invalid, 500 for server errors.
    """
    try:
        return ml_service.select_algorithm_service(algorithm)
    except HTTPException as he:
        logger.error(f"HTTP error in select_algorithm: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in select_algorithm: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred"
        )

@router.post(
    "/update_thresholds",
    response_model=ThresholdUpdateResponse,
    summary="Update threshold values",
    response_description="Returns updated threshold values confirmation"
)
async def update_thresholds(
    rcl_threshold: Optional[float] = Body(None, description="New RCL threshold value"),
    overhead_threshold: Optional[float] = Body(None, description="New overhead threshold value")
) -> ThresholdUpdateResponse:
    """
    Endpoint to update the RCL and Overhead thresholds.
    
    Args:
        rcl_threshold (Optional[float]): New value for the RCL threshold .
        overhead_threshold (Optional[float]): New value for the overhead threshold .
        
    Returns:
        ThresholdUpdateResponse: Confirmation message with updated threshold values.
        
    Raises:
        HTTPException: 400 if values are invalid, 500 for server errors.
    """
    try:
        return await ml_service.update_thresholds_service(rcl_threshold, overhead_threshold)
    except HTTPException as he:
        logger.error(f"HTTP error in update_thresholds: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in update_thresholds: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred"
        )

@router.get(
    "/selected-algorithm",
    response_model=AlgorithmResponse,
    summary="Get current algorithm",
    response_description="Returns currently selected algorithm"
)
async def get_selected_algorithm() -> AlgorithmResponse:
    """
    Endpoint to get the currently selected ML algorithm.
    
    Returns:
        AlgorithmResponse: Currently selected algorithm information.
        
    Raises:
        HTTPException: 500 for server errors.
    """
    logger.info(f"EKANA TO REQUEST")
    try:
        logger.info(f"KSEKINISA")
        return await ml_service.get_selected_algorithm()
    except Exception as e:
        logger.error(f"Unexpected error in get_selected_algorithm: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred"
        )

@router.get(
    "/selected_threshold",
    response_model=ThresholdResponse,
    summary="Get current threshold",
    response_description="Returns currently active threshold configuration"
)
async def get_selected_threshold() -> ThresholdResponse:
    """
    Endpoint to get the current active threshold configuration (either RCL or Overhead).
    
    Returns:
        ThresholdResponse: Current active threshold type and value.
        
    Raises:
        HTTPException: 400 if no threshold is configured, 500 for server errors.
    """
    try:
        return await ml_service.get_threshold_configuration()
    except HTTPException as he:
        logger.error(f"HTTP error in get_selected_threshold: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in get_selected_threshold: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred"
        )
 
@router.get("/optimize-parameters")
async def optimize_parameters(
    avg_neighbors: Optional[float] = Query(None, description="Average number of neighbors"),
    std_neighbors: Optional[float] = Query(None, description="Standard deviation of neighbors")
) -> Dict[str, Any]:
    """
    Endpoint to verify parameter passing - currently just returns the input values.
    """
    try:
        # Get selected algorithm
        selected_algorithm = await ml_service.get_selected_algorithm()
        algorithm_name = selected_algorithm.get("message", "").split(":")[1].strip()
        
        # Get threshold configuration
        threshold_config = await ml_service.get_threshold_configuration()
        threshold_value = threshold_config.get("value", 0.5)
        logger.info(f"Optimization Method: {selected_algorithm}")
       # Log values
        logger.info(f"FINISHED algorithm_name: {algorithm_name}")
        logger.info(f"FINISHED settings threshold: {threshold_value}")
        
       
        optimal_params = ml_service.find_optimal_parameters(
            optimization_method = algorithm_name,
            threshold=threshold_value,  
            avg_d=avg_neighbors,  
            std_d=std_neighbors  
        )

        # Prepare response
        response = {
            "selected_algorithm": algorithm_name,
            "threshold_value": threshold_value,
            "avg_neighbors": avg_neighbors,
            "std_neighbors": std_neighbors,
            "optimization_result": optimal_params
        }
        
        logger.info(f"Response parameters: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Error in optimize_parameters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
