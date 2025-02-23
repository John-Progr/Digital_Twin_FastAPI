import pickle
import itertools
import numpy as np
from loguru import logger
from typing import Tuple, List, Dict, Any
from .config import settings  
from ml_algorithm.models.constants import ModelConstants  
from ml_algorithm.models.exceptions import ModelNotFoundException, PredictionError  
from .schemas import OptimizationMethod  
from ml_algorithm.models.model_loader import load_rf_model, predict_overhead, calculate_rcl
import logging
import os
from fastapi import Body
from typing import Optional
import pandas as pd
class MLService:
    def __init__(self):
        """
        Initializes the MLService and loads the model from the specified path.
        """
        model_path = os.getenv('MODEL_PATH')
        if model_path is None:
            raise ValueError("MODEL_PATH environment variable not set.")
        try:
            self.model = load_rf_model(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise ModelNotFoundException(f"Error loading model: {str(e)}")
        
    def predict_overhead(self, features: List[float]) -> float:
        """
        Makes a prediction using the loaded model.
        """
        try:
            return predict_overhead(self.model, features)
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise PredictionError(f"Prediction error: {str(e)}")

    def calculate_rcl(self, h: float, t: float, w: float, avg_neighbors: float, std_neighbors: float) -> float:
        """
        Calculates the RCL value using the predefined coefficients.
        """
        try:
            return calculate_rcl(h, t, w, avg_neighbors, std_neighbors)
        except Exception as e:
            logger.error(f"Error calculating RCL: {str(e)}")
            raise PredictionError(f"Error calculating RCL: {str(e)}")

    def find_optimal_parameters(self, optimization_method: str, threshold: float, avg_d: float, std_d: float) -> Dict[str, Any]:
       logger.info(f"Optimization Method: {optimization_method}")
       if optimization_method == "minimize_rcl":
          return self._find_optimal_parameters_minimize_rcl(threshold, avg_d, std_d)
       else:
          return self._find_optimal_parameters_minimize_overhead(threshold, avg_d, std_d)


    def _find_optimal_parameters_minimize_rcl(self,avg_neighbors: float,std_neighbors: float, threshold: float ) -> Dict[str, Any]:
       
        combinations = list(itertools.product(settings.HELLO_INTERVAL, settings.TC_INTERVAL, settings.WINDOW_SIZE))
        below_threshold_triplets = []
        logger.info(f"Using threshold: {threshold}")
        """
        h = 4
        t = 7
        w = 8
        x_to_predict = [[h, t, w, avg_neighbors, std_neighbors]]
        try:
            # Predict the overhead using the model
            overhead = predict_overhead(self.model, x_to_predict[0]) 
            rcl = self.calculate_rcl(h, t, w, avg_neighbors, std_neighbors) # Pass the first row of x_to_predict
            return {
               'inputs': {
                   'hello_interval': h,
                   'tc_interval': t,
                   'window_size': w,
                   'avg_neighbors': avg_neighbors,
                   'std_neighbors': std_neighbors
                },
               'predicted_overhead': overhead,
               'rcl': rcl
            }
        except PredictionError as e:
             logger.error(f"Prediction error for inputs {(h, t, w, avg_neighbors, std_neighbors)}: {str(e)}")
             return {
              'inputs': {
                'hello_interval': h,
                'tc_interval': t,
                'window_size': w,
                'avg_neighbors': avg_neighbors,
                'std_neighbors': std_neighbors
              },
              'predicted_overhead': None,
              'rcl': None,
              'error': str(e)
           }
        """
        for h, t, w in combinations:

            features = [h, t, w,avg_neighbors,std_neighbors]
           
            try:
                overhead = predict_overhead(self.model, features)
                if overhead < threshold:
                  below_threshold_triplets.append((h, t, w, overhead))


            except PredictionError as e:
               logger.error(f"Prediction error for combination {(h, t, w)}: {str(e)}")
               continue
        if not below_threshold_triplets:
           return {
               'optimization_method': OptimizationMethod.MINIMIZE_RCL,
               'optimal_parameters': None,
               'overhead': None,
               'rcl': 8045
            }

        min_rcl = float('inf')
        optimal_params = None


        
        for h, t, w, ovhd in below_threshold_triplets:
            try:

               rcl = self.calculate_rcl(h, t, w, avg_neighbors, std_neighbors)
               if rcl < min_rcl:
                   min_rcl = rcl
                   optimal_params = {
                      'hello_interval': h,
                      'tc_interval': t,
                      'window_size': w,
                      'overhead': ovhd,
                      'rcl': rcl
                    }

            except PredictionError as e:
              logger.error(f"RCL calculation error for combination {(h, t, w)}: {str(e)}")
              continue
        return {
            'optimization_method': OptimizationMethod.MINIMIZE_RCL,
            'optimal_parameters': optimal_params,
            'overhead': optimal_params['overhead'] if optimal_params else None,
            'rcl': optimal_params['rcl'] if optimal_params else 54

        
        }
    

    def _find_optimal_parameters_minimize_overhead(self,avg_neighbors: float,std_neighbors: float,threshold: float ) -> Dict[str, Any]:
        combinations = list(itertools.product(settings.HELLO_INTERVAL, settings.TC_INTERVAL, settings.WINDOW_SIZE))
        close_to_target_triplets = []

        for h, t, w in combinations:
            rcl = self.calculate_rcl(h, t, w, avg_neighbors, std_neighbors)
            if abs(threshold - rcl) < threshold :
                close_to_target_triplets.append((h, t, w, rcl))

        min_overhead = float('inf')
        optimal_params = None
        
        for h, t, w, rcl in close_to_target_triplets:
            features = [h, t, w, settings.AVG_NEIGHBORS, settings.STD_NEIGHBORS]
            overhead = self.predict_overhead(features)
            if overhead < min_overhead:
                min_overhead = overhead
                optimal_params = {
                    'hello_interval': h,
                    'tc_interval': t,
                    'window_size': w,
                    'overhead': overhead,
                    'rcl': rcl
                }

        return {
            'optimization_method': OptimizationMethod.MINIMIZE_OVERHEAD,
            'optimal_parameters': optimal_params,
            'overhead': optimal_params['overhead'] if optimal_params else None,
            'rcl': optimal_params['rcl'] if optimal_params else None
        }

    def select_algorithm_service(self, algorithm: str) -> Dict[str, str]:
        """
        Service to select the algorithm for ML processing.
        """
        try:
         
            settings.SELECTED_ALGORITHM = algorithm.lower()  # Optional: Ensure it is case-insensitive
            logger.info(f"Algorithm successfully set to: {algorithm}")
            return {"message": f"Selected algorithm: {algorithm}"}
        except ValueError as e:
            logger.error(f"Invalid algorithm attempted: {algorithm}")
            raise HTTPException(status_code=400, detail=f"Invalid algorithm. Must be one of: {[algo.value for algo in Algorithm]}")
        except Exception as e:
            logger.error(f"Unexpected error in select_algorithm_service: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def update_thresholds_service(self, rcl_threshold: Optional[float] = None, overhead_threshold: Optional[float] = None) -> Dict[str, str]:
        """
        Service to update the RCL and Overhead thresholds.
        """
        try:
            if rcl_threshold is not None:
                settings.RCL_THRESHOLD = rcl_threshold
                logger.info(f"RCL threshold updated to: {rcl_threshold}")

            if overhead_threshold is not None:
                settings.THRESHOLD = overhead_threshold
                logger.info(f"Overhead threshold updated to: {overhead_threshold}")

            if rcl_threshold is None and overhead_threshold is None:
                logger.warning("Update thresholds called with no values")
                return {"message": "No thresholds were updated"}

            return {
                "message": (
                    f"Updated thresholds: "
                    f"RCL_THRESHOLD = {getattr(settings, 'RCL_THRESHOLD', 'unchanged')}, "
                    f"OVERHEAD_THRESHOLD = {getattr(settings, 'THRESHOLD', 'unchanged')}"
                )
            }

        except ValueError as e:
            logger.error(f"Invalid threshold value: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error in update_thresholds_service: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def get_threshold_configuration(self) -> Dict[str, float]:
        """
        Returns the currently active threshold configuration from settings.
        """
        try:
            selected_algorithm = getattr(settings, 'SELECTED_ALGORITHM', None)
            logger.info(f"{selected_algorithm}")

            if selected_algorithm == "minimize_rcl":
                if not hasattr(settings, 'THRESHOLD'):
                    raise ValueError("RMO threshold not configured")
                return {"type": "overhead", "value": settings.THRESHOLD}
            elif selected_algorithm == "minimize_overhead":
                if not hasattr(settings, 'RCL_THRESHOLD'):
                    raise ValueError("RCL threshold not configured")
                return {"type": "rcl", "value": settings.RCL_THRESHOLD}
            else:
                raise ValueError("No algorithm selected")
        except ValueError as e:
            logger.error(f"Configuration error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error in get_threshold_configuration: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def get_selected_algorithm(self) -> Dict[str, str]:
        """
        Returns the currently selected algorithm from settings.
        """
        try:
            selected_algorithm = getattr(settings, 'SELECTED_ALGORITHM', None)
            if selected_algorithm is None:
                raise ValueError("No algorithm currently selected")
            logger.info(f"Retrieved current algorithm: {selected_algorithm}")
            return {"message": f"Current algorithm: {selected_algorithm}"}
        except ValueError as e:
            logger.error(f"Configuration error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error in get_selected_algorithm: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")


    def print_current_thresholds(self):
      print(f"Current RCL_THRESHOLD: {getattr(settings, 'RCL_THRESHOLD', 'not set')}")
      print(f"Current OVERHEAD_THRESHOLD: {getattr(settings, 'THRESHOLD', 'not set')}")
