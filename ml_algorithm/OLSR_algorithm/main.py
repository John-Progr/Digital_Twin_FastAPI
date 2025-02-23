from fastapi import FastAPI,HTTPException
from OLSR_algorithm.service import MLService
from OLSR_algorithm.schemas import OptimalParametersResponse, OptimizationMethod
from OLSR_algorithm.router  import router
from OLSR_algorithm.config  import settings

from models.data_analysis import run_full_workflow_react
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost",  # React app running on this port during development
]

# Add the CORS middleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Specify the allowed origins
    allow_credentials=True,  # Allow cookies or other credentials
    allow_methods=["*"],  # Allow all HTTP methods like GET, POST, etc.
    allow_headers=["*"],  # Allow all headers
)


ml_service = MLService()
app.include_router(router)




# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Service is running"}



@app.get("/data-shift")
async def data_shift_analysis():
    """
    API endpoint to run the full workflow including:
    - Loading the dataset
    - Analyzing distributions
    - Generating synthetic data
    - Evaluating the Random Forest model
    Returns MSE, R² scores, and distribution analysis.
    """
    try:
        results = run_full_workflow_react()
        return results  
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
