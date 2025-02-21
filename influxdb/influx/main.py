from fastapi import FastAPI, HTTPException
from influxdb.influx.database import write_to_influx
from influxdb.influx.models import DataPoint

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, World"}

@app.post("/data")
async def save_data(data: DataPoint):
    """
    Endpoint to save data to InfluxDB.
    :param data: DataPoint object containing metrics.
    """
    try:
        write_to_influx(data)  # Write data to InfluxDB
        return {"message": "Data written successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))