from pydantic import BaseModel

class DataPoint(BaseModel):
    hl_int: float
    tc_int: float
    avg_d: float
    std_d: float