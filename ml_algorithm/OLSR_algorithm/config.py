from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "models/random_forest_model.pkl"
    HELLO_INTERVAL: list[int] = [1, 2, 3, 4, 5, 6, 7]
    TC_INTERVAL: list[int] = [3, 5, 7, 9, 11]
    WINDOW_SIZE: list[int] = [8, 16, 32]
    AVG_NEIGHBORS: float = 10.08333333
    STD_NEIGHBORS: float = 1.815792237
    THRESHOLD: float = 40
    RCL_THRESHOLD: float = 8.29484
    SELECTED_ALGORITHM: str = "minimize_rcl"

settings = Settings()
