from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    influx_url: str = os.getenv("INFLUX_URL", "http://localhost:8086")
    influx_token: str = open("./influxdb/secrets/influxdb2-admin-token").read().strip()
    influx_org: str = os.getenv("INFLUX_ORG", "default-org")


settings = Settings()