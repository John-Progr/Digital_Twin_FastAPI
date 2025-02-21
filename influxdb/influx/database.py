from influxdb_client import InfluxDBClient, Point
from influxdb.influx.config import settings
from influxdb.influx.models import DataPoint


def write_to_influx(data: DataPoint):
    with InfluxDBClient(
        url=settings.influx_url,
        token=settings.influx_token,
        org=settings.influx_org
    ) as client:
        write_api = client.write_api()
        point = (
            Point("metrics")
            .field("hl_int", data.hl_int)
            .field("tc_int", data.tc_int)
            .field("avg_d", data.avg_d)
            .field("std_d", data.std_d)
        )
        write_api.write(bucket="metrics_bucket", org=settings.influx_org, record=point)