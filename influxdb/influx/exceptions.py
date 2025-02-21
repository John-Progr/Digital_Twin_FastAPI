class InfluxNotAvailableException(Exception):
    STATUS_CODE = 503
    DESCRIPTION = "Unable to connect to InfluxDB."


class BucketNotFoundException(Exception):
    STATUS_CODE = 404
    DESCRIPTION = "Bucket Not Found."


class BadQueryException(Exception):
    STATUS_CODE = 400
    DESCRIPTION = "Bad Query."