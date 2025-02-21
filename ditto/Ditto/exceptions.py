# Ditto/exceptions.py

class DittoAPIException(Exception):
    """
    A custom exception for errors related to the Ditto API.
    This can be used when Ditto responds with a status code that indicates an error.
    """
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class NetworkPropertiesValidationError(Exception):
    """
    Custom exception to handle validation errors related to the NetworkProperties model.
    """
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class EventStreamParseException(Exception):
    """
    Custom exception to handle errors occurring while parsing the event stream data.
    """
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
