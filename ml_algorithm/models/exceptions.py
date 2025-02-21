class ModelNotFoundException(Exception):
    """Raised when the model file is not found."""
    pass

class PredictionError(Exception):
    """Raised when the prediction process encounters an error."""
    pass