class BaseError(Exception):
    """
    Error class for all local exceptions in this code base
    """
    pass

class PredictionResultExistsError(BaseError):
    pass

class FileExistsError(BaseError):
    pass