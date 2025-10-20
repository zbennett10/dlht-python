"""
Exception classes for LEAD DHT
"""


class LEADException(Exception):
    """Base exception for LEAD DHT"""
    pass


class NetworkException(LEADException):
    """Exception for network-related errors"""
    pass


class ModelException(LEADException):
    """Exception for model training/prediction errors"""
    pass


class KeyNotFoundException(LEADException):
    """Exception when a key is not found"""
    pass


class NodeNotReadyException(LEADException):
    """Exception when node is not ready for operations"""
    pass


class RPCException(NetworkException):
    """Exception for RPC failures"""
    pass


class StabilizationException(LEADException):
    """Exception during network stabilization"""
    pass