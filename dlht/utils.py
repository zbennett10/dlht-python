"""
Utility functions for LEAD DHT
"""

import hashlib
import struct
import json
from typing import Tuple


def sha1_hash(data: bytes) -> int:
    """Compute SHA-1 hash and return as integer"""
    return int.from_bytes(hashlib.sha1(data).digest(), 'big')


def peer_hash(ip: str, port: int) -> int:
    """Generate peer ID using consistent hashing"""
    data = f"{ip}:{port}".encode()
    return sha1_hash(data)


def distance(a: int, b: int, space_size: int) -> int:
    """Calculate ring distance from a to b"""
    if b >= a:
        return b - a
    return space_size - a + b


class RPCMessage:
    """RPC message encoding/decoding"""
    
    @staticmethod
    def encode(msg_type: str, payload: dict) -> bytes:
        """Encode message to bytes"""
        message = {'type': msg_type, 'payload': payload}
        data = json.dumps(message).encode()
        header = struct.pack('!I', len(data))
        return header + data
        
    @staticmethod
    def decode(data: bytes) -> Tuple[str, dict]:
        """Decode message from bytes"""
        message = json.loads(data.decode())
        return message['type'], message['payload']