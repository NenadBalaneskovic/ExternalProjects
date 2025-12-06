"""
cache.py
Simple caching layer for API responses.
"""

import time

class Cache:
    def __init__(self, ttl=300):
        self.ttl = ttl
        self.store = {}

    def set(self, key, value):
        self.store[key] = (value, time.time())

    def get(self, key):
        if key in self.store:
            value, timestamp = self.store[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.store[key]
        return None

    def clear(self):
        self.store.clear()
