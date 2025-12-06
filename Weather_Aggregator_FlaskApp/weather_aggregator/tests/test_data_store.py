"""
test_data_store.py
Unit tests for SQLite data store.
"""

import os
import tempfile
from data.data_store import DataStore

def test_save_and_get_forecast():
    db_fd, db_path = tempfile.mkstemp()
    store = DataStore(db_path)

    store.save_user_input("12345", "Frankfurt", "Germany")
    store.save_forecast("Frankfurt", "Germany", "SARIMAX", [22.5, 23.0], 21.0, 24.0)

    forecasts = store.get_forecasts("Frankfurt", "Germany")
    assert "SARIMAX" in forecasts
    assert isinstance(forecasts["SARIMAX"], list)

    os.close(db_fd)
    os.remove(db_path)
