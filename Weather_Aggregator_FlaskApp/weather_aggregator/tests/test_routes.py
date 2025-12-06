"""
test_routes.py
Unit tests for Flask routes.
"""

import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Enter Location" in response.data

def test_forecast_route(client):
    response = client.get("/forecast/Frankfurt/Germany")
    assert response.status_code == 200
    assert b"Forecast for Frankfurt, Germany" in response.data

def test_report_route(client):
    response = client.get("/report/Frankfurt/Germany")
    # Redirect back to forecast after report generation
    assert response.status_code in (302, 200)
