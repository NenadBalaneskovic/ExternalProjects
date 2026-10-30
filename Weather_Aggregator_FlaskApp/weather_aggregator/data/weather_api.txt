"""
weather_api.py
Integration with OpenWeatherMap (or similar) API.
Fetches and normalizes weather data for forecasting.
"""

import requests
from governance.config import Config
from governance.logger import get_logger

logger = get_logger(__name__)

def fetch_weather(city, country):
    """Fetch current weather and normalized time series."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country}&appid={Config.WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        logger.info(f"Weather data fetched for {city}, {country}")
        return {
            "temperature": data["main"]["temp"],
            "condition": data["weather"][0]["description"],
            "temperature_series": [data["main"]["temp"] - i for i in range(5)]  # stub series
        }
    else:
        logger.error(f"Error fetching weather data: {response.text}")
        return {"temperature_series": []}
