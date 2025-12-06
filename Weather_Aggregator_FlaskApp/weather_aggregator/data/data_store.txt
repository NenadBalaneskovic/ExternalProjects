"""
data_store.py
Handles persistence of weather forecasts and user inputs into a SQLite database.
"""

import sqlite3
import logging

logger = logging.getLogger(__name__)


class DataStore:
    def __init__(self, db_path="weather_data.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema if not exists."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Forecasts table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city TEXT NOT NULL,
                country TEXT NOT NULL,
                forecast REAL NOT NULL,
                lower_band REAL NOT NULL,
                upper_band REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # User inputs table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_inputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                po_box TEXT,
                city TEXT NOT NULL,
                country TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.commit()
        conn.close()

    def save_forecast(self, city, country, forecast, lower_band, upper_band):
        """
        Save forecast with confidence bands into the database.

        Parameters
        ----------
        city : str
        country : str
        forecast : float
        lower_band : float
        upper_band : float
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO forecasts (city, country, forecast, lower_band, upper_band)
            VALUES (?, ?, ?, ?, ?)
            """,
            (city, country, forecast, lower_band, upper_band),
        )
        conn.commit()
        conn.close()
        logger.info(
            f"Forecast saved: {city}, {country} -> {forecast} "
            f"(bands: {lower_band}-{upper_band})"
        )

    def get_forecasts(self, city, country):
        """Retrieve forecasts for a given city/country."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT forecast, lower_band, upper_band, timestamp
            FROM forecasts
            WHERE city=? AND country=?
            ORDER BY timestamp DESC
            """,
            (city, country),
        )
        rows = cur.fetchall()
        conn.close()
        return rows

    def save_user_input(self, po_box, city, country):
        """
        Save user input (PO Box, city, country) into the database.

        Parameters
        ----------
        po_box : str
        city : str
        country : str
        """
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_inputs (po_box, city, country)
            VALUES (?, ?, ?)
            """,
            (po_box, city, country),
        )
        conn.commit()
        conn.close()
        logger.info(f"User input saved: PO Box={po_box}, City={city}, Country={country}")

    def get_user_inputs(self):
        """Retrieve all saved user inputs."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT po_box, city, country, timestamp
            FROM user_inputs
            ORDER BY timestamp DESC
            """
        )
        rows = cur.fetchall()
        conn.close()
        return rows
