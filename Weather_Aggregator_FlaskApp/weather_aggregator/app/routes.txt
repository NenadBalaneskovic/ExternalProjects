"""
routes.py
Flask routes for Weather Aggregator.
Handles input form, dashboard rendering, and API endpoints.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash
from app.forms import LocationForm
from data.weather_api import fetch_weather
from data.data_store import DataStore
from data.pdf_report import PDFReport
from models import base_models, meta_learner
from governance.logger import get_logger
import numpy as np

bp = Blueprint("routes", __name__, template_folder="templates")
logger = get_logger(__name__)
store = DataStore("weather_data.db")


@bp.route("/", methods=["GET", "POST"])
def index():
    """Landing page with input form and weather card."""
    form = LocationForm()
    if form.validate_on_submit():
        po_box = form.po_box.data
        city = form.city.data
        country = form.country.data

        # Save user input
        store.save_user_input(po_box, city, country)
        logger.info(f"User input saved: {po_box}, {city}, {country}")

        return redirect(url_for("routes.forecast", city=city, country=country))
    return render_template("index.html", form=form)


@bp.route("/forecast/<city>/<country>", methods=["GET"])
def forecast(city, country):
    """Forecast page that aggregates model outputs and meta learner ensemble."""

    # Fetch weather data
    weather_data = fetch_weather(city, country)
    if not weather_data:
        flash("Weather data unavailable. Please check your API key or quota.")
        return redirect(url_for("routes.index"))

    # Run base models (each handles extraction internally)
    try:
        sarimax_pred = base_models.sarimax_forecast(weather_data, steps=1)
        kalman_pred = base_models.kalman_filter_forecast(weather_data, steps=1)
        rf_pred = base_models.random_forest_forecast(weather_data, steps=1)
        gb_pred = base_models.gradient_boosting_forecast(weather_data, steps=1)
        cnn_pred = base_models.cnn_forecast(weather_data, steps=1)
        lstm_pred = base_models.lstm_forecast(weather_data, steps=1)
        auto_pred = base_models.autoencoder_forecast(weather_data, steps=1)
    except Exception as e:
        logger.exception(f"Base model forecasting failed: {e}")
        flash("Forecasting failed. Please try again later.")
        return redirect(url_for("routes.index"))

    # Collect predictions into X (one row with all model outputs)
    X = [[
        float(sarimax_pred[0]) if len(sarimax_pred) > 0 else None,
        float(kalman_pred[0]) if len(kalman_pred) > 0 else None,
        float(rf_pred[0]) if len(rf_pred) > 0 else None,
        float(gb_pred[0]) if len(gb_pred) > 0 else None,
        float(cnn_pred[0]) if len(cnn_pred) > 0 else None,
        float(lstm_pred[0]) if len(lstm_pred) > 0 else None,
        float(auto_pred[0]) if len(auto_pred) > 0 else None,
    ]]

    # Construct y as row averages (same length as X)
    y = [np.mean([val for val in X[0] if val is not None])]

    # Fit meta learner
    meta = meta_learner.MetaLearner(method="ridge")
    meta.fit(X, y)

    # Predict ensemble forecast
    ensemble_pred = meta.predict(X)
    forecast_val = float(ensemble_pred[0])

    # Simple confidence bands (placeholder logic)
    lower_band = forecast_val * 0.95
    upper_band = forecast_val * 1.05

    # Save results in datastore
    store.save_forecast(city, country, forecast_val, lower_band, upper_band)
    logger.info(f"Forecast saved for {city}, {country}: {forecast_val} "
                f"(bands: {lower_band}-{upper_band})")

    # Generate PDF report
    report = PDFReport(city, country, ensemble_pred)
    report_path = report.generate()

    return render_template(
        "forecast.html",
        city=city,
        country=country,
        forecasts={
            "SARIMAX": sarimax_pred[0] if len(sarimax_pred) > 0 else None,
            "Kalman": kalman_pred[0] if len(kalman_pred) > 0 else None,
            "RandomForest": rf_pred[0] if len(rf_pred) > 0 else None,
            "GradientBoosting": gb_pred[0] if len(gb_pred) > 0 else None,
            "CNN": cnn_pred[0] if len(cnn_pred) > 0 else None,
            "LSTM": lstm_pred[0] if len(lstm_pred) > 0 else None,
            "Autoencoder": auto_pred[0] if len(auto_pred) > 0 else None,
            "Ensemble": forecast_val,
        },
        report_path=report_path,
    )

@bp.route("/report/<city>/<country>", methods=["GET"])
def report(city, country):
    """Generate a confidence band report for saved forecasts."""

    # Retrieve forecasts from the datastore
    forecasts = store.get_forecasts(city, country)
    if not forecasts:
        flash("No forecasts available for this location.")
        return redirect(url_for("routes.index"))

    # Build confidence bands dictionary keyed by timestamp
    confidence_bands = {
        row[3]: (float(row[1]), float(row[2]))   # timestamp : (lower_band, upper_band)
        for row in forecasts
        if row[1] is not None and row[2] is not None
    }

    # Collect forecast values for summary statistics (ensure numeric)
    forecast_values = []
    for row in forecasts:
        try:
            forecast_values.append(float(row[0]))
        except (TypeError, ValueError):
            logger.warning(f"Skipping non-numeric forecast value: {row[0]}")

    if not forecast_values:
        flash("No numeric forecasts available for this location.")
        return redirect(url_for("routes.index"))

    # Compute simple stats
    avg_forecast = float(np.mean(forecast_values))
    min_forecast = float(np.min(forecast_values))
    max_forecast = float(np.max(forecast_values))

    logger.info(
        f"Report generated for {city}, {country}: "
        f"avg={avg_forecast}, min={min_forecast}, max={max_forecast}"
    )

    return render_template(
        "report.html",
        city=city,
        country=country,
        forecasts=forecasts,
        confidence_bands=confidence_bands,
        avg_forecast=avg_forecast,
        min_forecast=min_forecast,
        max_forecast=max_forecast,
    )
