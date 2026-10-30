# agent/tools.py
from pipeline.hybrid_forecast import run_forecasting_pipeline


def forecast_asset(asset_name: str) -> str:
    """Run hybrid forecast for a given asset and save results."""
    try:
        run_forecasting_pipeline("assets/synthetic_stock_prices.csv",
                                 target_asset=asset_name)
        return (f"✅ Forecast for {asset_name} completed and saved "
                f"to 'forecast_results/'")

    except Exception as e:
        return f"❌ Error: {str(e)}"
