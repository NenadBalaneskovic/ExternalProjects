"""
__init__.py
Flask application factory for Weather Aggregator.
"""

from flask import Flask
import os

def create_app():
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static")
    )
    app.config["SECRET_KEY"] = "supersecretkey"  # replace with env var in production

    # Register blueprints
    from .routes import bp as routes_bp
    app.register_blueprint(routes_bp)

    return app
