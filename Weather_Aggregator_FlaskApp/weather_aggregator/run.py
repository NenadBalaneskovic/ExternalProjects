"""
run.py
Entry point for Weather Aggregator Flask app.
"""

from flask import Flask
from app.routes import bp as routes_bp

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "supersecretkey"  # replace with env var in production
    app.register_blueprint(routes_bp)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
