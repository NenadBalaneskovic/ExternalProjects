# mlflow_local_runner/setup.py
from setuptools import setup, find_packages
from pathlib import Path

# Projektbeschreibung aus README.md einlesen
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="mlflow_local_runner",
    version="0.1.0",
    description="Private, tokenfreie GUI zum lokalen Ausführen von MLflow-Experimentskripten.",
    long_description=long_description,
    long_description_content_type="text/markdown",

    author="Nenad",
    python_requires=">=3.9",

    # src/ Layout
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    include_package_data=True,

    install_requires=[
        "PySide6==6.7.0",
        "mlflow==3.1.1",
        "pandas==2.3.3",
        "numpy==2.4.1",
        "scikit-learn==1.7.0",
        "xgboost==2.1.0",
        "lightgbm==4.3.0",
        "catboost==1.2.5",
        "matplotlib==3.10.3",
        "plotly==5.22.0",
        "requests==2.32.3",
        "python-dotenv==1.0.1",
        "rich==13.7.1",
    ],

    entry_points={
        "console_scripts": [
            # Terminal-Befehl: mlflow-local-runner
            # Startet src/main.py → main()
            "mlflow-local-runner=mlflow_local_runner.main:main",
        ],
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: User Interfaces",
    ],
)