# 1. 🚀 Project Introduction: Weather Aggregator Flask App (Surrogate Stacking of LLM/LRM models)

## Objective  
The *Weather Aggregator Flask App* is conceived as a modular demonstration platform that integrates modern machine learning methodologies with practical 
data aggregation in the domain of weather forecasting. At its core, the project seeks to provide a transparent, auditable environment in which different 
predictive models can be compared, combined, and evaluated. By leveraging Flask as the lightweight web framework, the application offers an accessible 
interface for both technical and non‑technical stakeholders, enabling forecasts to be generated, visualized, and stored with ease. This design choice reflects 
a broader ambition: to bridge the gap between advanced ensemble learning research and executive‑ready reporting tools.

The central aim of the project is to test and characterize *stacked surrogate models*—meta‑learners that combine the outputs of diverse base predictors. 
In practice, this involves running multiple forecasting algorithms, such as SARIMAX, Kalman filters, Random Forests, Gradient Boosting, CNNs, LSTMs, and Autoencoders, 
on the same weather dataset. Each of these models captures different statistical or structural aspects of the data. The surrogate model, typically implemented as a 
ridge regression or similar linear meta‑learner, then integrates these heterogeneous predictions into a single ensemble forecast. This stacked approach allows the system 
to exploit complementary strengths while mitigating individual weaknesses, producing results that are more robust and generalizable.

A distinctive feature of the project is its emphasis on *explainability and governance*. Rather than treating ensemble learning as a black box, the app is designed to 
characterize the results of different learning strategies—both linear regression models (LRM) and more complex large language models (LLM) adapted for surrogate tasks. 
By storing forecasts alongside confidence bands, user inputs, and metadata, the system creates a traceable record that can be audited and analyzed. This ensures that the 
outputs are not only accurate but also narratable: stakeholders can understand how predictions were formed, which base models contributed most, and how uncertainty was quantified.

The use of stacked surrogate models in this context serves two purposes. First, it provides a rigorous testbed for evaluating ensemble learning techniques under real-world conditions, 
where data may be noisy, incomplete, or non-stationary. Second, it demonstrates how surrogate modeling can be operationalized within a consulting-ready application. By embedding these 
techniques in a Flask app with a clear user interface, the project shows how advanced machine learning can be packaged into tools that are both technically sound and organizationally impactful.

Beyond its immediate technical objectives, the *Weather Aggregator Flask App* also functions as a narrative asset. It illustrates how modular architectures, reproducible workflows, 
and governance‑compliant reporting can be combined to deliver value in enterprise settings. The project highlights the importance of transparency in ensemble learning, the role of 
surrogate models in characterizing complex outputs, and the potential of lightweight web applications to democratize access to advanced analytics. In doing so, it positions itself 
not merely as a forecasting tool, but as a demonstrator of best practices in explainable AI, ensemble modeling, and consulting‑oriented system design.
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/Weather_Aggregator_FlaskApp/Weather_Aggregator_FlaskApp.md#8--references) 1 - 3 below). 

## 1.1 🎯 **Primary Aim**

The motivation behind the *Weather Aggregator Flask App* arises from the growing need to evaluate and operationalize ensemble learning strategies in domains where predictive accuracy, 
transparency, and reproducibility are paramount. Weather forecasting provides an ideal testbed: it is inherently complex, data-rich, and subject to uncertainty. Traditional single-model 
approaches often struggle to capture the full variability of meteorological data, while ensemble methods promise improved robustness by combining diverse perspectives. Yet, ensemble 
learning itself introduces challenges of interpretability, governance, and deployment. This project is motivated by the desire to confront those challenges directly, creating a demonstrator 
that not only aggregates forecasts but also characterizes how stacked surrogate models perform in practice.

A central motivation is the pursuit of *explainability*. In both academic research and consulting environments, stakeholders increasingly demand that machine learning outputs be narratable and 
auditable. Black‑box ensembles may deliver strong predictive performance, but without clear insight into how results are formed, they risk eroding trust. By designing a system that explicitly 
logs base model outputs, surrogate model predictions, and confidence intervals, the *Weather Aggregator Flask App* ensures that every forecast can be traced back to its components. This transparency 
supports governance requirements and provides a foundation for executive‑ready reporting.

Another motivation is *methodological experimentation*. The project aims to test stacked surrogate models that integrate both linear regression models (LRM) and large language models (LLM) adapted 
for surrogate tasks. This dual focus reflects the evolving landscape of machine learning: linear models remain valuable for their simplicity and interpretability, while LLMs offer powerful 
representational capacity. By comparing and combining these approaches, the app provides insights into how different surrogate strategies characterize ensemble outputs. This experimentation is not 
purely academic; it is designed to inform consulting practice by identifying which methods yield the most reliable and narratable results under real-world conditions.

The objectives of the project can be grouped into four categories. First, **technical objectives**: to implement a modular architecture that supports multiple base models, a meta-learner, 
and a datastore for forecasts and user inputs. Second, **analytical objectives**: to evaluate the performance of stacked surrogate models, quantify uncertainty through confidence bands, and 
generate summary statistics that characterize ensemble behavior. Third, **reporting objectives**: to produce executive‑ready artifacts, including web pages and PDF reports, that present forecasts in 
a clear, auditable format. Fourth, **strategic objectives**: to demonstrate how explainability-first ensemble modeling can be embedded in lightweight applications, thereby bridging the gap between 
research prototypes and consulting deliverables.

Together, these motivations and objectives position the *Weather Aggregator Flask App* as more than a technical exercise. It is a narrative tool that illustrates best practices in ensemble learning, 
surrogate modeling, and explainable AI. By focusing on transparency, reproducibility, and governance, the project aligns with the needs of organizations that must balance innovation with accountability. 
Ultimately, the app serves as both a testbed for methodological exploration and a demonstrator of how advanced machine learning can be packaged into consulting-ready solutions that inspire confidence and drive impact.

## 1.2 🧩 Modular Components and Their Roles

### GUI Architecture Overview

The graphical user interface (GUI) of the Weather Aggregator Flask App is designed to be clean, modular, and consulting-ready, enabling users to interact with ensemble forecasts 
and surrogate model diagnostics through a streamlined web experience. Built with Flask, WTForms, and Jinja2 templating, the interface supports both input-driven workflows and traceable 
reporting, aligning with the app’s broader goals of explainability, governance, and operational clarity.

#### Layout and Navigation

The application follows a three-tier navigation structure:
- **Home (`index.html`)**: Entry point with a location input form (PO Box, City, Country).
- **Forecast (`forecast.html`)**: Displays base model outputs, ensemble prediction, and a link to the report.
- **Report (`report.html`)**: Presents historical forecasts, confidence bands, and summary statistics.

Each page inherits from a shared `base.html` template, ensuring consistent styling, typography, and navigation across the app. The navigation bar provides intuitive access to all major 
views, reinforcing modularity and user orientation.

#### Input Form and Forecast Trigger

The landing page features a form built with WTForms (`LocationForm`) that captures three fields:
- PO Box (optional)
- City (required)
- Country (required)

Upon submission, the form triggers the `/forecast/<city>/<country>` route, which orchestrates the model pipeline: fetching weather data, running base forecasts, fitting the meta learner, and storing results. 
This flow is fully traceable, with user inputs logged via `DataStore.save_user_input()` and forecasts persisted with confidence bands.

#### Forecast Display

The `forecast.html` template renders:
- A **list of base model forecasts** (SARIMAX, Kalman, Random Forest, Gradient Boosting, CNN, LSTM, Autoencoder).
- A **highlighted ensemble forecast**, computed via a stacked surrogate model (ridge regression).
- A **report link** that navigates to the `/report/<city>/<country>` view.

The ensemble forecast is visually emphasized to reflect its role as the aggregated output, while base models are listed for transparency and diagnostic comparison.

#### Report View and Traceability

The `report.html` template provides a governance-ready summary:
- **Summary statistics**: average, minimum, and maximum forecast values.
- **Forecast history table**: includes forecast values, confidence bands, and timestamps.
- **Confidence band table**: keyed by timestamp, showing lower and upper bounds.

This view supports auditability and executive reporting, with all data retrieved from the SQLite-backed `DataStore`. The layout is optimized for readability and export, and can be 
extended with visualizations or PDF generation.

#### Styling and Interactivity

The frontend is styled via `style.css`, which applies:
- Responsive layout containers
- Button and form styling
- Card-based sections for forecasts and reports

Basic interactivity is handled by `scripts.js`, including flash message auto-hide and forecast section toggling. These enhancements improve usability without introducing complexity.

### Model Pipeline Overview

The Weather Aggregator Flask App implements a modular forecasting pipeline that integrates classical statistical models, machine learning regressors, and deep learning stubs into 
a unified ensemble framework. This pipeline is designed to test stacked surrogate models, characterize ensemble outputs, and provide governance‑ready reporting.

#### Data Ingestion

- **Weather API Integration**: The app fetches current weather data from OpenWeatherMap via `weather_api.py`.  
- **Normalization**: The API response is transformed into a temperature series, ensuring compatibility across models.  
- **Caching and Persistence**: Results are cached (`cache.py`) for efficiency and stored in SQLite (`data_store.py`) for traceability.

#### Base Models

The system runs multiple base forecasters in parallel, each capturing different statistical or structural aspects of the data:

- **Classical Models**: SARIMAX (`statsmodels`) and a Kalman filter proxy.  
- **Tree‑Based Models**: Random Forest and Gradient Boosting regressors (`scikit‑learn`).  
- **Neural Models (stubs)**: CNN, LSTM, and Autoencoder architectures (`tensorflow/keras`).  

Each base model is implemented in `base_models.py` with guard clauses to handle sparse or noisy data, returning fallback averages when necessary.

#### Surrogate Meta Learner

- **MetaLearner Class** (`meta_learner.py`): Aggregates base model outputs using configurable methods:
  - Ridge regression (default)
  - Logistic regression
  - Gradient boosting  
- **Stacked Surrogate Approach**: The meta learner fits on base predictions and their averages, producing an ensemble forecast that balances complementary strengths.  
- **Fallback Logic**: If fitting fails, the meta learner defaults to row averages, ensuring robustness.

#### Explainability Layer

- **SHAP and LIME** (`explainability.py`): Provide local and global interpretability of ensemble outputs.  
- **Traceability Module** (`traceability.py`): Registers model parameters, versions, and timestamps, creating an auditable registry of runs.  
- **Governance Logging** (`logger.py`): Captures all model calls, errors, and metadata into centralized logs.

#### Forecast Persistence and Reporting

- **DataStore** (`data_store.py`): Saves forecasts with confidence bands and user inputs.  
- **Report Generation**:
  - **HTML Reports** (`report.html`): Display historical forecasts, confidence bands, and summary statistics.  
  - **PDF Reports** (`pdf_report.py`): Generate governance‑ready documents using ReportLab.  

#### User Workflow

1. **Input**: User enters PO Box, City, and Country via `LocationForm`.  
2. **Forecast**: Base models run, meta learner aggregates, and ensemble forecast is saved.  
3. **Report**: User navigates to the report view, which retrieves stored forecasts, computes summary stats, and displays confidence bands.  
4. **Traceability**: All steps are logged, versioned, and auditable.

### System Governance & Explainability Overview

The Weather Aggregator Flask App is designed not only to deliver accurate forecasts but also to ensure that every output is transparent, auditable, and narratable. 
Governance and explainability are embedded into the system architecture, reflecting best practices in consulting‑ready analytics and explainable AI.

#### Centralized Logging

- **Logger Module (`logger.py`)**: All application events—user inputs, model runs, errors, and forecast persistence—are captured in a centralized log file.  
- **Structured Records**: Each log entry includes timestamps, module names, and severity levels, ensuring traceability across the workflow.  
- **Governance Compliance**: Logging provides an audit trail that supports reproducibility and accountability in client‑facing environments.

#### Traceability and Model Versioning

- **Traceability Module (`traceability.py`)**: Registers each model run with metadata including parameters, version hashes, and timestamps.  
- **Version Hashing**: Parameters are hashed to create lightweight version identifiers, enabling reproducible experiments and clear differentiation between model configurations.  
- **Registry Access**: Users can list all registered models and retrieve metadata, ensuring that every forecast can be tied back to its originating configuration.

#### Explainability Layer

- **SHAP Analysis**: Provides global and local feature importance for ensemble forecasts, showing how base model outputs contribute to the surrogate learner’s predictions.  
- **LIME Analysis**: Offers instance‑level explanations, highlighting which features most influenced a specific forecast.  
- **Integration with Meta Learner**: Both SHAP and LIME can be applied to the stacked surrogate model, making ensemble outputs narratable and actionable.

#### Reporting and Transparency

- **HTML Reports (`report.html`)**: Present forecasts, confidence bands, and summary statistics in a clear, tabular format.  
- **PDF Reports (`pdf_report.py`)**: Generate governance‑ready documents using ReportLab, suitable for executive distribution.  
- **Confidence Bands**: Each forecast is accompanied by lower and upper bounds, quantifying uncertainty and supporting risk‑aware decision‑making.

#### Governance Principles

The system embodies three core governance principles:
1. **Transparency**: Every forecast is traceable to its inputs, base models, and surrogate learner.  
2. **Reproducibility**: Model parameters, version hashes, and logs ensure that results can be replicated and validated.  
3. **Accountability**: Reports and logs provide a narrative record that supports consulting engagements and organizational oversight.

 
## 1.3 🧠 **GUI sketch**  

In the following we address our full GUI sketch, a clean, structured layout for our Flask-based Weather Aggregator App. It includes all the key modules we discussed: 
objective and constraint input, method selection forecasting models), result display, visualization, and diagnostics.

![TWeatherAggregator_sketch.png](https://github.com/NenadBalaneskovic/ExternalProjects/blob/74e99d22089f0f2e2600a39aa126b64e62a6641e/Weather_Aggregator_FlaskApp/GUI_sketch_complete.png)

Here’s a walkthrough of the **Weather Aggregator architecture sketch** — designed to be modular, narratable, and consulting-ready:

### 🧭 Architecture Walkthrough

#### 1. **User Frontend**
- **Inputs**: PO Box, City, Country — simple form fields.
- **Outputs**: Weather card with current conditions and a 5-day forecast.
- **Visuals**: Icons, temperature ranges, and confidence bands.

#### 2. **Flask Backend**
- Orchestrates the entire pipeline.
- Routes user input to API calls and model inference.
- Handles dynamic model selection and report generation.

#### 3. **Weather API Integration**
- Pulls raw weather data (temperature, humidity, wind, etc.).
- Caches responses for reproducibility and auditability.

#### 4. **Model Ensemble Layer**
- **Base Models**:
  - SARIMAX, Kalman filters, decision trees.
  - CNNs for spatial patterns, LSTMs for temporal sequences.
  - Autoencoders for anomaly detection.
  - Physics-inspired models for critical transitions.
- **Meta Learner**:
  - Aggregates predictions.
  - Computes confidence bands.
  - Learns from model disagreement.

#### 5. **Hosted Model Integration**
- Uses Hugging Face Inference API.
- Avoids local downloads — keeps CPU/memory lean.
- Enables plug-and-play model experimentation.

#### 6. **Data Storage**
- Stores:
  - Raw API data
  - Model outputs
  - Confidence intervals
  - PDF reports for governance

#### 7. **Governance Layer**
- Tracks:
  - Model versions
  - Parameters
  - Timestamps
- Ensures:
  - Reproducibility
  - Audit trails
  - Executive-ready reporting

#### 8. **User Dashboard**
- Presents:
  - Forecasts with uncertainty bands
  - Model explainability (e.g., SHAP values)
  - Downloadable reports

This architecture is designed to showcase **transparency, modularity, and operational reliability** — ideal for consulting demos and technical interviews.  

**Flask runs locally. We can connect cost-free to Hugging Face models via their Serverless Inference API, which offers a free tier. For ensemble modeling, 
~30 models is a strong baseline, and meta-learners like logistic regression, ridge regression, or gradient boosting are ideal depending on our interpretability vs. 
performance tradeoff. Calculation time depends on model complexity and API latency, but with hosted inference, you can parallelize requests to keep response times within seconds.**

### 🔌 Cost-Free Hugging Face Integration
- **Serverless Inference API**: Hugging Face offers free-tier access to thousands of models via hosted endpoints — no local downloads required.
- **How it works**:
  - You select models from the Hugging Face Hub that support inference.
  - Authenticate using an API token.
  - Send requests via HTTP POST with input payloads.
- **Best practices**:
  - Use lightweight models for faster response.
  - Batch or parallelize requests to reduce latency.
  - Monitor usage to stay within free-tier limits.

### 🧠 Ensemble Size & Meta-Learner Strategy

#### ✅ Ensemble Size
- **30 models** is a solid minimum for robust meta-learning.
- Consider diversity: mix autoregressive LLMs, transformers, time-series regressors, and physics-inspired models.
- Use model confidence bands as features for the meta learner.

#### 🔁 Meta-Learner Options
| Meta Learner        | Pros                                  | Cons                                  | Use Case                          |
|---------------------|----------------------------------------|---------------------------------------|-----------------------------------|
| **Logistic Regression** | Interpretable, fast                   | Limited expressiveness                | Binary classification (e.g., rain/no rain) |
| **Linear Regression**   | Simple, interpretable                | Poor with nonlinearities              | Aggregating temperature forecasts |
| **Ridge/Lasso**         | Regularized, avoids overfitting      | Still linear                          | High-dimensional ensemble inputs  |
| **Gradient Boosting**   | High accuracy, handles interactions  | Less interpretable                    | Complex ensemble fusion           |
| **Stacked Neural Net**  | Flexible, nonlinear                  | Requires tuning, slower               | Deep ensemble fusion              |

**Recommendation**: Start with **ridge regression** for interpretability and robustness, then experiment with **gradient boosting** or **shallow neural nets** for performance.

### ⏱️ Calculation Time Estimates
- **Local Flask orchestration**: negligible overhead.
- **Weather API call**: ~0.5–1.5 seconds.
- **Hosted model inference**:
  - Lightweight models: ~0.5–2 seconds per call.
  - Heavy models (e.g., LLaMA, GPT-style): ~3–10 seconds.
- **Meta learner aggregation**: ~0.5 seconds locally.
- **Total latency** (parallelized): **~3–6 seconds** for full ensemble + forecast.

### 🧩 Optimization Tips
- Use **async Flask routes** to parallelize model calls.
- Cache frequent locations and model outputs.
- Precompute ensemble weights for common cities.
- Use **confidence-weighted voting** or **Bayesian averaging** to reduce runtime.

  
Latency of about 6 seconds is fine. Now we shall take all of the above discussion details into account and start with the implementation. We should first stub out the flask website 
structure on localhost via Python. I would prefer if we use LiteSQL since it is lightweight compared to postgresql. We should consider all mentioned optimization aspects when designing 
the website and also at later stages when addressing the model functionality. But let us start with the flask localhost website. I intend to run the entire code in a jupyter notebook. 
Only at the end of the project will we generate a modularized folder. We first sketch a detailed modularized folder structure with all necessary py-file modules, requirements.txt and 
explain their intended role in our project before we start with the implementation (stubbing out of the flask localhost website).

We are ready to scaffold this into actual Pythonic code and wire up the backend logic for method selection and parser/detector routing.

---

# 2. 🔐 Algorithmic concepts

This is a comprehensive, notebook-ready Markdown explanation of major Pythonic modules
our Flask app should support. Each section includes mathematical foundations, algorithmic steps, and illustrative examples.

## 2.1 📁 Project Structure

Here's a modularized folder structure tailored for our **Weather Aggregator** project, optimized for local Flask development in a Jupyter notebook and later transition to a production-ready consulting demo. 
It reflects all the architectural layers we've discussed — from frontend to ensemble modeling, Hugging Face integration, governance, and PDF reporting.

### 📁 Modularized Folder Structure Overview

```plaintext
weather_aggregator/
│
├── app/                          # Core Flask application
│   ├── __init__.py              # App factory and configuration
│   ├── routes.py                # Flask routes (input form, dashboard, API calls)
│   ├── forms.py                 # WTForms for PO Box, City, Country
│   ├── templates/               # HTML templates (Jinja2)
│   │   ├── base.html            # Layout scaffold
│   │   ├── index.html           # Input form and weather card
│   │   ├── forecast.html        # Forecast display with charts
│   └── static/                  # CSS, JS, icons
│       ├── style.css
│       └── scripts.js
│
├── models/                      # Ensemble modeling layer
│   ├── __init__.py
│   ├── base_models.py          # SARIMAX, Kalman, Trees, CNNs, LSTMs, Autoencoders
│   ├── huggingface_models.py   # Hosted model inference via API
│   ├── meta_learner.py         # Ridge, logistic, boosting, etc.
│   └── explainability.py       # SHAP, LIME, confidence bands
│
├── data/                        # Data ingestion and storage
│   ├── weather_api.py          # OpenWeatherMap or similar API integration
│   ├── data_store.py           # LiteSQL interface (SQLite wrapper)
│   ├── cache.py                # Caching layer for API responses
│   └── pdf_report.py           # PDF generation (ReportLab or LaTeX)
│
├── governance/                  # Auditability and reproducibility
│   ├── logger.py               # Logging model runs, parameters, timestamps
│   ├── traceability.py         # Model versioning and metadata
│   └── config.py               # Central config (API keys, model registry)
│
├── notebooks/                   # Jupyter notebooks for development
│   ├── 01_flask_stub.ipynb     # Initial Flask stub and input form
│   ├── 02_model_dev.ipynb      # Ensemble model prototyping
│   ├── 03_huggingface_api.ipynb# Hosted model integration
│   └── 04_dashboard.ipynb      # Forecast visualization and reporting
│
├── tests/                       # Unit tests
│   ├── test_routes.py
│   ├── test_models.py
│   ├── test_data_store.py
│   └── test_meta_learner.py
│
├── requirements.txt             # Dependencies
├── README.md                    # Project overview and usage
└── run.py                       # Entry point for Flask app
```

### 🧩 Module Roles Explained

#### 🔹 `app/`
- Central Flask logic and routing.
- HTML templates for input and output.
- Static assets for styling and interactivity.

#### 🔹 `models/`
- Implements all base models (SARIMAX, CNN, etc.).
- Wraps Hugging Face inference calls.
- Aggregates predictions via meta learners.
- Adds explainability and confidence band logic.

#### 🔹 `data/`
- Handles weather API ingestion.
- Stores and retrieves data via LiteSQL.
- Caches frequent queries.
- Generates PDF reports for governance.

#### 🔹 `governance/`
- Tracks model usage, parameters, and timestamps.
- Ensures reproducibility and auditability.
- Centralizes configuration and secrets.

#### 🔹 `notebooks/`
- Development workspace for iterative prototyping.
- Will be modularized into `.py` files later.

#### 🔹 `tests/`
- Ensures robustness and correctness of each module.
- Validates ensemble logic and data integrity.

This structure is designed to support:
- **Modular development** in Jupyter first.
- **Scalable deployment** later.
- **Governance and auditability** throughout.

Let’s now stub out the Flask website in our notebook — starting with the input form and routing logic.

## 2.2 📄 app folder

### 2.2.1 `run.py`

```python
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
```

### 2.2.2 `forms.py`

```python
"""
forms.py
WTForms definitions for Weather Aggregator.
Handles user input: PO Box, City, Country.
"""

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length

class LocationForm(FlaskForm):
    """Form for user to enter location details."""
    po_box = StringField(
        "PO Box",
        validators=[DataRequired(), Length(min=3, max=10)],
        render_kw={"placeholder": "Enter PO Box"}
    )
    city = StringField(
        "City",
        validators=[DataRequired(), Length(min=2, max=50)],
        render_kw={"placeholder": "Enter City"}
    )
    country = StringField(
        "Country",
        validators=[DataRequired(), Length(min=2, max=50)],
        render_kw={"placeholder": "Enter Country"}
    )
    submit = SubmitField("Get Forecast")
```

### 2.2.3 `routes.py` (with __init__.py)

```python
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

```

### 2.2.4 `__init__.py`

````
python
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
````

## 2.3 📄 js and css files

### 2.3.1 `scripts.js`

```json
// scripts.js
// Basic interactivity for Weather Aggregator frontend

document.addEventListener("DOMContentLoaded", function() {
    console.log("Weather Aggregator frontend loaded ✅");

    // Example: flash message auto-hide
    const flashes = document.querySelectorAll(".flash-message");
    flashes.forEach(msg => {
        setTimeout(() => {
            msg.style.display = "none";
        }, 4000);
    });

    // Example: toggle forecast section
    const toggleBtn = document.getElementById("toggle-forecast");
    if (toggleBtn) {
        toggleBtn.addEventListener("click", () => {
            const section = document.querySelector(".forecast-section");
            if (section.style.display === "none") {
                section.style.display = "block";
                toggleBtn.innerText = "Hide Forecasts";
            } else {
                section.style.display = "none";
                toggleBtn.innerText = "Show Forecasts";
            }
        });
    }
});
```

### 2.3.2 `style.css`

````css
/* style.css
   Basic styling for Weather Aggregator frontend
*/

body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background: #f4f7fa;
    color: #333;
}

header {
    background: #2c3e50;
    color: #fff;
    padding: 1rem;
    text-align: center;
}

header h1 {
    margin: 0;
    font-size: 1.8rem;
}

nav a {
    color: #ecf0f1;
    margin: 0 10px;
    text-decoration: none;
}

main {
    padding: 2rem;
}

form {
    background: #fff;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    max-width: 400px;
    margin: auto;
}

form div {
    margin-bottom: 1rem;
}

form input, form button {
    width: 100%;
    padding: 0.6rem;
    border: 1px solid #ccc;
    border-radius: 4px;
}

form button {
    background: #3498db;
    color: #fff;
    border: none;
    cursor: pointer;
}

form button:hover {
    background: #2980b9;
}

.weather-card, .forecast-section, .ensemble-section, .report-section {
    background: #fff;
    padding: 1.5rem;
    margin: 1rem auto;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    max-width: 600px;
}

.btn {
    display: inline-block;
    padding: 0.6rem 1.2rem;
    background: #27ae60;
    color: #fff;
    text-decoration: none;
    border-radius: 4px;
}

.btn:hover {
    background: #1e8449;
}

footer {
    text-align: center;
    padding: 1rem;
    background: #2c3e50;
    color: #fff;
    margin-top: 2rem;
}
````

## 2.4 📄 html files  

### 2.4.1 `base.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Weather Aggregator</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <script src="{{ url_for('static', filename='scripts.js') }}"></script>
</head>
<body>
    <header>
        <h1>🌦 Weather Aggregator</h1>
        <nav>
            <a href="{{ url_for('routes.index') }}">Home</a>
        </nav>
    </header>

    <main>
        {% block content %}{% endblock %}
    </main>

    <footer>
        <p>&copy; 2025 Weather Aggregator Demo</p>
    </footer>
</body>
</html>

```

### 2.4.2 `forecast.html`

```html
{% extends "base.html" %}

{% block content %}
<h2>Forecast for {{ city }}, {{ country }}</h2>

<div class="forecast-section">
    <h3>Base Model Forecasts</h3>
    <ul>
        {% for model, forecast in forecasts.items() %}
        <li><strong>{{ model }}:</strong> {{ forecast }}</li>
        {% endfor %}
    </ul>
</div>

<div class="ensemble-section">
    <h3>Ensemble Forecast</h3>
    <p>{{ ensemble }}</p>
</div>

<div class="report-section">
    <a href="{{ url_for('routes.report', city=city, country=country) }}" class="btn">Download PDF Report</a>
</div>
{% endblock %}
```

### 2.4.3 `index.html`

```html
{% extends "base.html" %}

{% block content %}
<h2>Enter Location</h2>
<form method="POST" action="{{ url_for('routes.index') }}">
    {{ form.hidden_tag() }}
    <div>
        {{ form.po_box.label }}<br>
        {{ form.po_box(size=20) }}
    </div>
    <div>
        {{ form.city.label }}<br>
        {{ form.city(size=20) }}
    </div>
    <div>
        {{ form.country.label }}<br>
        {{ form.country(size=20) }}
    </div>
    <div>
        {{ form.submit() }}
    </div>
</form>

{% if weather %}
<div class="weather-card">
    <h3>Current Weather</h3>
    <p>Temperature: {{ weather.temperature }} °C</p>
    <p>Condition: {{ weather.condition }}</p>
</div>
{% endif %}
{% endblock %}
```

### 2.4.4 `report.html`

```html
{% extends "base.html" %}

{% block content %}
<h2>Forecast Report for {{ city }}, {{ country }}</h2>

<div class="summary-section">
    <h3>Summary Statistics</h3>
    <ul>
        <li><strong>Average Forecast:</strong> {{ avg_forecast }}</li>
        <li><strong>Minimum Forecast:</strong> {{ min_forecast }}</li>
        <li><strong>Maximum Forecast:</strong> {{ max_forecast }}</li>
    </ul>
</div>

<div class="history-section">
    <h3>Forecast History</h3>
    <table>
        <thead>
            <tr>
                <th>Forecast</th>
                <th>Lower Band</th>
                <th>Upper Band</th>
                <th>Timestamp</th>
            </tr>
        </thead>
        <tbody>
            {% for forecast, lower, upper, timestamp in forecasts %}
            <tr>
                <td>{{ forecast }}</td>
                <td>{{ lower }}</td>
                <td>{{ upper }}</td>
                <td>{{ timestamp }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>

<div class="bands-section">
    <h3>Confidence Bands</h3>
    <table>
        <thead>
            <tr>
                <th>Timestamp</th>
                <th>Lower Band</th>
                <th>Upper Band</th>
            </tr>
        </thead>
        <tbody>
            {% for ts, bands in confidence_bands.items() %}
            <tr>
                <td>{{ ts }}</td>
                <td>{{ bands[0] }}</td>
                <td>{{ bands[1] }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>

<div class="back-section">
    <a href="{{ url_for('routes.forecast', city=city, country=country) }}" class="btn">Back to Forecast</a>
</div>
{% endblock %}
```

## 2.5 📄 data folder 

### 2.5.1 `weather_api.py`

```python
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
```

### 2.5.2 `pdf_report.py`

```python
"""
pdf_report.py
PDF generation for Weather Aggregator.
Uses ReportLab to create governance-ready reports.
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

class PDFReport:
    def __init__(self, city, country, forecast):
        self.city = city
        self.country = country
        self.forecast = forecast

    def generate(self):
        filename = f"{self.city}_{self.country}_forecast.pdf"
        c = canvas.Canvas(filename, pagesize=letter)

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, 720, f"Forecast Report for {self.city}, {self.country}")

        # Forecast content
        c.setFont("Helvetica", 12)
        c.drawString(72, 680, f"Forecast: {self.forecast}")

        # Add more details if needed
        c.drawString(72, 660, "Generated by Weather Aggregator Demo")

        c.showPage()
        c.save()

        return filename
```

### 2.5.3 `data_store.py`

```python
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
```

### 2.5.4 `cache.py`

```python
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
```

## 2.6 📄 governance folder 

### 2.6.1 `config.py`

```python
"""
config.py
Central configuration for Weather Aggregator.
Stores API keys, endpoints, and model registry.
"""

class Config:
    # API Keys
    WEATHER_API_KEY = "Insert your OpenWeatherMap API-Key"
    HF_API_TOKEN = "Insert your HuggingFace API-Key"

    # API URLs
    HF_API_URL = "https://api-inference.huggingface.co/models/"

    # Model Registry (extendable)
    MODEL_REGISTRY = {
        "SARIMAX": {"order": (1,1,1)},
        "RandomForest": {"n_estimators": 50},
        "Kalman": {"window": 3},
        "ridge": {"alpha": 1.0},
        "logistic": {},
        "boosting": {"n_estimators": 100}
    }
```

### 2.6.2 `logger.py`

```python
"""
logger.py
Centralized logging for Weather Aggregator.
Captures model runs, parameters, and timestamps.
"""

import logging
import os

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def get_logger(name: str):
    """Return a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(LOG_DIR, "weather_aggregator.log"))
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

### 2.6.3 `traceability.py`

```python
"""
traceability.py
Model versioning and metadata tracking for Weather Aggregator.
Ensures reproducibility and auditability of forecasts.
"""

import hashlib
import json
from datetime import datetime

class Traceability:
    def __init__(self):
        self.registry = {}

    def register_model(self, model_name, params):
        """Register model with parameters and version hash."""
        version_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
        metadata = {
            "model_name": model_name,
            "params": params,
            "version": version_hash,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.registry[model_name] = metadata
        return metadata

    def get_metadata(self, model_name):
        """Retrieve metadata for a registered model."""
        return self.registry.get(model_name, None)

    def list_registry(self):
        """List all registered models and metadata."""
        return self.registry
```

## 2.7 📄 models folder 

### 2.7.1 `meta_learner.py`

```python
"""
meta_learner.py
Meta learner that aggregates forecasts from base models.
Supports multiple methods (ridge, logistic, boosting).
"""

import logging
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)


class MetaLearner:
    def __init__(self, method="ridge"):
        """
        Initialize the meta learner with a chosen method.

        Parameters
        ----------
        method : str
            Aggregation method. Options: "ridge", "logistic", "boosting".
        """
        if method == "ridge":
            self.model = Ridge()
        elif method == "logistic":
            self.model = LogisticRegression()
        elif method == "boosting":
            self.model = GradientBoostingRegressor()
        else:
            raise ValueError(f"Unknown method: {method}")

        self.method = method
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the meta learner on model outputs.

        Parameters
        ----------
        X : list or np.ndarray
            Shape (n_samples, n_models). Each row = predictions from base models.
        y : list or np.ndarray
            Shape (n_samples,). Target values (e.g., average of base predictions).
        """
        X = np.array(X)
        y = np.array(y)

        logger.info(f"MetaLearner.fit called with X shape={X.shape}, y shape={y.shape}")

        # Guard clause: check consistency
        if X.shape[0] == 0 or y.shape[0] == 0:
            logger.warning("Empty input to MetaLearner.fit. Skipping training.")
            self.is_fitted = False
            return

        if X.shape[0] != y.shape[0]:
            logger.error(
                f"Inconsistent samples: X has {X.shape[0]} rows, y has {y.shape[0]} rows."
            )
            # Fallback: recompute y as row averages
            y = np.array([np.mean(row) for row in X])
            logger.info(f"Recomputed y from X. New y shape={y.shape}")

        try:
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info(f"MetaLearner ({self.method}) successfully fitted.")
        except Exception as e:
            logger.exception(f"MetaLearner.fit failed: {e}")
            self.is_fitted = False

    def predict(self, X):
        """
        Predict ensemble forecast.

        Parameters
        ----------
        X : list or np.ndarray
            Shape (n_samples, n_models). Each row = predictions from base models.

        Returns
        -------
        np.ndarray
            Ensemble predictions.
        """
        X = np.array(X)
        logger.info(f"MetaLearner.predict called with X shape={X.shape}")

        if not self.is_fitted:
            logger.warning("MetaLearner not fitted. Returning row averages as fallback.")
            return np.array([np.mean(row) for row in X])

        try:
            preds = self.model.predict(X)
            logger.info("MetaLearner.predict succeeded.")
            return preds
        except Exception as e:
            logger.exception(f"MetaLearner.predict failed: {e}")
            return np.array([np.mean(row) for row in X])
```

### 2.7.2 `huggingface_models.py`

```python
"""
huggingface_models.py
Hosted model inference via Hugging Face API.
"""

import requests
from governance.config import Config
from governance.logger import get_logger

logger = get_logger(__name__)

def query_hf_model(model_name, payload):
    """Query Hugging Face hosted model."""
    headers = {"Authorization": f"Bearer {Config.HF_API_TOKEN}"}
    response = requests.post(Config.HF_API_URL + model_name, headers=headers, json=payload)
    if response.status_code == 200:
        logger.info(f"Model {model_name} inference successful.")
        return response.json()
    else:
        logger.error(f"Error {response.status_code} from {model_name}: {response.text}")
        return None

def batch_query(models, payload):
    """Query multiple Hugging Face models sequentially."""
    results = {}
    for model in models:
        results[model] = query_hf_model(model, payload)
    return results
```

### 2.7.3 `explainability.py`

```python
"""
explainability.py
Explainability utilities for Weather Aggregator.
Provides SHAP and LIME analysis for ensemble forecasts.
"""

import shap
from lime.lime_tabular import LimeTabularExplainer

class Explainability:
    def __init__(self, training_data, feature_names=None):
        self.training_data = training_data
        self.feature_names = feature_names if feature_names else [f"f{i}" for i in range(training_data.shape[1])]

    def shap_analysis(self, model, sample):
        explainer = shap.Explainer(model, self.training_data)
        shap_values = explainer(sample)
        return shap_values

    def lime_analysis(self, model, sample):
        explainer = LimeTabularExplainer(
            training_data=self.training_data,
            feature_names=self.feature_names,
            verbose=True,
            mode="regression"
        )
        explanation = explainer.explain_instance(sample, model.predict, num_features=5)
        return explanation
```

### 2.7.4 `base_models.py`

```python
"""
base_models.py
Classical + ML forecasting models for Weather Aggregator.
Includes SARIMAX, Kalman filter, Trees, CNNs, LSTMs, Autoencoders.
"""

import logging
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten

logger = logging.getLogger(__name__)


# --- Helper to extract numeric series ---
def _extract_series(weather_data, key="temperature_series"):
    """
    Extract numeric temperature series from weather_data dict.
    Falls back to single temperature if no series is available.
    """
    if weather_data is None:
        return np.array([])

    # If API provides a time series (e.g. hourly temps)
    if key in weather_data and isinstance(weather_data[key], (list, np.ndarray)):
        return np.array(weather_data[key], dtype=float)

    # Fallback: use single temperature value
    if "temperature" in weather_data:
        return np.array([float(weather_data["temperature"])])

    return np.array([])


# --- Classical Models ---

def sarimax_forecast(weather_data, steps=5):
    """SARIMAX forecast."""
    series = _extract_series(weather_data)
    if len(series) < 3:
        logger.warning("Not enough data for SARIMAX. Returning average.")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])

    try:
        model = sm.tsa.SARIMAX(series, order=(1, 1, 1))
        results = model.fit(disp=False)
        forecast = results.forecast(steps=steps)
        return forecast
    except Exception as e:
        logger.exception(f"SARIMAX forecast failed: {e}")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])


def kalman_filter_forecast(weather_data, steps=5):
    """Simple Kalman filter forecast (rolling mean proxy)."""
    series = _extract_series(weather_data)
    if len(series) == 0:
        return np.array([])

    forecast = [np.mean(series[-3:])] * steps
    return np.array(forecast)


# --- Tree-Based Models ---

def random_forest_forecast(weather_data, steps=5):
    """Random Forest regression forecast."""
    series = _extract_series(weather_data)
    if len(series) < 2:
        logger.warning("Not enough data for RandomForest. Returning average.")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])

    try:
        X = np.arange(len(series)).reshape(-1, 1)
        y = series
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X, y)
        future = np.arange(len(series), len(series) + steps).reshape(-1, 1)
        return rf.predict(future)
    except Exception as e:
        logger.exception(f"RandomForest forecast failed: {e}")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])


def gradient_boosting_forecast(weather_data, steps=5):
    """Gradient Boosting regression forecast."""
    series = _extract_series(weather_data)
    if len(series) < 2:
        logger.warning("Not enough data for GradientBoosting. Returning average.")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])

    try:
        X = np.arange(len(series)).reshape(-1, 1)
        y = series
        gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
        gb.fit(X, y)
        future = np.arange(len(series), len(series) + steps).reshape(-1, 1)
        return gb.predict(future)
    except Exception as e:
        logger.exception(f"GradientBoosting forecast failed: {e}")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])


# --- Neural Models (stubs) ---

def lstm_forecast(weather_data, steps=5):
    """LSTM forecast stub."""
    series = _extract_series(weather_data)
    if len(series) < 2:
        logger.warning("Not enough data for LSTM. Returning average.")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])

    try:
        X = np.array(series).reshape(-1, 1, 1)
        model = Sequential([
            LSTM(10, input_shape=(1, 1)),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, series, epochs=5, verbose=0)
        preds = model.predict(np.array(series[-steps:]).reshape(-1, 1, 1))
        return preds.flatten()
    except Exception as e:
        logger.exception(f"LSTM forecast failed: {e}")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])


def cnn_forecast(weather_data, steps=5):
    """CNN forecast stub."""
    series = _extract_series(weather_data)
    if len(series) < 2:
        logger.warning("Not enough data for CNN. Returning average.")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])

    try:
        X = np.array(series).reshape(-1, 1, 1)
        model = Sequential([
            Conv1D(8, kernel_size=1, activation="relu", input_shape=(1, 1)),
            Flatten(),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, series, epochs=5, verbose=0)
        preds = model.predict(np.array(series[-steps:]).reshape(-1, 1, 1))
        return preds.flatten()
    except Exception as e:
        logger.exception(f"CNN forecast failed: {e}")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])


def autoencoder_forecast(weather_data, steps=5):
    """Autoencoder forecast stub (reconstruction-based)."""
    series = _extract_series(weather_data)
    if len(series) < 2:
        logger.warning("Not enough data for Autoencoder. Returning average.")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])

    try:
        X = np.array(series).reshape(-1, 1)
        model = Sequential([
            Dense(5, activation="relu", input_shape=(1,)),
            Dense(1, activation="linear")
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, series, epochs=5, verbose=0)
        preds = model.predict(np.array(series[-steps:]).reshape(-1, 1))
        return preds.flatten()
    except Exception as e:
        logger.exception(f"Autoencoder forecast failed: {e}")
        return np.array([float(np.mean(series))]) if len(series) > 0 else np.array([])
```

### 2.7.5 `__init__.py`

```python
"""
__init__.py
Initialization for Weather Aggregator models package.
Exposes base models, hosted Hugging Face models, meta learner, and explainability utilities.
"""

from .base_models import (
    sarimax_forecast,
    kalman_filter_forecast,
    random_forest_forecast,
    gradient_boosting_forecast,
    lstm_forecast,
    cnn_forecast,
    autoencoder_forecast,
)

from .huggingface_models import query_hf_model, batch_query
from .meta_learner import MetaLearner
from .explainability import Explainability

__all__ = [
    # Base models
    "sarimax_forecast",
    "kalman_filter_forecast",
    "random_forest_forecast",
    "gradient_boosting_forecast",
    "lstm_forecast",
    "cnn_forecast",
    "autoencoder_forecast",
    # Hugging Face integration
    "query_hf_model",
    "batch_query",
    # Meta learner
    "MetaLearner",
    # Explainability
    "Explainability",
]
```

## 2.8 📄 tests folder 

Let’s sketch the **`tests/` suite** so we can validate correctness and robustness of each module before integration. 
This ensures our Weather Aggregator remains reliable, reproducible, and governance-ready.

---

### 2.8.1 📁 `tests/` Folder Structure

```plaintext
tests/
├── test_routes.py
├── test_models.py
├── test_meta_learner.py
├── test_data_store.py
├── test_pdf_report.py
├── test_explainability.py
└── conftest.py
```

### 2.8.2 📄 `test_routes.py`
```python
import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.testing = True
    return app.test_client()

def test_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Weather Aggregator" in response.data
```

### 2.8.3 📄 `test_models.py`
```python
import numpy as np
from models import base_models

def test_sarimax_forecast():
    series = np.random.rand(50)
    forecast = base_models.sarimax_forecast(series, steps=5)
    assert len(forecast) == 5

def test_random_forest_forecast():
    series = np.random.rand(50)
    forecast = base_models.random_forest_forecast(series, steps=5)
    assert len(forecast) == 5
```

### 2.8.4 📄 `test_meta_learner.py`
```python
import numpy as np
from models.meta_learner import MetaLearner

def test_meta_learner_ridge():
    X = np.random.rand(10, 5)
    y = np.random.rand(10)
    meta = MetaLearner(method="ridge")
    meta.fit(X, y)
    preds = meta.predict(X)
    assert len(preds) == 10

def test_confidence_band():
    forecasts = np.random.rand(10, 5)
    meta = MetaLearner(method="ridge")
    lower, upper = meta.confidence_band(forecasts)
    assert len(lower) == len(upper) == 10
```

### 2.8.5 📄 `test_data_store.py`
```python
import os
from data_store import DataStore

def test_user_input_storage(tmp_path):
    db_path = tmp_path / "test.db"
    store = DataStore(db_path)
    store.save_user_input("12345", "Berlin", "Germany")
    results = store.get_forecasts("Berlin", "Germany")
    assert isinstance(results, list)
```

### 2.8.6 📄 `test_pdf_report.py`
```python
import os
from pdf_report import PDFReport

def test_pdf_generation(tmp_path):
    filename = tmp_path / "test_report.pdf"
    forecasts = {"SARIMAX": [22.5, 23.0]}
    confidence_bands = {"SARIMAX": (21.0, 24.0)}
    metadata = {"SARIMAX": "Order=(1,1,1)"}
    report = PDFReport(str(filename))
    generated = report.generate("Berlin", "Germany", forecasts, confidence_bands, metadata)
    assert os.path.exists(generated)
```

### 2.8.7 📄 `test_explainability.py`
```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from models.explainability import Explainability

def test_shap_lime_analysis():
    X = np.random.rand(20, 5)
    y = np.random.rand(20)
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X, y)
    exp = Explainability(X)
    sample = X[0]
    shap_values = exp.shap_analysis(rf, sample)
    assert shap_values is not None
```

### 2.8.8 📄 `conftest.py`
```python
# Shared pytest fixtures can go here
```

### 2.8.9 ✅ What This Test Suite Achieves
- **Routes**: Ensures Flask endpoints respond correctly.
- **Models**: Validates SARIMAX, Random Forest, and other base models.
- **Meta Learner**: Confirms aggregation and confidence band logic.
- **Data Store**: Tests LiteSQL storage and retrieval.
- **PDF Reports**: Verifies governance-ready report generation.
- **Explainability**: Confirms SHAP/LIME integration works.

This suite ensures **robustness, reproducibility, and governance compliance** across our Weather Aggregator system.  


## 2.9 📄 `requirements.txt`

Here’s the stub for the **`requirements.txt` file**, which captures all dependencies for our Weather Aggregator project. 
This ensures reproducibility and makes setup straightforward for consulting demos or deployment.

```plaintext
# Core framework
Flask==3.0.0

# Database
sqlite3-binary==0.0.1  # lightweight SQLite bindings

# Classical time series
statsmodels==0.14.0

# Machine learning
scikit-learn==1.3.0
numpy==1.26.0
pandas==2.1.0

# Deep learning (optional stubs)
torch==2.1.0
torchvision==0.16.0

# Explainability
shap==0.43.0
lime==0.2.0.1

# Reporting
reportlab==4.0.8

# API integration
requests==2.31.0

# Visualization
matplotlib==3.8.0

# Testing
pytest==7.4.0
```

### 🧩 How This Fits Into Our Project
- **Flask** → Web framework for routes and app factory.
- **SQLite** → Lightweight storage layer.
- **Statsmodels** → SARIMAX and classical forecasting.
- **Scikit-learn** → Random Forest, Gradient Boosting, Ridge meta learner.
- **Torch** → LSTM/Autoencoder stubs for deep learning.
- **SHAP/LIME** → Explainability layer.
- **ReportLab** → Governance-ready PDF reports.
- **Requests** → Hugging Face + Weather API integration.
- **Matplotlib** → Visualization of forecasts.
- **Pytest** → Robust test suite validation.

### 🔧 Example Usage
```bash
# Install dependencies
pip install -r requirements.txt
```

This file completes the **dependency management layer**, ensuring our Weather Aggregator demo is portable and reproducible.  


## 2.10 notebooks folder

## 2.10.1 📄 `01_flask_stub.ipynb`

This notebook will set up the local website structure, input form, and routing logic.

It:
- Accepts user input (PO Box, City, Country).
- Stores it in a lightweight SQLite database.
- Displays stubbed weather data.
- Ready for integration with weather APIs and ensemble models.

## 2.10.2 📄 `02_model_dev.ipynb`

This notebook will let us prototype the ensemble learning layer before wiring it into the Flask app. 

It: 
- Creates synthetic weather data.
- Fits multiple base models (SARIMAX, RF, GB).
- Aggregates predictions via Ridge regression.
- Visualizes ensemble forecast with confidence bands.  

In addition, the notebook also extends the scaffold with **Hugging Face hosted model integration** so we can start experimenting with ensembles of ~30 models. 
This will be the backbone of our `02_model_dev.ipynb` before we modularize into `models/huggingface_models.py`. 

The portion of the notebook: 
- Connects Flask backend to Hugging Face hosted models **without local downloads**.
- Allows scaling to ~30 models for ensemble diversity.
- Keeps CPU/memory lean by offloading inference.
- Meta learner aggregates both **local statistical models** and **remote transformer-based models**.

## 2.10.3 📄 `03_huggingface_api.ipynb`

This notebook focuses on integrating Hugging Face hosted models into our Weather Aggregator pipeline. This notebook is designed for prototyping API calls, batch queries, and ensemble-ready outputs.

It: 
- Connects to Hugging Face hosted models via API.
- Queries multiple models sequentially (scalable to ~30).
- Collects outputs for ensemble integration.
- Demonstrates meta learner aggregation with local + hosted forecasts.  

The notebook is extended with a **parallel query scaffold** so we can efficiently scale Hugging Face calls across ~30 models. This will reduce latency and make ensemble inference more practical.  

This notebook extension demonstrates **scalable hosted inference**:
- Parallelization reduces latency for ensemble runs.
- Async queries show readiness for enterprise‑scale workloads.
- Both approaches are modular and governance‑ready.

## 2.10.4 📄 `04_dashboard.ipynb`

This notebook builds a simple visualization dashboard to display forecasts, confidence bands, and explainability results. This notebook is designed for prototyping before we modularize into Flask templates.

It:
- **Forecast Visualization**: Plots forecasts with confidence bands for each model.
- **Ensemble Aggregation**: Shows meta learner output alongside base models.
- **Explainability**: Demonstrates SHAP beeswarm plot for feature importance.
- **Consulting Narrative**: Provides a dashboard view that executives can interpret easily.

Thus, this notebook is a **prototype dashboard** — once validated, we can modularize the plots into Flask templates or export them into your PDF reports. 

---

# 3. GUI design and its user interaction flow

Let us walk through the user interaction flow for our Weather Aggregator app.
I will break it down into intuitive stages:  

Here’s the **data flow diagram walkthrough** for our Weather Aggregator system — this ties together the Flask frontend, 
local models, Hugging Face hosted models, and the meta learner before results are sent back to the user dashboard.

## 3.1 🔄 Data Flow Overview

### 1. **User Input (Frontend)**
- PO Box, City, Country entered into the Flask form.
- Request sent to Flask backend.

### 2. **Flask Backend**
- Validates input.
- Stores metadata in LiteSQL (SQLite).
- Triggers weather API call for raw data.

### 3. **Weather API Integration**
- Retrieves current conditions + 3–5 day forecast.
- Normalizes data into time series format.

### 4. **Local Models**
- SARIMAX → classical time series forecast.
- Random Forest / Gradient Boosting → nonlinear regression.
- CNN/LSTM → deep learning temporal/spatial patterns.
- Autoencoder → anomaly detection.
- Kalman filter → physics-inspired smoothing.

### 5. **Hugging Face Hosted Models**
- Input sequence sent via API to ~30 diverse models.
- Each model returns predictions or embeddings.
- Results cached for reproducibility.

### 6. **Meta Learner**
- Aggregates local + hosted model outputs.
- Options: Ridge regression, logistic regression, gradient boosting, shallow neural net.
- Produces final forecast + confidence bands.

### 7. **Governance Layer**
- Logs model versions, parameters, timestamps.
- Stores raw data, forecasts, and PDF reports.
- Ensures reproducibility and auditability.

### 8. **User Dashboard**
- Displays:
  - Current weather card.
  - 3–5 day ensemble forecast.
  - Confidence intervals.
  - Downloadable PDF report.

### 9. 🎯 Executive Narrative
This flow demonstrates **modularity, transparency, and operational reliability**:
- **Frontend simplicity** → user-friendly input and display.
- **Backend orchestration** → scalable integration of APIs and models.
- **Ensemble diversity** → ~30 models ensure robust statistics.
- **Governance-first design** → audit trails and reproducibility baked in.

## 3.2 `run.py`:

This is essentially the **entry point** for our Weather Aggregator Flask application:

### 📂 File Purpose
- **Role**: Defines how the Flask app is created and started.
- **Context**: This is the script you run (`python run.py`) to launch the web server locally.

### 🧩 Key Components

#### 1. Import Statements
```python
from flask import Flask
from app.routes import bp as routes_bp
```
- Imports the Flask framework.
- Imports your blueprint (`bp`) from `app.routes`.  
  - Blueprints are modular containers for routes, templates, and logic.  
  - Here, `routes_bp` holds all your URL endpoints (`index`, `forecast`, `report`).

#### 2. `create_app()` Function
```python
def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "supersecretkey"  # replace with env var in production
    app.register_blueprint(routes_bp)
    return app
```
- **Creates a Flask application instance**.
- Sets a `SECRET_KEY`:
  - Used by Flask for session management, CSRF protection, and secure cookies.
  - In production, this should be stored as an environment variable, not hard‑coded.
- Registers the blueprint (`routes_bp`), which attaches all your routes (`/`, `/forecast/<city>/<country>`, `/report/<city>/<country>`) to the app.
- Returns the configured app object.

#### 3. Entry Point Logic
```python
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
```
- Ensures the script only runs when executed directly (not when imported).
- Calls `create_app()` to build the app.
- Starts the Flask development server with `debug=True`:
  - Enables hot reloading (server restarts when code changes).
  - Provides detailed error pages for debugging.

### 🔑 Summary
- `run.py` is the **bootstrap file** for your Flask app.
- It:
  1. Creates the app instance.
  2. Configures security and blueprints.
  3. Runs the server in debug mode for development.

👉 In production, we would typically:
- Remove `debug=True`.
- Use a WSGI server like **Gunicorn** or **uWSGI** to run the app.
- Store `SECRET_KEY` securely in environment variables.


## 3.3 `forms.py`:

### 📂 File Purpose
- **Role**: Defines the **input form** that users see on the landing page (`index.html`).
- **Context**: Uses **WTForms** (via Flask‑WTF) to handle form rendering, validation, and submission securely.

### 🧩 Key Components

#### 1. Imports
```python
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length
```
- **FlaskForm**: Base class for all forms in Flask‑WTF.
- **StringField**: Input fields for text (PO Box, City, Country).
- **SubmitField**: A button to submit the form.
- **Validators**:
  - `DataRequired`: Ensures the field is not left empty.
  - `Length`: Enforces minimum and maximum character lengths.

#### 2. `LocationForm` Class
```python
class LocationForm(FlaskForm):
    """Form for user to enter location details."""
```
Defines a form with three text fields and one submit button.

##### Fields:
- **PO Box**
  ```python
  po_box = StringField(
      "PO Box",
      validators=[DataRequired(), Length(min=3, max=10)],
      render_kw={"placeholder": "Enter PO Box"}
  )
  ```
  - Required field.
  - Must be between 3–10 characters.
  - Placeholder text shown in the input box.

- **City**
  ```python
  city = StringField(
      "City",
      validators=[DataRequired(), Length(min=2, max=50)],
      render_kw={"placeholder": "Enter City"}
  )
  ```
  - Required field.
  - Must be between 2–50 characters.

- **Country**
  ```python
  country = StringField(
      "Country",
      validators=[DataRequired(), Length(min=2, max=50)],
      render_kw={"placeholder": "Enter Country"}
  )
  ```
  - Required field.
  - Must be between 2–50 characters.

- **Submit Button**
  ```python
  submit = SubmitField("Get Forecast")
  ```
  - Renders a button labeled **“Get Forecast”**.
  - When clicked, triggers form validation and submission.

### 🔑 Summary
- `forms.py` defines the **LocationForm**, which captures PO Box, City, and Country from the user.
- Each field has **validation rules** to prevent empty or invalid input.
- The form integrates seamlessly with Flask routes (`index`), where `form.validate_on_submit()` checks validity before redirecting to the forecast page.
- This ensures clean, validated user input flows into your forecasting pipeline.


## 3.4 `routes.py`:

### 📂 File Purpose
- **Role**: Defines all the Flask routes (URLs/endpoints) for the application.
- **Context**: Handles user input, orchestrates forecasting logic, saves results, and renders templates (`index.html`, `forecast.html`, `report.html`).

### 🧩 Key Components

#### 1. Imports and Setup
```python
from flask import Blueprint, render_template, request, redirect, url_for, flash
from app.forms import LocationForm
from data.weather_api import fetch_weather
from data.data_store import DataStore
from data.pdf_report import PDFReport
from models import base_models, meta_learner
from governance.logger import get_logger
import numpy as np
```
- **Blueprint**: Allows modular route definitions (`bp`).
- **LocationForm**: User input form (PO Box, City, Country).
- **fetch_weather**: API call to OpenWeatherMap for current weather data.
- **DataStore**: SQLite persistence layer for forecasts and user inputs.
- **PDFReport**: Generates governance‑ready PDF reports.
- **base_models**: Collection of forecasting models (SARIMAX, Kalman, RF, GB, CNN, LSTM, Autoencoder).
- **meta_learner**: Surrogate ensemble model (ridge regression by default).
- **logger**: Centralized logging for traceability.
- **NumPy**: Used for averaging and statistics.

#### 2. Index Route (`/`)
```python
@bp.route("/", methods=["GET", "POST"])
def index():
    form = LocationForm()
    if form.validate_on_submit():
        po_box = form.po_box.data
        city = form.city.data
        country = form.country.data

        store.save_user_input(po_box, city, country)
        logger.info(f"User input saved: {po_box}, {city}, {country}")

        return redirect(url_for("routes.forecast", city=city, country=country))
    return render_template("index.html", form=form)
```
- Displays the **landing page** with the input form.
- Validates user input (PO Box, City, Country).
- Saves input to the database.
- Redirects to the forecast page for the chosen location.

#### 3. Forecast Route (`/forecast/<city>/<country>`)
```python
@bp.route("/forecast/<city>/<country>", methods=["GET"])
def forecast(city, country):
    ...
```
- **Fetches weather data** via API.
- Runs **all base models** to generate predictions.
- Collects predictions into a matrix `X`.
- Constructs target `y` as the average of base predictions.
- Fits the **MetaLearner** (ridge regression surrogate).
- Predicts the **ensemble forecast**.
- Computes **confidence bands** (±5% placeholder).
- Saves results in the database.
- Generates a **PDF report**.
- Renders `forecast.html` with:
  - Base model forecasts
  - Ensemble forecast
  - Link to the generated report

#### 4. Report Route (`/report/<city>/<country>`)
```python
@bp.route("/report/<city>/<country>", methods=["GET"])
def report(city, country):
    ...
```
- Retrieves forecasts for the given city/country from the database.
- Builds a dictionary of **confidence bands** keyed by timestamp.
- Collects numeric forecast values for statistics.
- Computes **summary stats**: average, minimum, maximum.
- Logs report generation.
- Renders `report.html` with:
  - Forecast history
  - Confidence bands
  - Summary statistics

### 🔑 Summary
- `routes.py` is the **controller layer** of your app.
- It connects:
  - **Frontend templates** (`index.html`, `forecast.html`, `report.html`)
  - **Backend models** (base models + meta learner)
  - **Persistence layer** (SQLite datastore)
  - **Governance tools** (logging, PDF reporting)
- Provides a full workflow:
  1. User enters location.
  2. Forecasts are generated by base models and ensemble learner.
  3. Results are saved, logged, and reported.
  4. User can view both forecasts and governance‑ready reports.


## 3.5 📄 `scripts.js`

### 📂 File Purpose
- **Role**: Adds lightweight interactivity and user experience enhancements to the frontend.
- **Context**: Runs in the browser after the page loads, manipulating DOM elements to improve usability.

### 🧩 Key Components

#### 1. DOM Ready Event
```javascript
document.addEventListener("DOMContentLoaded", function() {
    console.log("Weather Aggregator frontend loaded ✅");
    ...
});
```
- Ensures the script runs only after the HTML document has fully loaded.
- Logs a confirmation message to the browser console for debugging.

#### 2. Flash Message Auto‑Hide
```javascript
const flashes = document.querySelectorAll(".flash-message");
flashes.forEach(msg => {
    setTimeout(() => {
        msg.style.display = "none";
    }, 4000);
});
```
- Selects all elements with the class `.flash-message` (used by Flask’s `flash()` system).
- Automatically hides each flash message after **4 seconds**.
- Prevents clutter and ensures temporary notifications disappear gracefully.

#### 3. Forecast Section Toggle
```javascript
const toggleBtn = document.getElementById("toggle-forecast");
if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
        const section = document.querySelector(".forecast-section");
        if (section.style.display === "none") {
            section.style.display = "block";
            toggleBtn.innerText = "Hide Forecasts";
        } else {
            section.style.display = "none";
            toggleBtn.innerText = "Show Forecasts";
        }
    });
}
```
- Looks for a button with the ID `toggle-forecast`.
- When clicked:
  - Toggles the visibility of the `.forecast-section` element.
  - Updates the button text dynamically between **“Show Forecasts”** and **“Hide Forecasts”**.
- Provides users control over whether they want to see or hide the detailed base model forecasts.

### 🔑 Summary
- `scripts.js` enhances the frontend by:
  1. **Auto‑hiding flash messages** after a short delay.
  2. **Adding toggle functionality** for forecast sections.
- Keeps the interface clean, responsive, and user‑friendly without requiring page reloads.
- Complements your Flask templates (`forecast.html`, `report.html`) by adding dynamic behavior.


## 3.6 📄 `style.css`

### 📂 File Purpose
- **Role**: Provides the **visual styling** for the frontend templates (`index.html`, `forecast.html`, `report.html`).
- **Context**: Ensures a clean, modern, and consistent look across the app, with responsive layout and user‑friendly design.

### 🧩 Key Styling Sections

#### 1. Global Styles
```css
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background: #f4f7fa;
    color: #333;
}
```
- Sets a default font (`Arial`).
- Removes default margins/padding.
- Applies a light gray background (`#f4f7fa`) for a soft, professional feel.
- Uses dark gray text (`#333`) for readability.

#### 2. Header
```css
header {
    background: #2c3e50;
    color: #fff;
    padding: 1rem;
    text-align: center;
}
header h1 {
    margin: 0;
    font-size: 1.8rem;
}
```
- Dark blue/gray header bar (`#2c3e50`).
- White text for contrast.
- Centered title with larger font size.

#### 3. Navigation Links
```css
nav a {
    color: #ecf0f1;
    margin: 0 10px;
    text-decoration: none;
}
```
- Light gray links (`#ecf0f1`) against the dark header.
- Spacing between links.
- Removes underlines for a clean look.

#### 4. Main Content
```css
main {
    padding: 2rem;
}
```
- Adds padding around the main content area for breathing room.

#### 5. Forms
```css
form {
    background: #fff;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    max-width: 400px;
    margin: auto;
}
```
- White card background for forms.
- Rounded corners and subtle shadow for depth.
- Centered form with max width for readability.

Form inputs and buttons:
```css
form input, form button {
    width: 100%;
    padding: 0.6rem;
    border: 1px solid #ccc;
    border-radius: 4px;
}
form button {
    background: #3498db;
    color: #fff;
    border: none;
    cursor: pointer;
}
form button:hover {
    background: #2980b9;
}
```
- Full‑width inputs and buttons.
- Blue button (`#3498db`) with hover effect (`#2980b9`).
- Rounded corners for modern design.

#### 6. Content Cards
```css
.weather-card, .forecast-section, .ensemble-section, .report-section {
    background: #fff;
    padding: 1.5rem;
    margin: 1rem auto;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    max-width: 600px;
}
```
- White card containers for weather, forecasts, ensemble, and reports.
- Consistent styling with rounded corners and shadows.
- Max width for readability.

#### 7. Buttons
```css
.btn {
    display: inline-block;
    padding: 0.6rem 1.2rem;
    background: #27ae60;
    color: #fff;
    text-decoration: none;
    border-radius: 4px;
}
.btn:hover {
    background: #1e8449;
}
```
- Green action buttons (`#27ae60`).
- White text for contrast.
- Hover effect darkens the green (`#1e8449`).

#### 8. Footer
```css
footer {
    text-align: center;
    padding: 1rem;
    background: #2c3e50;
    color: #fff;
    margin-top: 2rem;
}
```
- Matches header styling (dark background, white text).
- Centered content.
- Provides closure at the bottom of the page.

### 🔑 Summary
- `style.css` defines a **modern, card‑based UI** with consistent spacing, shadows, and rounded corners.
- Uses a **blue/green accent palette** for buttons and headers.
- Ensures forms, forecasts, and reports are visually distinct but stylistically unified.
- Provides a professional, consulting‑ready look aligned with the app’s explainability and governance goals.

## 3.7 📄 `base.html`

### 📂 File Purpose
- **Role**: Acts as the **base template** for all other HTML pages (`index.html`, `forecast.html`, `report.html`).
- **Context**: Provides a consistent layout (header, navigation, footer) and includes shared CSS/JS assets.  
- **Mechanism**: Uses Jinja2’s `{% block content %}` to allow child templates to inject page‑specific content.

### 🧩 Key Components

#### 1. Document Setup
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Weather Aggregator</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <script src="{{ url_for('static', filename='scripts.js') }}"></script>
</head>
```
- Declares HTML5 document type and language (`en`).
- Sets character encoding (`UTF-8`).
- Defines the page title: **Weather Aggregator**.
- Loads **CSS styling** (`style.css`) and **JavaScript interactivity** (`scripts.js`) from the `static/` folder using Flask’s `url_for`.

#### 2. Header
```html
<header>
    <h1>🌦 Weather Aggregator</h1>
    <nav>
        <a href="{{ url_for('routes.index') }}">Home</a>
    </nav>
</header>
```
- Displays the app title with a weather emoji 🌦 for branding.
- Provides a navigation bar with a **Home** link (points to the `index` route).
- This section is consistent across all pages.

#### 3. Main Content Block
```html
<main>
    {% block content %}{% endblock %}
</main>
```
- Defines a placeholder block (`content`) that child templates override.
- Example:
  - `index.html` injects the location input form.
  - `forecast.html` injects base model forecasts and ensemble results.
  - `report.html` injects tables of historical forecasts and confidence bands.

#### 4. Footer
```html
<footer>
    <p>&copy; 2025 Weather Aggregator Demo</p>
</footer>
```
- Provides a consistent footer across all pages.
- Displays copyright.

### 🔑 Summary
- `base.html` is the **foundation template** for your app.
- It ensures:
  - Shared styling and scripts are loaded everywhere.
  - A consistent header, navigation, and footer.
  - Page‑specific content is injected via `{% block content %}`.
- This approach reduces duplication and makes the GUI architecture modular and maintainable.

## 3.8 📄 `forecast.html`

### 📂 File Purpose
- **Role**: Displays the forecast results for a given city and country.
- **Context**: Extends the shared `base.html` layout, injecting forecast‑specific content into the `{% block content %}` section.
- **Functionality**: Shows base model predictions, the ensemble forecast, and provides a link to the report view.

### 🧩 Key Components

#### 1. Template Inheritance
```jinja
{% extends "base.html" %}
```
- Inherits the header, navigation, footer, and styling from `base.html`.
- Ensures consistent look and feel across all pages.

#### 2. Content Block
```jinja
{% block content %}
<h2>Forecast for {{ city }}, {{ country }}</h2>
...
{% endblock %}
```
- Defines the page‑specific content injected into the `main` section of `base.html`.
- Displays the location (city and country) dynamically, passed in from the Flask route.

#### 3. Base Model Forecasts
```jinja
<div class="forecast-section">
    <h3>Base Model Forecasts</h3>
    <ul>
        {% for model, forecast in forecasts.items() %}
        <li><strong>{{ model }}:</strong> {{ forecast }}</li>
        {% endfor %}
    </ul>
</div>
```
- Iterates over the `forecasts` dictionary passed from the route.
- Displays each base model name (SARIMAX, Kalman, RandomForest, GradientBoosting, CNN, LSTM, Autoencoder) alongside its forecast value.
- Provides transparency into individual model outputs.

#### 4. Ensemble Forecast
```jinja
<div class="ensemble-section">
    <h3>Ensemble Forecast</h3>
    <p>{{ ensemble }}</p>
</div>
```
- Highlights the **ensemble forecast** (computed by the meta learner).
- Displayed prominently to emphasize its role as the aggregated prediction.

#### 5. Report Link
```jinja
<div class="report-section">
    <a href="{{ url_for('routes.report', city=city, country=country) }}" class="btn">Download PDF Report</a>
</div>
```
- Provides a button linking to the `/report/<city>/<country>` route.
- Despite the label “Download PDF Report,” this currently navigates to the **HTML report page** (`report.html`) with tables and summary statistics.
- Styling (`btn` class) makes it visually distinct.

### 🔑 Summary
- `forecast.html` is the **results page** for forecasts.
- It:
  1. Shows base model predictions.
  2. Highlights the ensemble forecast.
  3. Links to the report view for deeper analysis.
- Works in tandem with `routes.py` (which passes `forecasts`, `ensemble`, and location data) and `report.html` (which provides detailed reporting).

## 3.9 🏗 `index.html`

### 📂 File Purpose
- **Role**: Serves as the **landing page** of the application.
- **Context**: Extends the shared `base.html` layout and injects the location input form plus optional current weather display.
- **Functionality**: Collects user input (PO Box, City, Country) and optionally shows current weather data if available.

### 🧩 Key Components

#### 1. Template Inheritance
```jinja
{% extends "base.html" %}
```
- Inherits the header, navigation, footer, and styling from `base.html`.
- Ensures consistent design across all pages.

#### 2. Content Block
```jinja
{% block content %}
<h2>Enter Location</h2>
...
{% endblock %}
```
- Defines the page‑specific content injected into the `main` section of `base.html`.
- Displays a heading prompting the user to enter location details.

#### 3. Input Form
```jinja
<form method="POST" action="{{ url_for('routes.index') }}">
    {{ form.hidden_tag() }}
    <div>
        {{ form.po_box.label }}<br>
        {{ form.po_box(size=20) }}
    </div>
    <div>
        {{ form.city.label }}<br>
        {{ form.city(size=20) }}
    </div>
    <div>
        {{ form.country.label }}<br>
        {{ form.country(size=20) }}
    </div>
    <div>
        {{ form.submit() }}
    </div>
</form>
```
- Uses **WTForms** (`LocationForm`) to render fields:
  - **PO Box** (with label and input box).
  - **City**.
  - **Country**.
  - **Submit button** labeled “Get Forecast.”
- `form.hidden_tag()` ensures CSRF protection (security feature provided by Flask‑WTF).
- Submits via POST to the `routes.index` route, which validates input and redirects to the forecast page.

#### 4. Current Weather Card (Optional)
```jinja
{% if weather %}
<div class="weather-card">
    <h3>Current Weather</h3>
    <p>Temperature: {{ weather.temperature }} °C</p>
    <p>Condition: {{ weather.condition }}</p>
</div>
{% endif %}
```
- Conditionally displays a **weather card** if `weather` data is passed from the route.
- Shows:
  - Current temperature (`weather.temperature`).
  - Weather condition (`weather.condition`).
- Provides immediate feedback to the user before forecasts are generated.

### 🔑 Summary
- `index.html` is the **entry point** for user interaction.
- It:
  1. Collects location details via a secure form.
  2. Submits input to the backend for processing.
  3. Optionally displays current weather fetched from the API.
- Works in tandem with:
  - `forms.py` (defines the form fields and validation).
  - `routes.py` (`index` route handles form submission and weather fetching).
  - `style.css` (styles the form and weather card).

## 3.10 🏗 `report.html`

### 📂 File Purpose
- **Role**: Provides the **report view** for forecasts of a given city and country.
- **Context**: Extends the shared `base.html` layout and injects reporting content into the `{% block content %}` section.
- **Functionality**: Displays summary statistics, historical forecasts, confidence bands, and a navigation link back to the forecast page.

### 🧩 Key Components

#### 1. Template Inheritance
```jinja
{% extends "base.html" %}
```
- Inherits the header, navigation, footer, and styling from `base.html`.
- Ensures consistent design across all pages.

#### 2. Content Block
```jinja
{% block content %}
<h2>Forecast Report for {{ city }}, {{ country }}</h2>
...
{% endblock %}
```
- Defines the page‑specific content injected into the `main` section of `base.html`.
- Displays the location (city and country) dynamically, passed in from the Flask route.

#### 3. Summary Statistics
```jinja
<div class="summary-section">
    <h3>Summary Statistics</h3>
    <ul>
        <li><strong>Average Forecast:</strong> {{ avg_forecast }}</li>
        <li><strong>Minimum Forecast:</strong> {{ min_forecast }}</li>
        <li><strong>Maximum Forecast:</strong> {{ max_forecast }}</li>
    </ul>
</div>
```
- Shows simple descriptive statistics for the forecasts:
  - Average forecast value.
  - Minimum forecast value.
  - Maximum forecast value.
- Values are computed in `routes.py` and passed into the template.

#### 4. Forecast History Table
```jinja
<div class="history-section">
    <h3>Forecast History</h3>
    <table>
        <thead>
            <tr>
                <th>Forecast</th>
                <th>Lower Band</th>
                <th>Upper Band</th>
                <th>Timestamp</th>
            </tr>
        </thead>
        <tbody>
            {% for forecast, lower, upper, timestamp in forecasts %}
            <tr>
                <td>{{ forecast }}</td>
                <td>{{ lower }}</td>
                <td>{{ upper }}</td>
                <td>{{ timestamp }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
```
- Iterates over the `forecasts` list retrieved from the datastore.
- Displays each forecast value, its confidence bands, and the timestamp.
- Provides traceability and historical context.

#### 5. Confidence Bands Table
```jinja
<div class="bands-section">
    <h3>Confidence Bands</h3>
    <table>
        <thead>
            <tr>
                <th>Timestamp</th>
                <th>Lower Band</th>
                <th>Upper Band</th>
            </tr>
        </thead>
        <tbody>
            {% for ts, bands in confidence_bands.items() %}
            <tr>
                <td>{{ ts }}</td>
                <td>{{ bands[0] }}</td>
                <td>{{ bands[1] }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
```
- Iterates over the `confidence_bands` dictionary (timestamp → [lower, upper]).
- Displays uncertainty ranges for each forecast.
- Reinforces governance and risk awareness.

#### 6. Back Navigation
```jinja
<div class="back-section">
    <a href="{{ url_for('routes.forecast', city=city, country=country) }}" class="btn">Back to Forecast</a>
</div>
```
- Provides a styled button (`btn` class) to return to the forecast page.
- Improves navigation and user experience.

### 🔑 Summary
- `report.html` is the **reporting view** of the app.
- It:
  1. Shows summary statistics (average, min, max).
  2. Displays historical forecasts with confidence bands and timestamps.
  3. Provides a separate confidence band table for clarity.
  4. Includes navigation back to the forecast page.
- Works in tandem with:
  - `routes.py` (`report` route computes stats and passes data).
  - `style.css` (styles tables, sections, and buttons).
  - `pdf_report.py` (generates downloadable governance‑ready reports).
  
## 3.11 `weather_api.py`

### 📂 File Purpose
- **Role**: Integrates with the **OpenWeatherMap API** (or similar) to fetch live weather data.
- **Context**: Provides normalized weather information (temperature, condition, and a stub time series) for use by forecasting models.
- **Functionality**: Acts as the **data ingestion layer** of your app.

### 🧩 Key Components

#### 1. Imports
```python
import requests
from governance.config import Config
from governance.logger import get_logger
```
- **requests**: Used to send HTTP requests to the weather API.
- **Config**: Holds the API key (`Config.WEATHER_API_KEY`) and other configuration values.
- **get_logger**: Provides centralized logging for traceability.

#### 2. Logger Setup
```python
logger = get_logger(__name__)
```
- Creates a logger instance for this module.
- Ensures all API calls and errors are logged for governance and debugging.

#### 3. `fetch_weather` Function
```python
def fetch_weather(city, country):
    """Fetch current weather and normalized time series."""
```
- Main function to retrieve weather data for a given **city** and **country**.

##### API Request
```python
url = f"http://api.openweathermap.org/data/2.5/weather?q={city},{country}&appid={Config.WEATHER_API_KEY}&units=metric"
response = requests.get(url)
```
- Constructs the API URL with:
  - City and country.
  - API key from `Config`.
  - Units set to metric (°C).
- Sends a GET request to OpenWeatherMap.

##### Success Case
```python
if response.status_code == 200:
    data = response.json()
    logger.info(f"Weather data fetched for {city}, {country}")
    return {
        "temperature": data["main"]["temp"],
        "condition": data["weather"][0]["description"],
        "temperature_series": [data["main"]["temp"] - i for i in range(5)]  # stub series
    }
```
- Parses JSON response.
- Logs successful fetch.
- Returns a dictionary with:
  - **temperature**: Current temperature in °C.
  - **condition**: Weather description (e.g., "clear sky").
  - **temperature_series**: A stub time series (current temp minus 0–4).  
  *(This is placeholder data for testing; in production, you’d replace it with actual historical/forecast series.)*

##### Error Case
```python
else:
    logger.error(f"Error fetching weather data: {response.text}")
    return {"temperature_series": []}
```
- Logs the error message.
- Returns an empty series to prevent downstream crashes.

### 🔑 Summary
- `weather_api.py` is the **data ingestion module**.
- It:
  1. Calls OpenWeatherMap API with city/country input.
  2. Logs success or failure.
  3. Returns normalized weather data (temperature, condition, stub series).
- Provides the **raw input** for forecasting models in `base_models.py` and ensemble logic in `routes.py`.


## 3.12 `pdf_report.py`

### 📂 File Purpose
- **Role**: Generates governance‑ready PDF reports summarizing forecasts.
- **Context**: Uses the **ReportLab** library to create portable, shareable documents that can be distributed to stakeholders.
- **Functionality**: Converts forecast results (city, country, ensemble prediction) into a structured PDF file.

### 🧩 Key Components

#### 1. Imports
```python
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
```
- **ReportLab**: Provides PDF generation capabilities.
  - `letter`: Standard US letter page size.
  - `canvas`: Object used to draw text and graphics onto the PDF.
- **os**: Imported but not used in this snippet (could be useful for file path handling in extensions).

#### 2. `PDFReport` Class
```python
class PDFReport:
    def __init__(self, city, country, forecast):
        self.city = city
        self.country = country
        self.forecast = forecast
```
- Encapsulates PDF report generation logic.
- Stores the **city**, **country**, and **forecast** values as instance attributes.

#### 3. `generate()` Method
```python
def generate(self):
    filename = f"{self.city}_{self.country}_forecast.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
```
- Creates a filename based on city and country (e.g., `Berlin_Germany_forecast.pdf`).
- Initializes a ReportLab canvas with letter page size.

##### Title Section
```python
c.setFont("Helvetica-Bold", 16)
c.drawString(72, 720, f"Forecast Report for {self.city}, {self.country}")
```
- Sets font to bold Helvetica, size 16.
- Draws the report title near the top of the page.

##### Forecast Content
```python
c.setFont("Helvetica", 12)
c.drawString(72, 680, f"Forecast: {self.forecast}")
```
- Switches to regular Helvetica, size 12.
- Prints the forecast value.

##### Footer/Metadata
```python
c.drawString(72, 660, "Generated by Weather Aggregator Demo")
```
- Adds a simple footer line for provenance and governance.

##### Finalization
```python
c.showPage()
c.save()
return filename
```
- Finalizes the page.
- Saves the PDF file to disk.
- Returns the filename for downstream use (e.g., linking in the forecast route).

### 🔑 Summary
- `pdf_report.py` is the **report generation module**.
- It:
  1. Creates a PDF file named after the city and country.
  2. Adds a title, forecast value, and metadata.
  3. Saves the file and returns its path.
- Provides governance‑ready, portable documentation of forecasts, complementing the HTML report (`report.html`).

## 3.13 `data_store.py`

### 📂 File Purpose
- **Role**: Provides a persistence layer using **SQLite** for forecasts and user inputs.
- **Context**: Ensures that forecasts, confidence bands, and user‑entered locations are stored and retrievable for reporting and governance.
- **Functionality**: Encapsulates database schema creation, insert operations, and retrieval queries in a single class (`DataStore`).

### 🧩 Key Components

#### 1. Imports and Logger
```python
import sqlite3
import logging

logger = logging.getLogger(__name__)
```
- **sqlite3**: Python’s built‑in library for SQLite database operations.
- **logging**: Used to log database operations for traceability.
- **logger**: Captures events like saving forecasts or user inputs.

#### 2. `DataStore` Class Initialization
```python
class DataStore:
    def __init__(self, db_path="weather_data.db"):
        self.db_path = db_path
        self._init_db()
```
- Initializes the datastore with a default database file (`weather_data.db`).
- Calls `_init_db()` to ensure tables exist.

#### 3. Database Schema Setup
```python
def _init_db(self):
    """Initialize database schema if not exists."""
    conn = sqlite3.connect(self.db_path)
    cur = conn.cursor()
```
Creates two tables if they don’t already exist:

- **Forecasts Table**
```sql
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    forecast REAL NOT NULL,
    lower_band REAL NOT NULL,
    upper_band REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
```
Stores:
- Forecast value
- Confidence bands (lower, upper)
- City, country
- Timestamp (auto‑generated)

- **User Inputs Table**
```sql
CREATE TABLE IF NOT EXISTS user_inputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    po_box TEXT,
    city TEXT NOT NULL,
    country TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
```
Stores:
- User input details (PO Box, City, Country)
- Timestamp (auto‑generated)

#### 4. Saving Forecasts
```python
def save_forecast(self, city, country, forecast, lower_band, upper_band):
    ...
```
- Inserts a new forecast record into the `forecasts` table.
- Logs the operation with forecast value and confidence bands.

#### 5. Retrieving Forecasts
```python
def get_forecasts(self, city, country):
    ...
```
- Queries the `forecasts` table for a given city/country.
- Returns rows ordered by timestamp (latest first).
- Each row contains: forecast, lower_band, upper_band, timestamp.

#### 6. Saving User Inputs
```python
def save_user_input(self, po_box, city, country):
    ...
```
- Inserts a new user input record into the `user_inputs` table.
- Logs the operation with PO Box, city, and country.

#### 7. Retrieving User Inputs
```python
def get_user_inputs(self):
    ...
```
- Queries the `user_inputs` table.
- Returns all saved inputs ordered by timestamp (latest first).

### 🔑 Summary
- `data_store.py` is the **persistence backbone** of the app.
- It:
  1. Creates and manages SQLite tables for forecasts and user inputs.
  2. Saves forecasts with confidence bands.
  3. Saves user input details.
  4. Retrieves historical forecasts and inputs for reporting.
- Ensures **traceability, reproducibility, and governance** by persisting all critical data.


## 3.14 `cache.py`

### 📂 File Purpose
- **Role**: Implements a **simple in‑memory caching layer** for API responses.
- **Context**: Helps reduce redundant calls to external services (like OpenWeatherMap) by temporarily storing results.
- **Functionality**: Stores key–value pairs with a time‑to‑live (TTL) expiration.

### 🧩 Key Components

#### 1. Imports
```python
import time
```
- Provides access to the current system time (`time.time()`), used for tracking when cache entries were stored.

#### 2. `Cache` Class
```python
class Cache:
    def __init__(self, ttl=300):
        self.ttl = ttl
        self.store = {}
```
- **`ttl` (time‑to‑live)**: Default is 300 seconds (5 minutes).
- **`store`**: Dictionary that holds cached items in the format:
  ```python
  { key: (value, timestamp) }
  ```
  where `timestamp` is the time the item was cached.

#### 3. `set()` Method
```python
def set(self, key, value):
    self.store[key] = (value, time.time())
```
- Saves a value in the cache under a given key.
- Records the current time so expiration can be checked later.

#### 4. `get()` Method
```python
def get(self, key):
    if key in self.store:
        value, timestamp = self.store[key]
        if time.time() - timestamp < self.ttl:
            return value
        else:
            del self.store[key]
    return None
```
- Retrieves a cached value if:
  - The key exists.
  - The entry has not expired (`current_time - timestamp < ttl`).
- If expired, deletes the entry and returns `None`.
- If not found, returns `None`.

#### 5. `clear()` Method
```python
def clear(self):
    self.store.clear()
```
- Clears all entries from the cache.
- Useful for resetting state or freeing memory.

### 🔑 Summary
- `cache.py` provides a **lightweight caching mechanism**:
  1. Stores API responses temporarily.
  2. Automatically expires entries after a configurable TTL.
  3. Prevents unnecessary repeated API calls.
- It’s simple, fast, and avoids external dependencies, making it ideal for small‑scale apps like your Weather Aggregator.


## 3.15 `config.py`

### 📂 File Purpose
- **Role**: Acts as the **central configuration hub** for the application.
- **Context**: Stores API keys, endpoints, and model registry settings in one place, making the app easier to maintain and extend.
- **Functionality**: Provides a single source of truth for external service credentials and model parameters.

### 🧩 Key Components

#### 1. `Config` Class
```python
class Config:
```
- Encapsulates all configuration values in a single class.
- Allows other modules to import and reference settings consistently.

#### 2. API Keys
```python
WEATHER_API_KEY = "Insert your OpenWeatherMap API-Key"
HF_API_TOKEN = "Insert your HuggingFace API-Key"
```
- **`WEATHER_API_KEY`**: Required for accessing OpenWeatherMap’s weather data API.
- **`HF_API_TOKEN`**: Required for accessing HuggingFace’s inference API (e.g., for ML models).
- ⚠️ In production, these should be stored securely (e.g., environment variables or secrets manager), not hard‑coded.

#### 3. API URLs
```python
HF_API_URL = "https://api-inference.huggingface.co/models/"
```
- Base URL for HuggingFace’s inference API.
- Used to construct requests to specific models hosted on HuggingFace.

#### 4. Model Registry
```python
MODEL_REGISTRY = {
    "SARIMAX": {"order": (1,1,1)},
    "RandomForest": {"n_estimators": 50},
    "Kalman": {"window": 3},
    "ridge": {"alpha": 1.0},
    "logistic": {},
    "boosting": {"n_estimators": 100}
}
```
- Defines default parameters for each forecasting or meta‑learner model.
- **Examples**:
  - **SARIMAX**: Order `(1,1,1)` for autoregressive, differencing, and moving average terms.
  - **RandomForest**: 50 trees.
  - **Kalman**: Window size of 3.
  - **Ridge Regression**: Regularization parameter `alpha=1.0`.
  - **Logistic Regression**: No parameters specified (defaults used).
  - **Boosting**: 100 estimators.
- **Extendable**: New models can be added easily by updating this dictionary.

### 🔑 Summary
- `config.py` is the **configuration backbone** of the app.
- It:
  1. Stores API keys for external services.
  2. Defines endpoints for HuggingFace integration.
  3. Provides a registry of model parameters for forecasting and meta‑learning.
- Centralizes settings, making the app easier to maintain, extend, and secure.

## 3.16 `logger.py`

### 📂 File Purpose
- **Role**: Provides a **centralized logging utility** for the application.
- **Context**: Ensures that all modules (routes, models, data store, API calls) can log events consistently.
- **Functionality**: Captures model runs, parameters, timestamps, and errors into a single log file for governance and debugging.

### 🧩 Key Components

#### 1. Imports
```python
import logging
import os
```
- **logging**: Python’s built‑in logging library.
- **os**: Used to check and create the log directory if it doesn’t exist.

#### 2. Log Directory Setup
```python
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
```
- Defines a directory named `logs`.
- Ensures the directory exists before writing logs.
- Prevents errors when trying to create a log file in a missing folder.

#### 3. `get_logger()` Function
```python
def get_logger(name: str):
    """Return a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(LOG_DIR, "weather_aggregator.log"))
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

##### Steps:
1. **Retrieve logger**:
   - Creates or retrieves a logger instance by name (usually the module name).
2. **Prevent duplicate handlers**:
   - Checks if the logger already has handlers to avoid duplicate log entries.
3. **FileHandler**:
   - Writes logs to `logs/weather_aggregator.log`.
4. **Formatter**:
   - Defines log format:
     ```
     TIMESTAMP - LOGGER_NAME - LEVEL - MESSAGE
     ```
     Example:
     ```
     2025-12-06 10:22:00,123 - routes - INFO - Forecast saved for Berlin, Germany
     ```
5. **Attach handler**:
   - Adds the file handler to the logger.
6. **Set log level**:
   - Configures logger to capture `INFO` and above (INFO, WARNING, ERROR, CRITICAL).

### 🔑 Summary
- `logger.py` is the **logging backbone** of the app.
- It:
  1. Creates a centralized log file (`weather_aggregator.log`).
  2. Ensures consistent formatting across modules.
  3. Captures timestamps, module names, log levels, and messages.
- Provides **traceability and governance** by recording all critical events in one place.

## 3.17 `traceability.py` 

### 📂 File Purpose
- **Role**: Provides **model versioning and metadata tracking**.
- **Context**: Ensures forecasts are **reproducible and auditable** by recording model parameters, version hashes, and timestamps.
- **Functionality**: Acts as a lightweight registry for tracking which models (and configurations) were used in forecasting.

### 🧩 Key Components

#### 1. Imports
```python
import hashlib
import json
from datetime import datetime
```
- **hashlib**: Used to generate a unique hash (version identifier) for model parameters.
- **json**: Serializes parameters into a consistent format before hashing.
- **datetime**: Provides UTC timestamps for when models are registered.

#### 2. `Traceability` Class
```python
class Traceability:
    def __init__(self):
        self.registry = {}
```
- Initializes an empty dictionary (`registry`) to store metadata for registered models.
- Each entry is keyed by model name.

#### 3. `register_model()` Method
```python
def register_model(self, model_name, params):
    """Register model with parameters and version hash."""
    version_hash = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    metadata = {
        "model_name": model_name,
        "params": params,
        "version": version_hash,
        "timestamp": datetime.utcnow().isoformat()
    }
    self.registry[model_name] = metadata
    return metadata
```
- **Purpose**: Registers a model with its parameters and generates a version identifier.
- **Steps**:
  1. Serializes parameters (`params`) into JSON with sorted keys.
  2. Hashes the JSON string using SHA‑256.
  3. Truncates the hash to 8 characters for a compact version ID.
  4. Builds a metadata dictionary with:
     - Model name
     - Parameters
     - Version hash
     - UTC timestamp
  5. Stores metadata in the registry under the model name.
  6. Returns the metadata for immediate use/logging.

#### 4. `get_metadata()` Method
```python
def get_metadata(self, model_name):
    """Retrieve metadata for a registered model."""
    return self.registry.get(model_name, None)
```
- Retrieves metadata for a specific model by name.
- Returns `None` if the model hasn’t been registered.

#### 5. `list_registry()` Method
```python
def list_registry(self):
    """List all registered models and metadata."""
    return self.registry
```
- Returns the entire registry dictionary.
- Useful for auditing all models currently tracked.

### 🔑 Summary
- `traceability.py` is the **auditability module** of the app.
- It:
  1. Registers models with parameters and generates reproducible version hashes.
  2. Tracks metadata (model name, parameters, version, timestamp).
  3. Provides retrieval and listing functions for governance and reproducibility.
- Ensures forecasts can be traced back to the exact model configuration used.

## 3.18 `meta_learner.py`

### 📂 File Purpose
- **Role**: Implements the **meta learner (ensemble model)** that combines outputs from multiple base forecasting models.
- **Context**: Provides flexibility by supporting different aggregation methods (`ridge`, `logistic`, `boosting`).
- **Functionality**: Fits on base model predictions and produces ensemble forecasts, with fallbacks for robustness.

### 🧩 Key Components

#### 1. Imports
```python
import logging
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
```
- **logging**: Captures training and prediction events for traceability.
- **numpy**: Handles arrays and numerical operations.
- **Ridge**: Linear regression with L2 regularization.
- **LogisticRegression**: Classification model (can be adapted for binary forecast aggregation).
- **GradientBoostingRegressor**: Nonlinear ensemble method for regression tasks.

#### 2. Logger Setup
```python
logger = logging.getLogger(__name__)
```
- Creates a logger for this module.
- Ensures all fit/predict events are recorded.

#### 3. `MetaLearner` Class
```python
class MetaLearner:
    def __init__(self, method="ridge"):
        ...
```
- Initializes the meta learner with a chosen method.
- Supported methods:
  - `"ridge"` → Ridge regression.
  - `"logistic"` → Logistic regression.
  - `"boosting"` → Gradient boosting regressor.
- Raises `ValueError` if an unsupported method is passed.
- Tracks whether the model has been fitted (`self.is_fitted`).

#### 4. `fit()` Method
```python
def fit(self, X, y):
    ...
```
- Fits the meta learner on base model outputs.
- **Inputs**:
  - `X`: Shape `(n_samples, n_models)` → predictions from base models.
  - `y`: Shape `(n_samples,)` → target values (often averages of base predictions).
- **Steps**:
  1. Converts inputs to NumPy arrays.
  2. Logs shapes for traceability.
  3. Guard clause: skips training if inputs are empty.
  4. Handles mismatched sample sizes:
     - Logs error.
     - Recomputes `y` as row averages of `X`.
  5. Attempts to fit the chosen model.
  6. Logs success or failure, updates `self.is_fitted`.

#### 5. `predict()` Method
```python
def predict(self, X):
    ...
```
- Predicts ensemble forecasts using the fitted meta learner.
- **Inputs**:
  - `X`: Shape `(n_samples, n_models)` → predictions from base models.
- **Steps**:
  1. Converts input to NumPy array.
  2. Logs shape for traceability.
  3. If not fitted:
     - Logs warning.
     - Returns row averages of `X` as a fallback.
  4. If fitted:
     - Attempts to predict using the chosen model.
     - Logs success or failure.
     - On failure, falls back to row averages.

### 🔑 Summary
- `meta_learner.py` is the **ensemble aggregation module**.
- It:
  1. Supports multiple meta‑learning strategies (`ridge`, `logistic`, `boosting`).
  2. Fits on base model predictions and produces ensemble forecasts.
  3. Provides robust fallbacks (row averages) if fitting or prediction fails.
  4. Logs all operations for governance and debugging.
- Ensures forecasts are **transparent, auditable, and resilient** to input inconsistencies.

## 3.19 `huggingface_models.py`

### 📂 File Purpose
- **Role**: Provides a wrapper for **hosted model inference** via the Hugging Face API.
- **Context**: Allows the app to query Hugging Face models directly, using an API token for authentication.
- **Functionality**: Supports both single‑model queries and batch queries, with logging for governance and debugging.

### 🧩 Key Components

#### 1. Imports
```python
import requests
from governance.config import Config
from governance.logger import get_logger
```
- **requests**: Used to send HTTP POST requests to Hugging Face’s inference API.
- **Config**: Provides the Hugging Face API token (`HF_API_TOKEN`) and base URL (`HF_API_URL`).
- **get_logger**: Centralized logging utility for traceability.

#### 2. Logger Setup
```python
logger = get_logger(__name__)
```
- Creates a logger instance for this module.
- Ensures all inference calls and errors are logged.

#### 3. `query_hf_model()` Function
```python
def query_hf_model(model_name, payload):
    """Query Hugging Face hosted model."""
    headers = {"Authorization": f"Bearer {Config.HF_API_TOKEN}"}
    response = requests.post(Config.HF_API_URL + model_name, headers=headers, json=payload)
    if response.status_code == 200:
        logger.info(f"Model {model_name} inference successful.")
        return response.json()
    else:
        logger.error(f"Error {response.status_code} from {model_name}: {response.text}")
        return None
```

- **Purpose**: Sends a request to a Hugging Face model endpoint.
- **Steps**:
  1. Builds authorization headers using the API token.
  2. Sends a POST request to the Hugging Face API (`HF_API_URL + model_name`) with the given payload.
  3. On success (`200 OK`):
     - Logs success.
     - Returns the JSON response (model inference output).
  4. On failure:
     - Logs error with status code and message.
     - Returns `None`.

#### 4. `batch_query()` Function
```python
def batch_query(models, payload):
    """Query multiple Hugging Face models sequentially."""
    results = {}
    for model in models:
        results[model] = query_hf_model(model, payload)
    return results
```
- **Purpose**: Queries multiple Hugging Face models one after another.
- **Steps**:
  1. Iterates over a list of model names.
  2. Calls `query_hf_model()` for each model with the same payload.
  3. Collects results in a dictionary keyed by model name.
  4. Returns the dictionary of results.

### 🔑 Summary
- `huggingface_models.py` is the **integration module** for Hugging Face hosted models.
- It:
  1. Provides a function to query a single Hugging Face model (`query_hf_model`).
  2. Provides a function to query multiple models sequentially (`batch_query`).
  3. Logs all inference attempts for governance and debugging.
- Enables the Weather Aggregator to leverage external ML models hosted on Hugging Face for advanced forecasting or analysis.

## 3.20 `explainability.py`

### 📂 File Purpose
- **Role**: Provides **explainability utilities** for ensemble forecasts.
- **Context**: Uses **SHAP** and **LIME** to interpret model predictions, making forecasts more transparent and auditable.
- **Functionality**: Helps stakeholders understand how input features influence the ensemble’s output.

### 🧩 Key Components

#### 1. Imports
```python
import shap
from lime.lime_tabular import LimeTabularExplainer
```
- **SHAP**: A unified framework for interpreting predictions using Shapley values.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Explains individual predictions by approximating the model locally with interpretable models.

#### 2. `Explainability` Class
```python
class Explainability:
    def __init__(self, training_data, feature_names=None):
        self.training_data = training_data
        self.feature_names = feature_names if feature_names else [f"f{i}" for i in range(training_data.shape[1])]
```
- **Purpose**: Initializes explainability utilities with training data and optional feature names.
- **Parameters**:
  - `training_data`: Dataset used to fit SHAP/LIME explainers.
  - `feature_names`: Names of features; defaults to `f0, f1, f2, …` if not provided.
- Ensures both SHAP and LIME have consistent reference data.

#### 3. `shap_analysis()` Method
```python
def shap_analysis(self, model, sample):
    explainer = shap.Explainer(model, self.training_data)
    shap_values = explainer(sample)
    return shap_values
```
- **Purpose**: Runs SHAP analysis on a given model and sample.
- **Steps**:
  1. Creates a SHAP explainer using the model and training data.
  2. Computes SHAP values for the provided sample.
  3. Returns SHAP values (feature contributions to the prediction).
- **Use Case**: Explains how each feature influenced the ensemble forecast.

#### 4. `lime_analysis()` Method
```python
def lime_analysis(self, model, sample):
    explainer = LimeTabularExplainer(
        training_data=self.training_data,
        feature_names=self.feature_names,
        verbose=True,
        mode="regression"
    )
    explanation = explainer.explain_instance(sample, model.predict, num_features=5)
    return explanation
```
- **Purpose**: Runs LIME analysis on a given model and sample.
- **Steps**:
  1. Creates a LIME tabular explainer with training data and feature names.
  2. Configures it for regression mode (since forecasts are numeric).
  3. Explains the given sample using the model’s `predict` function.
  4. Returns an explanation object with feature importance for that instance.
- **Use Case**: Provides a local, human‑interpretable explanation of why the model made a specific forecast.

### 🔑 Summary
- `explainability.py` is the **interpretability module** of the app.
- It:
  1. Uses **SHAP** for global and local feature attribution.
  2. Uses **LIME** for local instance‑level explanations.
  3. Ensures forecasts are **transparent, auditable, and stakeholder‑friendly**.
- Bridges the gap between complex ensemble models and human understanding.

## 3.21 `base_models.py`

### 📂 File Purpose
- **Role**: Implements a suite of **classical and machine learning forecasting models**.
- **Context**: Provides diverse model outputs (SARIMAX, Kalman, RandomForest, GradientBoosting, CNN, LSTM, Autoencoder) that feed into the **meta learner** for ensemble aggregation.
- **Functionality**: Each model consumes normalized weather data (temperature series) and produces forecasts, with fallbacks for robustness.

### 🧩 Key Components

#### 1. Imports
```python
import logging
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
```
- **logging**: Captures warnings/errors for traceability.
- **numpy**: Handles numerical arrays.
- **statsmodels**: Provides SARIMAX (classical time series).
- **scikit‑learn**: Provides RandomForest and GradientBoosting regressors.
- **TensorFlow/Keras**: Provides neural network layers (Dense, LSTM, CNN, Autoencoder).

#### 2. Helper Function
```python
def _extract_series(weather_data, key="temperature_series"):
    ...
```
- Extracts numeric temperature series from `weather_data`.
- Fallbacks:
  - Uses `temperature_series` if available.
  - Falls back to single `temperature` value.
  - Returns empty array if no data.

#### 3. Classical Models
#### SARIMAX
```python
def sarimax_forecast(weather_data, steps=5):
    ...
```
- Fits a SARIMAX(1,1,1) model.
- Forecasts `steps` ahead.
- Fallback: returns average if insufficient data or error.

##### Kalman Filter (proxy)
```python
def kalman_filter_forecast(weather_data, steps=5):
    ...
```
- Simplified Kalman filter using rolling mean of last 3 values.
- Forecasts constant values for `steps`.

#### 4. Tree‑Based Models
#### Random Forest
```python
def random_forest_forecast(weather_data, steps=5):
    ...
```
- Fits a RandomForest regressor on index vs. series.
- Predicts future indices.
- Fallback: returns average if insufficient data or error.

##### Gradient Boosting
```python
def gradient_boosting_forecast(weather_data, steps=5):
    ...
```
- Fits a GradientBoosting regressor on index vs. series.
- Predicts future indices.
- Fallback: returns average if insufficient data or error.

#### 5. Neural Models (Stubs)
#### LSTM
```python
def lstm_forecast(weather_data, steps=5):
    ...
```
- Builds a simple LSTM network with 10 units.
- Trains briefly (5 epochs).
- Predicts last `steps` values.
- Fallback: returns average if insufficient data or error.

##### CNN
```python
def cnn_forecast(weather_data, steps=5):
    ...
```
- Builds a 1D CNN with Conv1D → Flatten → Dense.
- Trains briefly (5 epochs).
- Predicts last `steps` values.
- Fallback: returns average if insufficient data or error.

##### Autoencoder
```python
def autoencoder_forecast(weather_data, steps=5):
    ...
```
- Builds a simple autoencoder (Dense → Dense).
- Trains briefly (5 epochs).
- Predicts reconstruction of last `steps` values.
- Fallback: returns average if insufficient data or error.

### 🔑 Summary
- `base_models.py` is the **model zoo** of the app.
- It:
  1. Extracts numeric series from weather data.
  2. Runs multiple forecasting models (classical, tree‑based, neural).
  3. Provides robust fallbacks (averages) if data is insufficient or errors occur.
- Outputs from these models feed into the **meta learner** (`meta_learner.py`) for ensemble aggregation.


---

# 🧠 4. Run instructions

Here’s a simple and effective **runner** that launches our Weather Aggregator App. 
This is especially useful for testing, demos, or integrating app workflows into broader data science pipelines.

## 🚀 Startup
- **Terminal** → `python run.py`
- **Jupyter** → Paste startup cell at end of `01_flask_stub.ipynb`:
  ```python
  from threading import Thread
  from run import create_app
  app = create_app()
  Thread(target=lambda: app.run(port=5000, debug=True, use_reloader=False)).start()
  ```

## 🖥 Key Routes
- `/` → Input form (PO Box, City, Country)
- `/forecast/<city>/<country>` → Forecast dashboard
- `/report/<city>/<country>` → PDF report with charts + explainability

## 📊 Models
- **Classical** → SARIMAX, Kalman
- **ML** → Random Forest, Gradient Boosting
- **DL Stubs** → CNN, LSTM, Autoencoder
- **Hosted** → Hugging Face (~30 models)
- **Meta Learner** → Ridge, Logistic, Boosting aggregation

## 🔍 Explainability
- **SHAP** → Global feature importance
- **LIME** → Local interpretability
- **Confidence Bands** → Lower/upper bounds for forecasts

## 📂 Governance
- **Audit Logs** → SQLite entries for inputs + forecasts
- **Traceability** → Model versioning metadata
- **Reports** → PDF handouts with forecasts + governance notes

## 🧪 Testing
- Run all tests → `pytest tests/`
- Coverage:
  - `test_routes.py` → Flask routes
  - `test_models.py` → Base models + explainability
  - `test_data_store.py` → SQLite storage
  - `test_meta_learner.py` → Meta learner aggregation

## 🧩 Example Demo Flow
1. Input: `PO Box=12345, City=Frankfurt, Country=Germany`
2. System:
   - Fetch weather API
   - Run SARIMAX, Kalman, RF, Hugging Face
   - Aggregate with Ridge meta learner
   - Generate confidence bands
   - Log run in SQLite
3. Outputs:
   - Forecast dashboard
   - PDF report
   - Audit log entry


---
 

# 🔥 5. Architecture and Sequence Diagrams (Mermaid)

## 5.1 📊 Architecture Diagram (Mermaid)

Here’s a **modular architecture diagram** in **Mermaid syntax** that visually shows the flow between components in our Weather Aggregator project. 
This complements our README and makes the system narratable for executives and technical teams alike.

```mermaid
flowchart TD

    subgraph A["User Input"]
        A1["PO Box"]
        A2["City"]
        A3["Country"]
    end

    subgraph B["Frontend"]
        B1["Flask (routes.py)"]
    end

    subgraph C["Weather API"]
        C1["OpenWeatherMap"]
    end

    subgraph D["Normalized Data"]
        D1["Time Series"]
    end

    subgraph E["Base Models"]
        E1["SARIMAX"]
        E2["Random Forest"]
        E3["Kalman"]
        E4["LSTM"]
        E5["Autoencoder"]
    end

    subgraph F["Hosted Models"]
        F1["Hugging Face"]
    end

    subgraph G["Meta Learner"]
        G1["Ridge"]
        G2["Logistic"]
        G3["Boosting"]
    end

    subgraph H["Explainability"]
        H1["SHAP"]
        H2["LIME"]
    end

    subgraph I["Forecast Output"]
        I1["Forecasts"]
        I2["Confidence Bands"]
    end

    subgraph J["Reporting"]
        J1["PDF Report"]
        J2["Dashboard"]
    end

    subgraph K["Governance"]
        K1["Logger"]
        K2["Audit Logs"]
    end

    A3 --> B
    B --> C
    C --> D
    D --> E
    D --> F
    E --> G
    F --> G
    G --> H
    G --> I
    I --> J
    J1 --> K
    C --> K
```


### 🧩 How to Read This Diagram
- **User Input → Flask Frontend**: Entry point for PO Box, City, Country.
- **Flask → Storage + Weather API**: Inputs logged, raw weather data retrieved.
- **Data → Base + Hosted Models**: Classical + Hugging Face models generate forecasts.
- **Meta Learner**: Aggregates ~30 model outputs into one ensemble forecast.
- **Explainability**: SHAP/LIME interpretability for transparency.
- **Outputs**: Forecasts shown in dashboard and exported as PDF reports.
- **Governance Layer**: Logs every action for auditability and reproducibility.

This diagram makes the **system modular, transparent, and narratable** — exactly what we need for consulting demos. 

---

## 5.2 📊 Sequence Diagram (Mermaid)

Below is the **sequence diagram** that shows the step‑by‑step interaction flow in our Weather Aggregator system. This 
complements the architecture diagram by narrating the runtime process from user input to final report.

```mermaid
sequenceDiagram
    participant U as User
    participant F as Flask Frontend (routes.py)
    participant DS as DataStore (SQLite)
    participant WA as Weather API
    participant BM as Base Models (SARIMAX, RF, Kalman, LSTM)
    participant HF as Hugging Face Models (~30 hosted)
    participant ML as Meta Learner (ridge/logistic/boosting)
    participant EX as Explainability (SHAP/LIME)
    participant PDF as PDF Report
    participant DB as Dashboard

    U->>F: Submit PO Box, City, Country
    F->>DS: Save user input
    F->>WA: Request raw weather data
    WA-->>F: Return normalized time series

    F->>BM: Run classical + ML models
    F->>HF: Query hosted Hugging Face models
    BM-->>F: Forecast outputs
    HF-->>F: Forecast outputs

    F->>ML: Aggregate forecasts
    ML-->>F: Ensemble forecast + confidence bands

    F->>EX: Run SHAP/LIME analysis
    EX-->>F: Explainability results

    F->>DS: Save forecasts + audit logs
    F->>PDF: Generate governance-ready report
    F->>DB: Render dashboard with forecasts

    DB-->>U: Display weather card + ensemble forecast
    PDF-->>U: Provide downloadable PDF report
```

### 🧩 How to Read This Diagram
- **User → Flask**: Input submission triggers the workflow.
- **Flask → DataStore + Weather API**: Inputs logged, raw data retrieved.
- **Models**: Base + hosted models generate forecasts.
- **Meta Learner**: Aggregates outputs into ensemble forecast.
- **Explainability**: SHAP/LIME adds transparency.
- **Outputs**: Forecasts stored, reports generated, dashboard updated.
- **User**: Receives both interactive dashboard and governance-ready PDF.

This sequence diagram makes the **runtime workflow narratable** for executives and technical teams alike.    

---

# 6. 🔗 Results and conclusions

## 6.1 📊 Start the Weather Aggregator App

### ✅ Step 1: Download the folder

Download the main folder
📁 [Weather_Aggregator_Flask_App](https://github.com/NenadBalaneskovic/ExternalProjects/tree/ae6755f2c2611f3ceb88d452f12f3d3489a623fe/Weather_Aggregator_FlaskApp)
 which has the following structure:  
 
   <img src="https://github.com/NenadBalaneskovic/ExternalProjects/blob/4f25734f8a76f4f2125772e9e627d96a8e8c6c49/Weather_Aggregator_FlaskApp/MainFolder_structure.PNG" width="400" height="200"/>

### ✅ Step 2: Run the jupyter runner

#### 1. **Directly call `run.py` from a notebook cell**
Since we already have `run.py` as the entry point:

```python
!python run.py
```

- This launches the Flask development server exactly as if you ran it from the terminal.
- We will see logs in the notebook output.
- We open `http://127.0.0.1:5000` in our browser to interact with the app.
- ⚠️ The notebook will block until weu stop the server (Ctrl+C in the kernel).

#### 2. **Run Flask in a background thread (non-blocking)**
If we want to keep Jupyter interactive while the app runs:

```python
from threading import Thread
from run import create_app

app = create_app()

def run_flask():
    app.run(port=5000, debug=True, use_reloader=False)

thread = Thread(target=run_flask)
thread.start()
```

- This starts Flask in a background thread.
- We can continue running other notebook cells while the app is live.
- Visit `http://127.0.0.1:5000` in ther browser.

#### 3. **Use `%run` magic for inline execution**
```python
%run run.py
```
- Similar to `!python run.py`, but integrates with Jupyter’s execution model.
- Useful if we want inline logs and tighter notebook control.

#### 4. ✅ Best Practice for Our Structure
- Because we have a **factory pattern** (`create_app()` in `app/__init__.py`), the **background thread approach (Option 2)** is the most flexible. 
It lets us import and run the app inside Jupyter without blocking, while still respecting your modular structure (`routes.py`, `forms.py`, etc.).
- For production‑like testing, we stick with **Option 1** (`!python run.py`) so we are running the app exactly as intended.

### ✅ Step 3: Interact with the Weather Aggregator App

Interact with the Weather Aggregator App by providing reasonable inputs in the left half-plane and pressing the "__Get Forecast__"-button.

## 6.2 🧠 Interpretation of results

### 🧠 This Weather Aggregator Flask App...

- Accepts user inputs:
  - PO Box
  - City
  - Country
  
- Displays weather forecasting results associated with different ML models for a 3-day period (including the corresponding overall confidence intervals) within the app-plane

- Stores obtained results and characterizations as pdf file (see figure below)

![Weather_Forecast_Results_csv](https://github.com/NenadBalaneskovic/ExternalProjects/blob/1e13811b55cd23fe8a7e1092107746b31d28bf0d/Weather_Aggregator_FlaskApp/WeatherForecast_pdfReport.PNG)

- Automatically displays the forecast history (see figure below).

![Weather_Forecast_Flask_App_Functionality](https://github.com/NenadBalaneskovic/ExternalProjects/blob/e6263c8a1be70e3d0f294ab777a8d7c449a2a87e/Invoice_TaxTracker_GUI/GUI_complete_results.png)

## 6.3 🏁 Final Thoughts

The *Weather Aggregator Flask App* demonstrates how ensemble learning can be operationalized in a transparent, modular, and consulting-ready environment. By integrating diverse base models 
with stacked surrogate meta-learners, the project provides a rigorous testbed for evaluating predictive performance while foregrounding explainability and governance. Weather forecasting, 
with its inherent complexity and uncertainty, serves as an ideal domain for this exploration, but the lessons extend far beyond meteorology. The architecture illustrates how organizations can 
harness ensemble methods responsibly, ensuring that outputs are not only accurate but also narratable, auditable, and reproducible.

A key achievement of the project lies in its ability to characterize ensemble results rather than simply produce them. By storing forecasts alongside confidence bands, user inputs, and metadata, 
the app creates a traceable record that supports both methodological analysis and executive reporting. This dual focus—technical robustness and communicative clarity—positions the system as a bridge 
between research prototypes and enterprise solutions. It shows that ensemble learning need not remain a black box; with careful design, surrogate models can provide insight into how predictions are 
formed, which base models contribute most, and how uncertainty is quantified.

The project also underscores the value of modularity. Each component—data ingestion, base forecasting models, meta-learner, datastore, and reporting interface—was designed to be replaceable and extensible. 
This ensures that the app can evolve as new models or requirements emerge, while maintaining a consistent framework for evaluation and reporting. Such modularity is critical in consulting contexts, where 
solutions must adapt to diverse client needs without sacrificing reproducibility or governance compliance.

From a strategic perspective, the *Weather Aggregator Flask App* highlights how lightweight web applications can democratize access to advanced analytics. By embedding ensemble learning in a Flask interface, 
the project makes complex methodologies accessible to non-technical stakeholders, enabling them to interact with forecasts, view confidence bands, and download reports. This accessibility fosters trust and 
engagement, ensuring that machine learning outputs are not confined to technical teams but can inform decision‑making across an organization.

Ultimately, the project achieves its aim of testing stacked surrogate models and characterizing the results of different ensemble learning strategies, including both linear regression models (LRM) and large 
language models (LLM) adapted for surrogate tasks. In doing so, it contributes to the broader discourse on explainable AI, demonstrating that predictive power and transparency can coexist. 
The *Weather Aggregator Flask App* is not merely a forecasting tool; it is a narrative asset that illustrates best practices in ensemble modeling, surrogate characterization, and consulting-oriented system design. 
As such, it provides a foundation for future work in applying explainability-first ensemble learning to other domains, reinforcing the principle that innovation must be accompanied by accountability and clarity.

---

# 7. 🧾 Weather Aggregator Flask App  
### 🛠️ User Manual

Here’s a **comprehensive, example-driven user manual** for our [Weather Aggregator app](https://github.com/NenadBalaneskovic/ExternalProjects/blob/55d3a20e095ec7c17419f3b614faf9b8d3dab837/Weather_Aggregator_FlaskApp/weather_aggregator/README.md).

## 🌦 Weather Aggregator User Manual

### 📖 Introduction
The Weather Aggregator is a modular, explainability‑first forecasting system. It combines classical models (SARIMAX, Kalman), 
machine learning (Random Forest, Gradient Boosting), deep learning stubs (CNN, LSTM, Autoencoder), and hosted Hugging Face inference into a transparent, auditable ensemble.  

This manual guides you through installation, usage, reporting, and governance features with practical examples.

### ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/weather-aggregator.git
   cd weather-aggregator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   pytest tests/
   ```
   You should see all unit tests passing.

### ▶️ Running the App

#### Option A: Terminal
```bash
python run.py
```
Visit `http://127.0.0.1:5000`.

#### Option B: Jupyter Notebook (interactive)
Paste this cell at the end of `01_flask_stub.ipynb`:

```python
from threading import Thread
from run import create_app

app = create_app()

def run_flask():
    app.run(port=5000, debug=True, use_reloader=False)

thread = Thread(target=run_flask)
thread.start()

print("✅ Weather Aggregator app is running at http://127.0.0.1:5000")
```

### 🖥 User Interface

#### 1. **Home Page (`/`)**
- Input form for **PO Box, City, Country**.
- Example:
  ```
  PO Box: 12345
  City: Frankfurt
  Country: Germany
  ```
- Click **Submit** → Redirects to forecast dashboard.

#### 2. **Forecast Page (`/forecast/<city>/<country>`)**
- Displays:
  - Current weather card (from OpenWeatherMap).
  - Forecasts from SARIMAX, Kalman, Random Forest, Hugging Face models.
  - Confidence bands (lower/upper bounds).
  - Ensemble forecast (meta learner).

**Example Output:**
```
Forecast for Frankfurt, Germany
SARIMAX: [22.5, 23.0, 21.8]
Kalman: [22.3, 22.4, 22.5]
Random Forest: [22.0, 22.7, 23.1]
Ensemble (Ridge): [22.3, 22.7, 22.5]
Confidence Band: (21.0, 24.0)
```

#### 3. **Report Page (`/report/<city>/<country>`)**
- Generates a **PDF report** with:
  - Forecast comparison chart.
  - Confidence bands.
  - Explainability summary (SHAP/LIME).
  - Metadata (model versions, run timestamp).

**Example PDF Sections:**
- *Forecast Comparison*: Line chart of SARIMAX vs Random Forest vs Ensemble.
- *Explainability*: SHAP values showing feature importance.
- *Governance*: Audit log entry with model parameters.

### 🔍 Explainability & Governance

- **Explainability**
  - SHAP analysis: highlights which features drive forecasts.
  - LIME analysis: local interpretability for individual predictions.

- **Governance**
  - Audit logs: stored in SQLite (`data_store.py`).
  - Traceability: model versioning in `governance/traceability.py`.
  - Reports: PDF handouts for executive review.

### 🧪 Testing

Run all unit tests:
```bash
pytest tests/
```

- **`test_routes.py`** → Validates Flask routes.
- **`test_models.py`** → Ensures base models and explainability utilities.
- **`test_data_store.py`** → Confirms SQLite wrapper saves/retrieves forecasts.
- **`test_meta_learner.py`** → Tests meta learner aggregation and confidence bands.

### 📊 Example Workflow

1. **User Input**
   - PO Box: `12345`
   - City: `Frankfurt`
   - Country: `Germany`

2. **System Actions**
   - Fetch current weather from API.
   - Run SARIMAX, Kalman, Random Forest, Hugging Face models.
   - Aggregate with Ridge meta learner.
   - Generate confidence bands.
   - Log run in SQLite.

3. **Outputs**
   - Forecast dashboard with charts.
   - PDF report with explainability.
   - Audit log entry.

### 🧩 Tips
- Use **Jupyter notebooks (`02_model_dev.ipynb`, `03_huggingface_api.ipynb`, `04_dashboard.ipynb`)** for prototyping models and dashboards.
- Switch between **dev/test/prod configs** by extending `app/__init__.py` with environment‑specific settings.
- For consulting demos, emphasize **ensemble transparency** and **governance modules**.

Let’s map out the **LLM/LRM (Large Language / Large Regression Models)** that our Weather Aggregator app is designed to use, based on the scaffolds we’ve built:

### 📊 Core Modeling Layer

#### 🔢 Classical / Regression Models
- **SARIMAX** → Seasonal ARIMA with exogenous variables for time‑series forecasting.
- **Kalman Filter** → State‑space model for smoothing and short‑term predictions.
- **Ridge / Logistic Regression** → Used in the **meta learner** to aggregate base forecasts.
- **Gradient Boosting / Random Forest** → Tree‑based regressors for nonlinear relationships.

#### 🤖 Deep Learning Stubs
- **CNN (Convolutional Neural Network)** → For spatial weather data (e.g., gridded forecasts).
- **LSTM (Long Short‑Term Memory)** → For sequential weather/time‑series data.
- **Autoencoder** → For dimensionality reduction and anomaly detection in weather patterns.

#### 🌐 Hosted LLM Integration
- **Hugging Face Models** (via API):
  - General text‑generation models (e.g., GPT‑2, DistilGPT‑2) for narrative forecasts.
  - Domain‑specific hosted models (weather/forecast fine‑tuned LLMs, if available).
  - These are queried through `huggingface_models.py` and wrapped with explainability stubs.

#### 🧩 Meta Learner (LRM)
- Aggregates outputs from SARIMAX, Kalman, Random Forest, CNN/LSTM, and Hugging Face models.
- Uses **Ridge regression** or **boosting** to produce a single ensemble forecast.
- Confidence bands are generated for governance and explainability.

### ✅ Summary
Your app uses a **hybrid stack**:
- **LRMs** → SARIMAX, Kalman, Ridge, Random Forest, Gradient Boosting.  
- **LLMs** → Hugging Face hosted models for narrative/weather text generation.  
- **DL stubs** → CNN, LSTM, Autoencoder for advanced prototyping.  
- **Meta learner** → Ridge/boosting aggregator to unify all forecasts.

This combination ensures **robustness (LRM)**, **expressiveness (LLM)**, and **transparency (explainability + governance)** — exactly aligned with our consulting‑ready, narratable demo goals.  

### App Flow Chart

![AppFlowChart.png](https://github.com/NenadBalaneskovic/ExternalProjects/blob/eeb866299c31931f4f6dc154858ef2ba56bcf17c/Weather_Aggregator_FlaskApp/AppFlowChart.png)  

Here’s a detailed walkthrough of the above architecture diagram for our Weather Aggregator Flask Application:

### 🧭 Overall Layout
The diagram is structured as a **modular flowchart**, showing how different components interact from **user input** to **forecast output**, with governance and explainability integrated throughout.

### 🧩 Sections and Flow

#### 1. **Weather Aggregator Flask Application (Top Center)**
- This is the central orchestrator.
- It connects all modules: input, models, explainability, governance, and output.

#### 2. **Input Form (Far Right)**
- Accepts:
  - **PO Box**
  - **City**
  - **Country**
- This form triggers the forecasting pipeline.

#### 3. **Forecast Dashboard (Below Input Form)**
- Displays:
  - **Forecasts** from all models
  - **Confidence Bands**
  - **Ensemble Output**
- This is the user-facing visualization layer.

#### 4. **Model Integration Layer (Center)**
- Combines:
  - **Python models**
  - **Hugging Face API models**
    - Examples: GPT-2, DistilGPT-2, BERT, RoBERTa
- Acts as a bridge between hosted inference and local forecasting.

#### 5. **Forecasting Models (Below Integration Layer)**
- Includes:
  - **SARIMAX** → classical time-series
  - **Kalman Filter** → smoothing and short-term prediction
  - **Random Forest** → tree-based regression
  - **CNN, LSTM (stubs)** → deep learning prototypes

#### 6. **Ensemble Explainability (Far Left)**
- Tools:
  - **SHAP** → global feature importance
  - **LIME** → local interpretability
- These modules narrate how each model contributes to the ensemble.

#### 7. **Governance (Far Left, above Explainability)**
- Modules:
  - **Audit Logs** → tracks inputs, outputs, timestamps
  - **Versioning** → stores model versions and parameters
  - **PDF Reports** → generates executive-ready summaries

### 🔄 Arrows and Interactions
- **Input Form → Forecasting** → triggers model execution.
- **Forecasting → Dashboard** → sends results for visualization.
- **Forecasting → Explainability** → feeds SHAP/LIME analysis.
- **Forecasting → Governance** → logs metadata and generates reports.
- **Explainability ↔ Governance** → bidirectional flow for traceable reporting.

### ✅ What This Diagram Communicates
- **Modularity** → Each layer is independently testable and narratable.
- **Transparency** → Explainability and governance are first-class citizens.
- **Consulting Readiness** → Designed for demos, executive reporting, and auditability.

## Troubleshooting 1: ✅ How to Fix the API Key Issue

1. **Get a valid API key**
   - Sign in to [OpenWeatherMap](https://home.openweathermap.org/users/sign_up).
   - Go to your profile → “API keys”.
   - Copy your personal key (looks like a long alphanumeric string).

2. **Update your app configuration**
   - In your project, you probably have something like `data/weather_api.py` using:
     ```python
     API_KEY = "your_api_key_here"
     ```
   - Replace that with your real key, or better:
     - Store it in an environment variable:
       ```bash
       setx OPENWEATHER_API_KEY "your_real_key"
       ```
     - Then load it in Python:
       ```python
       import os
       API_KEY = os.getenv("OPENWEATHER_API_KEY")
       ```

3. **Verify the API call works**
   - Test directly in your terminal:
     ```bash
     curl "http://api.openweathermap.org/data/2.5/weather?q=Frankfurt,de&appid=YOUR_KEY&units=metric"
     ```
   - You should see JSON with temperature, condition, etc.

4. **Add error handling in your app**
   - In `weather_api.py`, catch the 401 and flash a user‑friendly message:
     ```python
     if response.status_code == 401:
         logger.error("Invalid API key")
         return None
     ```

## 🧪 Debugging Checklist
- Confirm the API key is valid and active.
- Ensure you’re passing the key in the request URL (`appid=YOUR_KEY`).
- Double‑check you’re not mixing test vs production keys.
- Restart the app after updating the environment variable.

👉 Once the API key is valid, our app will fetch real weather data, the feature arrays will be non‑empty, and the RandomForest will stop complaining.  

**Signing up for OpenWeatherMap is free.** You can create an account at no cost, and once registered you’ll receive a free API key that allows you to make weather data requests. 
The free tier includes access to current weather, forecasts, and some limited historical data, with a daily quota of API calls.

## 📖 Details on OpenWeatherMap Access

- **Free Account**
  - Registration is free at [OpenWeatherMap sign‑up](https://home.openweathermap.org/users/sign_up).
  - You get a personal API key (APPID) sent to your email.
  - The free plan allows **up to 1,000 API calls per day** at no cost.
  - Covers basic endpoints: current weather, 48‑hour hourly forecast, 8‑day daily forecast.

- **Paid Plans**
  - If you need more than 1,000 calls/day or advanced features (like long‑term forecasts, alerts, or 46+ years of historical data), you can upgrade.
  - Pricing starts at **$40/month for the Startup plan**, with higher tiers for Developer, Professional, Expert, and Enterprise.

- **Usage Notes**
  - Free API keys sometimes take up to 2 hours to activate after registration.
  - Always include your API key in requests:  
    ```
    http://api.openweathermap.org/data/2.5/weather?q=Frankfurt,de&appid=YOUR_KEY&units=metric
    ```

## ✅ What This Means for Our App
- You can run your Weather Aggregator app on the **free tier** without paying anything.
- For consulting demos or light usage, the free quota is usually sufficient.
- If you plan to scale (e.g., many users or continuous polling), you’ll want to consider a paid plan.  

## Troubleshoot 2: Hugging Face Access Tokens  

**You obtain your Hugging Face token by creating an account on Hugging Face and generating an access token in your account settings.** It’s free to sign up, 
and the token is what authenticates your app when calling Hugging Face APIs.  

## 🛠 Step‑by‑Step Guide

### 1. Create a Hugging Face account
- Go to [huggingface.co](https://huggingface.co).
- Click **Sign Up** (top right).
- Register using your email, GitHub, or Google account.
- Verify your email address (this step is required before tokens can be created).

### 2. Navigate to Access Tokens
- Once logged in, click your **profile picture** (top right).
- Select **Settings**.
- In the left sidebar, choose **Access Tokens**.

### 3. Generate a new token
- Click **New Token**.
- Give it a descriptive name (e.g., `weather_app`).
- Choose a **role/scope**:
  - **Read** → allows downloading models/datasets (most common for inference).
  - **Write** → allows uploading models/datasets.
  - **Admin** → full access (use cautiously).
- Click **Generate**.

### 4. Copy and store the token securely
- The token will look like:  
  ```
  hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  ```
- Copy it and store it in a safe place.
- You’ll use it in your app like:
  ```python
  import os
  from huggingface_hub import InferenceApi

  HF_TOKEN = os.getenv("HF_TOKEN")  # or paste directly
  api = InferenceApi(repo_id="bert-base-uncased", token=HF_TOKEN)
  ```

### 5. Best practices
- **Do not hardcode** the token in public repos.
- Use environment variables (`HF_TOKEN`) or a `.env` file.
- Rotate tokens periodically for security.

## 📌 Sources
You can find detailed instructions on Hugging Face’s official documentation: [User access tokens](https://huggingface.co/docs/hub/security-tokens) and
a step-by-step guide on [GeeksforGeeks](https://www.geeksforgeeks.org/artificial-intelligence/how-to-access-huggingface-api-key/).

✅ With this manual, we can install, run, and demonstrate the Weather Aggregator app in both technical and executive contexts.  

---

# 8. 📚 References
1. **Chip Huyen**, *AI Engineering: Building Applications with Foundation Models*, 1st Edition, O’Reilly Media, 2025; **Michael Lanham**, *AI Agents in Action*, 1st Edition, Manning Publications, 2025;
 **Melanie Mitchell**, *Artificial Intelligence: A Guide for Thinking Humans*, 1st Edition, Pelican Books, 2019; **Brian Christian & Tom Griffiths**, *Algorithms to Live By: The Computer Science of Human Decisions*, 1st Edition, Henry Holt and Company, 2016;
**Ray Kurzweil**, *The Singularity Is Nearer: When We Merge with AI*, 1st Edition, Viking, 2024; OpenWeatherMap: https://openweathermap.org/, HuggingFace: https://huggingface.co/, 
LLM vs LRM: https://www.aryaxai.com/article/llm-vs-lrm-vs-lam-understanding-the-future-of-language-based-ai-systems, https://magazine.sebastianraschka.com/p/understanding-reasoning-llms
2. [![Jupyter Notebook | English](https://img.shields.io/badge/Jupyter%20Notebook-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/9b64196b88f00af6bd0ad1e1971374884d45bdcd/Weather_Aggregator_FlaskApp/Flask_Weather_App.ipynb)
3. [![TaxInvoiceTracker Report | English](https://img.shields.io/badge/TaxInvoiceTracker%20Report-English-yellowblue?logoColor=blue&labelColor=red)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/3afad96973578ad9935d2c5ba8819c76291f2548/Invoice_TaxTracker_GUI/InvoiceTaxTracker_Report.pdf) 
4. A. Meister , T. Sonar: "__Numerik__", 1st Ed. Springer-Spektrum (2019); S. Chapra, R. Canale: "__Numerical Methods for Engineers__", Mcgraw-Hill, 6th Edition (2010). 
5. J. Kilty, A. M. McAllister: "__Mathematical Modeling and Applied Calculus__", 1st Ed. Oxford University Press (2018).
6. U. Kockelkorn: "__Statistik für Anwender__", 1st Ed. Springer (2012), s. chapters 7 - 8.
7. Robert H. Shumway, David S. Stoffer: "__Time Series Analysis and Its Applications with R Examples__", Springer (2011).
8. Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor: "__An Introduction to Statistical Learning with Applications in Python__", Springer (2023).
9. Cornelis W. Oosterlee, Lech A. Grzelak: "__Mathematical Modeling and Computation in Finance with Exercises and Python and MATLAB Computer Codes__", World Scientific (2020).
10. Richard Szeliski: "__Computer Vision - Algorithms and Applications__", Springer (2022).
11. Anthony Scopatz, Kathryn D. Huff: "__Effective Computation in Physics - Field Guide to Research with Python__", O'Reilly Media (2015).
12. Alex Gezerlis: "__Numerical Methods in Physics with Python__", Cambridge University Press (2020).
13. Gary Hutson, Matt Jackson: "__Graph Data Modeling in Python. A practical guide__", Packt-Publishing (2023).
14. Hagen Kleinert: "__Path Integrals in Quantum Mechanics, Statistics, Polymer Physics, and Financial Markets__", 5th Edition, World Scientific Publishing Company (2009).
15. Peter Richmond, Jurgen Mimkes, Stefan Hutzler: "__Econophysics and Physical Economics__", Oxford University Press (2013).
16. A. Coryn , L. Bailer Jones: "__Practical Bayesian Inference A Primer for Physical Scientists__", Cambridge University Press (2017).
17. Avram Sidi: "__Practical Extrapolation Methods - Theory and Applications__", Cambridge university Press (2003).
18. Volker Ziemann: "__Physics and Finance__", Springer (2021).
19. Zhi-Hua Zhou: "__Ensemble methods, foundations and algorithms__", CRC Press (2012).
20. B. S. Everitt, et al.: "__Cluster analysis__", Wiley (2011).
21. Lior Rokach, Oded Maimon: "__Data Mining With Decision Trees - Theory and Applications__", World Scientific (2015).
22. Bernhard Schölkopf, Alexander J. Smola: "__Learning with kernels - support vector machines, regularization, optimization and beyond__", MIT Press (2009).
23. Johan A. K. Suykens: "__Regularization, Optimization, Kernels, and Support Vector Machines__", CRC Press (2014).
24. Sarah Depaoli: "__Bayesian Structural Equation Modeling__", Guilford Press (2021).
25. Rex B. Kline: "__Principles and Practice of Structural Equation Modeling__", Guilford Press (2023).
26. Ekaterina Kochmar: "__Getting Started with Natural Language Processing__", Manning (2022).
27. Jakub Langr, Vladimir Bok: "__GANs in Action__", Computer Vision Lead at Founders Factory (2019).
28. David Foster: "__Generative Deep Learning__", O'Reilly(2023).
29. Rowel Atienza: "__Advanced Deep Learning with Keras: Applying GANs and other new deep learning algorithms to the real world__", Packt Publishing (2018).
30. Josh Kalin: "__Generative Adversarial Networks Cookbook__", Packt Publishing (2018).  
31. Thomas Haslwanter: "__Hands-on Signal Analysis with Python: An Introduction__", Springer (2021).
32. Jose Unpingco: "__Python for Signal Processing__", Springer (2023).
33. R. K. Burdick, C. M. Borror, D. C. Montgomery: "__Design and Analysis of Gauge R&R Studies__", 1st Ed. SIAM (2005); 
S. H. Derakhshan , C. V. Deutsch: "__Numerical Integration of Bivariate Gaussian Distribution__", Paper 405, CCG Anual Report 13 (2011).
34. C. Paar, J. Pelzl: "__Understanding Cryptography__", Springer (2010); H. Delfs, H. Knebl: "__Introduction to Cryptography__", 3rd Ed. Springer (2015); J. Katz, Y. lindell: "__Introduction to Modern Cryptography__", 2nd Ed, CRC Press (2015); 
O. Goldreich: "__Foundations of Cryptography__", Cambridge University Press (2008); J. P. Aumasson: "__Serious Cryptography__", no starch press (2018).  
35. J. Berk, P. DeMarzo: „__Corporate Finance__“, 6th Ed., Pearson (2023); R. W. Melicher, E. A. Norton: "__Introduction to Finance__", 16th Ed. WILEY (2017); 
Anatoly B. Schmidt: "__Quantitative Finance for Physicists: An Introduction__", 1st Ed. Academic Press (2005); Alex Backwell: "__An Intuitive Introduction to Finance and Derivatives: Concepts, Terminology and Models__",
 1st Ed, Springer (2023); Michael Isichenko: "__Quantitative Portfolio Management: The Art and Science of Statistical Arbitrage__", 1st Ed., Springer (2021); John H. Cochrane: "__Asset Pricing__", Revised Ed., Princeton University Press (2005);
 Antti Ilmanen: "__Expected Returns: An Investor’s Guide to Harvesting Market Rewards__", 1st Ed., WILEY (2011); Steven E. Shreve: "__Stochastic Calculus for Finance I & II__", 1st Ed., Springer (2004); 
 Andrew Pole: "__Statistical Arbitrage: Algorithmic Trading Insights and Techniques__", 1st Ed., WILEY (2007); Mark S. Joshi: "__The Concepts and Practice of Mathematical Finance__", 2nd Ed., Cambridge University Press (2008);
Kaggle-link: competition-documentation: https://www.kaggle.com/competitions/drw-crypto-market-prediction.
36. R. Nystrom: "__Game Programming Patterns__", 1st Ed. genever benning (2014); A. A. Stepanov, D. E. Rose: "__From Mathematics to Generic Programming__", 1st Ed. Addison-Wesley (2015);
37. E. Parzen: "__Stochastic Processes__", 3rd Ed. Dover Publications (2015); S. Aloorravi: "__Metaprogramming with Python__", 1st Ed. Packt (2022); B. Klein, P. Klein: "__Funktionale Programmierung mit Python__", Hanser (2025);
K. Webel, D. Wied: "__Stochastische Prozesse__", 2. Auflage Springer (2016); L. Held: "__Methoden der statistischen Inferenz__", 1. Auflage Spektrum (2008); E. Cinlar: "__Stochastic Processes__", Dover (2013);
N. Bäuerle, U. Rieder: "__Finanzmathematik in diskreter Zeit__", Springer-Spektrum (2017); M. Albrecht, R. Maurer: "__Investment- und Risikomanagement__", 3. Auflage, Schäffer Poeschel (2008);
N. H. Bingham, R. Kiesel: "__Risk Neutral Valuation: Pricing and Hedging of Financial Derivatives__", 2. Auflage Springer (2004); T. Björk: "__Arbitrage Theory in Continuous Time__", 3rd Ed. Oxford University Press (2009);
N. J. Cutland, A. Roux: "__Derivative Pricing in Discrete Time__", Springer (2013); F. Delbaen, W. Schachermayer: "__The Mathematics of Arbitrage__", Springer (2006); 
R. J. Elliott, P. E. Kopp: "__Mathematics of Financial Markets__", 2nd Ed. Springer (2005); H. Föllmer, A. Scheid: "__A Stochastic Finance: An Introduction in Discrete Time__", 3rd Ed. de Gruyter (2011);
J. C. Hull: "__Options, Futures and Other Derivatives__", 8th Ed. Pearson (2011); J. Kremer: "__Einführung in die diskrete Finanzmathematik__", Springer (2005); 
D. Lamberton, B. Lapeyre: "__Introduction to Stochastic Calculus Applied to Finance__", Chapman & Hall (2007); D. G. Luenberger: "__Investment Science__", Oxford University Press (1998);
S. R. Pliska: "__Introduction to Mathematical Finance: Discrete Time Models__", Blackwell (2000); A. N. Shiryaev: "__Essentials of Stochastic Finance__", World Scientific (2001);
S. E. Shreve: "__Stochastic Calculus for Finance I: The Binomial Asset Pricing Model__", Springer (2004); J. Kremer: "__Portfoliotheorie, Risikomanagement und die Bewertung von Derivaten__", Springer (2011);
L. Rüschendorf: "__Mathematical Risk Analysis__", Springer (2013). 
38. A. Becker: "__Kalman Filter - From the Ground Up__", 1st Ed. private publication (2023); K. Triantafyllopoulos: "__Bayesian Inference of State Space Models__", 1st Ed. Springer (2021); 
P. Zarchan, H. Musoff: "__Fundamentals of Kalman Filtering: A Practical Approach__", 
3rd Ed. AIAA (2009); A. Sidi: "__Vector Extrapolation Methods with Applications__", 1st Ed. SIAM (2019); C. Brezinski, M. R. Zaglia: "__Extrapolation Methods - Theory and Practice__", 2nd Ed. North-Holland (2002); 
C. Gardiner, P. Zoller: "__Quantum Noise: A Handbook of Markovian and Non-Markovian Quantum Stochastic Methods with Applications to Quantum Optics__", 3rd Ed. Springer (2004); 
K. Kendre: "__Machine Learning for Quantum Noise Reduction__", https://arxiv.org/abs/2509.16242 (2025); D. C. Marinescu, G. M. Marinescu: "__Classical and Quantum Information__", 1sr Ed. Academic Press (2012); 
Liao, H et al.: "__Machine Learning for Practical Quantum Error Mitigation__", arXiv:2309.17368v2 (2024), https://arxiv.org/pdf/2309.17368; Streamlit: https://streamlit.io/; 
Mitiq-package: https://quantum-journal.org/papers/q-2022-08-11-774/, https://arxiv.org/abs/2009.04417; Extrapolation packages: https://pypi.org/project/extrapolation/  
39. A. Koop, H. Moock: "__Lineare Optimierung - Eine anwendungsorientierte Einführung in Operations Research__", 1st Ed. Spektrum (2008); 
G, B, Dantzig, M. N. Thalpa: "__Linear Programming 1: Introduction__", 1st Ed. Springer (1997) & "__Linear Programming 2: Theory and Extensions__", 1st Ed. Springer (2003); 
H. S. Kasana, K. D. Kumar: "__Introductory Operations Research, Theory and Applications__", 1st Ed. Springer (2004); D. G. Luenberger: "__Linear and Nonlinear Programming__", 2nd Ed. Kluwer (2004); 
R. J. Boucherie, A. Braaksma, H. Tijms: "__Operations Research - Introduction to Models and Methods__", 1st Ed. World Scientific (2022); 
A. J. King, S. W. Wallace: "__Modeling with Stochastic Programming__", 2nd Ed. Springer (2024); 
J. O. Royset, R. J.-B. Wets: "__An Optimization Primer__", 1st Ed. Springer (2021); cvxpy package: https://www.cvxpy.org/, https://pypi.org/project/cvxpy/;
py-packages for operations research: https://wiki.python.org/moin/PythonForOperationsResearch 
40. (Py-)tesseract package: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract), https://pypi.org/project/pytesseract/,
https://builtin.com/data-science/python-ocr, https://www.analyticsvidhya.com/blog/2024/04/ocr-libraries-in-python/ and [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki).
































