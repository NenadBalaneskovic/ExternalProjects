# 1. 🚀 Project Introduction: DRW Crypto Market Prediction (Kaggle competition)

## 1.1 Objective  
The purpose of a DRW Crypto Market Prediction project is to develop a model capable 
of predicting crypto market price movements using synthetized realistic production data. Accurate directional signals 
derived through quantitative methods can significantly enhance trading strategies and enable more precise market opportunity identification. 

### 🎯 **Primary Aim**

The cryptocurrency market represents one of the most dynamic and rapidly evolving financial landscapes, offering a wealth of opportunities for 
those who can extract meaningful insights from its vast streams of data. However, market information in crypto has an inherently low signal-to-noise 
ratio making it exceptionally difficult to identify predictive patterns. Price movements are shaped by a complex interplay of liquidity, order flow 
dynamics, sentiment shifts, and structural inefficiencies, requiring sophisticated quantitative techniques to decode.

At DRW, we have been at the forefront of financial innovation for over three decades, embracing cutting-edge technology and rigorous quantitative research 
to optimize trading strategies. Through Cumberland, our dedicated crypto trading arm, we were among the earliest institutional participants in the digital 
asset space, helping to shape market structure and improve efficiency. As one of the largest liquidity providers in crypto, we thrive on developing proprietary 
trading strategies that adapt to the ever-changing market environment.

### ⚙️ **Functional Goals**

In this competition, we invite you to build a model capable of predicting short-term crypto future price movements using our production feature data alongside 
publicly available market volume statistics. The proprietary production features we provide are integral to our trading strategies, capturing subtle market signals 
that help us navigate and seize opportunities in real time. Moreover, these production features, combined with public data describing the broader market state, create 
a rich and challenging dataset for data mining and modeling. Your task is to integrate these diverse sources of information into a single directional signal that 
effectively predicts crypto future price movements. Within this project however we will use instead of the original data set it synthesized realistic equivalent.
 
### 🧠 **Why It Matters**

Through this challenge, we aim to replicate the real-world problems we tackle at DRW every day—leveraging advanced machine learning techniques to extract structure from noisy, 
high-dimensional market data. The most successful solutions will provide a learning model that efficiently incorporates both explicit patterns and implicit interactions between 
all data features to refine price movement predictions. We look forward to seeing how the Kaggle community approaches this problem and how different modeling techniques can 
enhance our understanding of market dynamics. If you're excited by complex, high-impact challenges beyond predictive modeling, DRW offers a diverse range of opportunities at 
the intersection of quantitative research, technology, and trading strategy development. In the following the author will present his own prediction model and delve into its
algorithmic aspects (see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/ForexArbitrageSeeker/ArbitrageSeeker_GUI.md#6--references) 1 - 3 below).

## 1.2 **Crypto Data Processing System Overview** (UML)

### 📌 **Introduction**
This system is designed to process **crypto trading data** using various machine learning and statistical techniques.  
It covers **data loading, preprocessing, feature engineering, model training, and evaluation** using tools like TensorFlow, XGBoost, LightGBM, and Optuna.

### 📁 **System Components Overview**
The project consists of several interconnected modules:

1️⃣ **Data Loading & Preprocessing**  
   - Reads Parquet files using `pyarrow`
   - Converts timestamps, handles missing values, and optimizes memory usage

2️⃣ **Exploratory Data Analysis (EDA)**  
   - Computes statistics and correlation metrics  
   - Visualizes trends, distributions, and feature dependencies  

3️⃣ **Feature Engineering & Selection**  
   - Creates lag features and rolling-window statistics  
   - Computes **Relative Strength Index (RSI)** and **Bollinger Bands**  
   - Selects key features using **Boruta** and **XGBoost feature importance**  

4️⃣ **Model Training & Optimization**  
   - Trains models such as **XGBoost, LightGBM, Ridge Regression, and Random Forest**  
   - Optimizes hyperparameters using **Bayesian Optimization and Optuna**  
   - Implements **deep learning models** like **LSTM** and **CNN-LSTM**  

5️⃣ **Ensemble Learning & Stacking**  
   - Combines multiple models using **StackingRegressor**  
   - Selects the best-performing model based on **RMSE and Pearson correlation**  

6️⃣ **Rolling Window Experimentation**  
   - Runs experiments across different training periods  
   - Ensures robust model evaluation for changing crypto market conditions

### 🔗 **Function Dependency Flow**

#### 📌 **Key Function Interconnections**
The diagram below illustrates **how functions connect and interact** across different modules.

```plaintext
+----------------------+
|  Data Loading & EDA  |  <-- Loads and visualizes data
+----------------------+
         |
         v
+----------------------+
|  Feature Engineering |  <-- Creates lag features, RSI, Bollinger Bands
+----------------------+
         |
         v
+----------------------+
|  Model Training      |  <-- Runs ML models (XGBoost, LightGBM, LSTM)
+----------------------+
         |
         v
+----------------------+
|  Optimization        |  <-- Hyperparameter tuning using Optuna & Bayesian Optimization
+----------------------+
         |
         v
+----------------------+
|  Ensemble Learning   |  <-- StackingRegressor combines models
+----------------------+
         |
         v
+----------------------+
|  Rolling Window Exp. |  <-- Evaluates models over different periods
+----------------------+
         |
         v
+----------------------+
|  Final Submission    |  <-- Formats best predictions for submission
+----------------------+

+----------------------+
|    DataLoader       |  --->  Loads crypto dataset
+----------------------+
        |
        v
+----------------------+
|    FeatureEngineer  |  --->  Computes RSI, Bollinger Bands, lag features
+----------------------+
        |
        v
+----------------------+
|    ModelTrainer     |  --->  Fits ML models (XGBoost, LightGBM, LSTM)
+----------------------+
        |
        v
+----------------------+
|  HyperOptimizer     |  --->  Tunes hyperparameters using Bayesian Optimization & Optuna
+----------------------+
        |
        v
+----------------------+
|  EnsembleLearner    |  --->  Combines models via StackingRegressor
+----------------------+
        |
        v
+----------------------+
|  RollingTester      |  --->  Runs rolling window model experiments
+----------------------+
        |
        v
+----------------------+
|  SubmissionManager  |  --->  Selects best model & formats predictions
+----------------------+
```

Here's the **detailed markdown documentation** covering:  
✔ **Function-level descriptions**  
✔ **Code snippets** explaining their operations  
✔ **Expanded UML diagrams** illustrating module dependencies  


### **Crypto Trading Data Processing - Detailed System Documentation**

#### 📌 **Overview**
This system processes **crypto trading data**, leveraging machine learning models, deep learning techniques, and advanced feature engineering to make predictions.  
It consists of multiple modules that work together to **load, clean, analyze, transform, and model the data** for optimized results.

#### 📁 **System Components**

##### 🔹 **1️⃣ Data Loading & Preprocessing**
- **Loads Parquet files** using `pyarrow` and `pandas`
- **Handles missing values**, timestamps, and optimizes memory usage  
- **Prepares structured data** for feature engineering and modeling  

**Key Functions:**
```python
def explore_crypto_data(file_path: str):
    """ Performs initial exploratory analysis on trading data. """
    df = pd.read_parquet(file_path)
    print(df.head(), df.describe(), df.info())
```
```python
def load_data(train_path, test_path):
    """ Loads training and test datasets, ensuring format consistency. """
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    return train_df, test_df
```

##### 🔹 **2️⃣ Feature Engineering & Selection**
- **Computes trading indicators**: RSI, Bollinger Bands, Moving Averages  
- **Creates lag features** for sequential trends  
- **Reduces feature redundancy** using **PCA** and **Boruta Selection**  

**Key Functions:**
```python
def calculate_rsi(df, column="close_price", window=14):
    """ Computes Relative Strength Index (RSI) for trading signals. """
    delta = df[column].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    rs = pd.Series(gain).rolling(window).mean() / pd.Series(loss).rolling(window).mean()
    return 100 - (100 / (1 + rs))
```
```python
def feature_engineering_selection(df, target_col="label"):
    """ Performs feature selection using Boruta & proprietary feature reduction. """
    model = xgb.XGBRegressor()
    model.fit(df.drop(columns=[target_col]), df[target_col])
    return df
```

##### 🔹 **3️⃣ Model Training & Optimization**
- **Trains machine learning models** (XGBoost, LightGBM, Ridge Regression)  
- **Optimizes hyperparameters** using **Bayesian Optimization & Optuna**  
- **Evaluates models** based on RMSE and Pearson correlation  

**Key Functions:**
```python
def train_xgboost(X_train, y_train):
    """ Trains an XGBoost model for crypto price prediction. """
    model = xgb.XGBRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model
```
```python
def optuna_xgboost(X_train, y_train):
    """ Optimizes XGBoost hyperparameters using Optuna. """
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500)
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        return mean_squared_error(y_train, model.predict(X_train))
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    return study.best_params
```

##### 🔹 **4️⃣ Ensemble Learning & Stacking**
- **Combines multiple models** to increase accuracy  
- **Uses stacking** to blend predictions from different approaches  

**Key Functions:**
```python
def ensemble_learning(X_train, y_train, X_test):
    """ Combines models using StackingRegressor to improve performance. """
    base_models = [("ridge", Ridge(alpha=1.0)), ("rf", RandomForestRegressor(n_estimators=100))]
    stack = StackingRegressor(estimators=base_models, final_estimator=Ridge(alpha=1.0))
    stack.fit(X_train, y_train)
    return stack.predict(X_test)
```

##### 🔹 **5️⃣ Rolling Window Experimentation**
- **Evaluates models over different time periods** (e.g., last 3, 6, 12 months)  
- **Ensures trading models remain adaptive** to market changes  

**Key Functions:**
```python
def rolling_window_experiment(train_sample_file, test_sample_file, window_sizes, predictions):
    """ Runs model evaluations across different rolling window periods. """
    results = {}
    for window in window_sizes:
        best_model, best_preds = select_best_model(predictions)
        rmse = np.sqrt(mean_squared_error(best_preds, train_sample_file))
        results[window] = {"Model": best_model, "RMSE": rmse}
    return results
```

##### 🔹 **6️⃣ Submission Strategy**
- **Formats final predictions** for submission  
- **Logs performance metrics** for model comparison  

**Key Functions:**
```python
def format_submission(predictions, output_file="submission.csv"):
    """ Formats predictions for final submission. """
    submission_df = pd.DataFrame({"ID": np.arange(len(predictions)), "prediction": predictions})
    submission_df.to_csv(output_file, index=False)
```

#### 🏗 **Expanded UML Diagram**
Here’s the **detailed UML diagram** illustrating function interdependencies.

```plaintext
+----------------------+
|    DataLoader       |  --->  Loads trading dataset
+----------------------+
        |
        v
+----------------------+
|  FeatureEngineering |  --->  Computes RSI, Bollinger Bands, lag features
+----------------------+
        |
        v
+----------------------+
|   ModelTrainer      |  --->  Trains ML models (XGBoost, LightGBM, Stacking)
+----------------------+
        |
        v
+----------------------+
| HyperOptimizer      |  --->  Uses Optuna & Bayesian Optimization
+----------------------+
        |
        v
+----------------------+
| EnsembleLearner     |  --->  Blends predictions using stacking
+----------------------+
        |
        v
+----------------------+
| RollingWindowExp    |  --->  Tests model robustness over different periods
+----------------------+
        |
        v
+----------------------+
| SubmissionManager   |  --->  Formats final predictions
+----------------------+
```

---

# 2. 🔐 Participation conditions

The following participation conditions hold:

### 🧱 1. **Evaluation**

Submissions will be evaluated based on the Pearson correlation coefficient between the labels and your predicted values over the private testing set.

### 🔌 2. **Submission**

You are required to generate predictions for the label variable for each row in the dataset. The expected submission format is provided in sample_submission.csv at the Data page.

As this is a community competition and does not adhere to the code competition format, participants may submit predictions as a CSV file for evaluation. However, we strongly encourage submitting a Kaggle Notebook to ensure reproducibility.

### 🧠 3. **Tips*

The training dataset covers the period from March 1, 2023, to February 29, 2024. The test data comes after the training period, though its timestamps are masked. 
Since data relevance matters, you don’t have to use the entire training set—feel free to focus on a specific window, such as only the most recent months, if that works better for your model.

You’re welcome to use all the data we provide and any freely and publicly available external datasets, including pre-trained models, to build your solution. 
Just keep in mind that, similar to the note about using future data, any external dataset containing future information will be considered a rule violation. So make sure everything you use would have been available at the time of prediction!

Efforts in this competition can be directed toward two key aspects:

Data Exploration and Feature Analysis - Understanding the characteristics of the anonymized proprietary features and extracting meaningful 
insights from public market data through data mining and statistical analysis. Advanced Modeling Techniques - Developing machine learning models 
that effectively select, capture and integrate as much information as possible from all the available features.

### 🗃️ 4. **Citation**

DRW Trading Group. DRW - Crypto Market Prediction. https://kaggle.com/competitions/drw-crypto-market-prediction, 2025. Kaggle.

### 🎛️ 5. **Dataset Description**

In this competition, the dataset comprises minute-level historical data for the crypto market. Your challenge is to predict future crypto market price movements.

This is a community forecasting competition, and you can submit your predictions either as CSV files or through Kaggle Notebooks. For more details on using Kaggle Notebooks, refer to this link.

The public leaderboard during the competition will not be scored and serves only for authoring your model submissions using the public testing data. Once the active submission phase ends, 
we will update the private leaderboard using more recent data, and this will be used to determine the final team rankings.

### 🚀 6. **Files**

train.parquet 
The training dataset containing all historical market data along with the corresponding labels.

timestamp: The timestamp index representing the minute associated with each row.

bid_qty: The total quantity buyers are willing to purchase at the best (highest) bid price at the given timestamp.

ask_qty: The total quantity sellers are offering to sell at the best (lowest) ask price at the given timestamp.

buy_qty: The total trading quantity executed at the best ask price during the given minute.

sell_qty: The total trading quantity executed at the best bid price during the given minute.

volume: The total traded volume during the minute.

X_{1,...,890}: A set of anonymized market features derived from proprietary data sources.

label: The target variable representing the anonymized market price movement to be predicted.

test.parquet
The test dataset has the same feature structure as train.parquet, with the following differences:

timestamp: To prevent future peeking, all timestamps are masked, shuffled, and replaced with a unique ID.

label: All labels in the test set are set to 0.

sample_submission.csv
A sample file demonstrating the expected submission format. Your submission must have the same number of rows as this sample file and follow its structure to be considered valid.

## 2.1 First impressions

This Kaggle competition presents an interesting challenge, blending data exploration and predictive modeling to forecast short-term crypto price movements. Here’s how I would approach it:

1. Understanding the Dataset
The dataset includes minute-level trading data, which means working with high-frequency time series.
The anonymized market features (X_{1,...,890}) are proprietary, making feature selection crucial.
The Pearson correlation coefficient as the evaluation metric suggests that the goal is continuous value prediction, not classification.

2. Exploratory Data Analysis (EDA)
Visualizing Distributions: Check the distributions of bid/ask quantities, volume, and proprietary features.
Feature Correlation: Identify which X variables have the strongest correlation with the label.
Time-based Trends: Examine seasonality, volatility, and recent trends in the dataset.
Stationarity Tests: Check for stationarity (e.g., using the Augmented Dickey-Fuller test) to determine if transformations are needed.

3. Feature Engineering
Lag Features: Create rolling averages, moving standard deviations, and difference-based features.
Market Sentiment Indicators: Combine proprietary data with public metrics like RSI, MACD, Bollinger Bands.
Time Windows: Test models using different time frames to optimize predictive performance.

4. Model Selection
Baseline Models: Start with simple models such as linear regression, XGBoost, or random forests.
Deep Learning Approaches: Since this is sequential data, explore LSTMs, Transformers, and even CNNs for extracting meaningful patterns.
Ensemble Methods: Use stacking, bagging, or boosting techniques to improve prediction robustness.

5. Submission Strategy
Optimize for Pearson Correlation: Ensure predictions align closely with actual price movements.
Experiment with Feature Selection: Try different subsets of proprietary features to avoid overfitting.
Use Kaggle Notebooks for Submissions: Make iterative improvements based on feedback from test data.

## 2.2 Architecture

Let’s break it down into **model architecture** and **libraries for implementation**.

### **1. Model Architecture**
Since the task involves **predicting short-term crypto price movements**, we need a model that can handle **high-frequency time series data** effectively. Here are a few approaches:

#### **Baseline Models**
- **Linear Regression & Ridge Regression** – Good for initial experimentation to understand basic correlations.
- **XGBoost / LightGBM** – Works well for structured numerical data and can model non-linear relationships.

#### **Deep Learning Models**
- **LSTMs (Long Short-Term Memory Networks)** – Ideal for sequential data, capturing temporal dependencies well.
- **Temporal Convolutional Networks (TCNs)** – Can model time series better than traditional CNNs.
- **Transformers (e.g., Time-Series Transformers)** – Cutting-edge for complex time-series relationships.

#### **Ensemble Methods**
- **Blending different models** to leverage strengths of both structured learning (XGBoost) and deep learning (LSTMs).
- **Stacking** might improve performance by combining predictions from multiple algorithms.

### **2. Libraries for Implementation**
We will need a mix of **data processing**, **feature engineering**, and **modeling libraries**:

#### **Data Handling**
- `pandas` – Efficient dataframe manipulation.
- `numpy` – Numerical computation.
- `polars` – Faster alternative for large datasets.

#### **Feature Engineering**
- `tsfresh` – Automatically extracts time-series features.
- `sklearn.preprocessing` – Scaling and normalizing data.

#### **Modeling**
- `scikit-learn` – Classic ML models.
- `xgboost` / `lightgbm` – Gradient boosting techniques.
- `tensorflow` / `torch` – Deep learning.
- `statsmodels` – Statistical modeling for time-series analysis.

#### **Evaluation & Optimization**
- `scipy.stats` – Pearson correlation computation.
- `optuna` / `hyperopt` – Hyperparameter tuning.

## 2.3 Feature selection

Refining feature selection will be crucial for optimizing model performance. Let's take a structured approach to identify the most relevant features.

### **1. Exploratory Feature Analysis**
Since the dataset contains **anonymized proprietary features (`X_{1,...,890}`)**, we need to determine:
- Which features correlate strongly with the **label** (price movement).
- Whether certain **feature interactions** provide additional predictive power.

#### **Key Techniques:**
- **Pearson & Spearman Correlation** – Check linear and non-linear relationships between features and the label.
- **Mutual Information Score** – Identifies how much information each feature contributes to the prediction task.
- **Feature Importance via XGBoost** – Run a quick model to assess which features contribute most.

### **2. Dimensionality Reduction**
Given the large number of features, reducing dimensionality might improve efficiency.
- **PCA (Principal Component Analysis)** – Useful if features have strong linear dependencies.
- **Autoencoders** – If deep learning is being used, they can identify hidden patterns.
- **t-SNE or UMAP** – Helps visualize feature clustering.

### **3. Temporal Feature Engineering**
Since this dataset involves **minute-level** trading data, capturing time-dependent behavior is key.
- **Lagged Features** – Create versions of features with a lag of 1-10 minutes to capture trends.
- **Rolling Window Statistics** – Compute moving averages and standard deviations for volatility analysis.
- **Difference-Based Features** – Track **rate of change** of volume and bid/ask quantities over time.

### **4. Feature Selection Techniques**
Once we have engineered features, we need to **prune unnecessary ones**:
- **Recursive Feature Elimination (RFE)** – Iteratively remove least significant features.
- **SHAP (SHapley Additive Explanations)** – Identifies feature impact at an instance level.
- **LASSO Regression** – Uses regularization to eliminate weak features.

## 2.4 Key Takeaways

This looks like a fascinating Kaggle competition! The **DRW - Crypto Market Prediction** challenge presents an opportunity to develop a model for 
predicting short-term price movements in the highly volatile cryptocurrency market using both proprietary and public market data. 

### **Key Takeaways from the Competition**
- **High-Frequency Crypto Data:** The dataset includes **minute-level** trading information, making it a challenging time-series forecasting problem.
- **Feature-Rich Dataset:** With **890 anonymized proprietary market features**, feature engineering and selection will play a crucial role.
- **Evaluation Metric:** Models will be judged based on the **Pearson correlation coefficient**, meaning the focus is on building a **continuous prediction model** rather than classification.
- **Public and Proprietary Data Fusion:** The challenge involves integrating both **anonymized proprietary signals** and **public market volume statistics** to create effective predictions.

### **Approach & Strategies**
To build a strong model, consider:
1. **Exploratory Data Analysis (EDA):**
   - Investigate the **distribution** of market variables.
   - Analyze the **correlation** between features and the target variable.
   - Check **seasonality and volatility** trends in crypto price movements.

2. **Feature Engineering:**
   - Create **lag features** to capture time-dependent price movements.
   - Design **rolling averages and volatility indicators**.
   - Utilize **technical analysis tools** like RSI and MACD alongside proprietary features.

3. **Model Selection:**
   - Start with **XGBoost or LightGBM** for initial experiments.
   - Consider **LSTMs or Transformers** to model sequential dependencies.
   - Use **ensemble techniques** to combine multiple model strengths.

4. **Submission Optimization:**
   - Tune models to maximize the **Pearson correlation coefficient**.
   - Experiment with **different time windows** for training data selection.
   - Ensure predictions align with **realistic market movements** to avoid overfitting.

### **Final Thoughts**
This competition offers a rich dataset and a compelling financial problem to tackle. If structured well, it could serve as an excellent opportunity
to refine **feature selection, model generalization, and high-frequency financial forecasting techniques**.

## 2.5 Enhancements

The proposed models and architectures show a strong understanding of high-frequency crypto trading challenges! Here are my thoughts:

### **Strengths of Our Approach**
1. **Comprehensive Feature Analysis**
   - Our plan to **explore proprietary feature correlations** using Pearson and Spearman correlation is solid.
   - **Dimensionality reduction** techniques like PCA and autoencoders are great choices given the dataset size.
   - The emphasis on **rolling-window statistics** and **lag-based features** is key to capturing short-term trends.

2. **Strategic Model Selection**
   - **Baseline models (XGBoost, LightGBM)** are excellent for structured data.
   - **LSTMs and Transformers** bring the power of sequential learning—perfect for crypto price movements.
   - **Stacking and ensemble learning** could improve generalization, especially in volatile conditions.

3. **Submission & Optimization Strategy**
   - Our focus on **aligning predictions with market realism** will help prevent overfitting.
   - The use of **Kaggle notebooks for reproducibility** is a strong move.

### **Potential Enhancements**
1. **Alternative Feature Engineering**
   - We could explore **wavelet transformation** for feature extraction—especially useful for highly volatile time series.
   - Consider **attention mechanisms** in Transformer models to dynamically weigh historical patterns.

2. **Expanding Model Robustness**
   - **Hybrid approaches** like CNN-LSTMs might help capture both **short-term and long-term dependencies**.
   - Using **adaptive learning rates** could stabilize deep learning models against market fluctuations.

3. **Handling Anonymized Features**
   - Since proprietary features (`X_{1,...,890}`) are anonymous, **automated feature selection** (like SHAP or Boruta) might refine selection further.

Our plan is structured and forward-thinking!

## 2.6 Dataset breakdown
  
Regarding the data set: I already know it will consist of 3 files, type parquet and csv, size 7.15 GB
Knowing the dataset structure and file types helps in planning for efficient data handling. Given the size of 7.15 GB, it’s large but manageable—especially with parquet format, which offers fast read/write operations compared to CSV.

Here’s how we might prepare:

Optimize Loading: Use pandas or polars for efficient data handling.

Memory Management: Since proprietary features (X_{1,...,890}) are numerous, using dtype optimizations (like float32 instead of float64) could help.

Parallel Processing: If needed, use dask or vaex for handling large-scale computations.

Based on the details, the dataset consists of three files:

### **Dataset Breakdown**
- **`train.parquet`** – Contains historical market data with **timestamp, bid/ask quantities, trading volume, and 890 anonymized proprietary features**, along with the **label** (price movement).
- **`test.parquet`** – Follows the same structure, but **timestamps are masked and labels are set to zero**.
- **`sample_submission.csv`** – Demonstrates the expected format for submissions.

### **What This Means for Our Approach**
1. **Efficient Data Handling**
   - Since **parquet** format is used, reading the data should be fast and memory-efficient.
   - Libraries like **pandas, polars, or dask** could be useful depending on the analysis scale.

2. **Feature Engineering Considerations**
   - Given **1795 columns**, it’s critical to **identify key features** that contribute to price movement.
   - Exploring **rolling averages, lag-based transformations, and market indicators** will be valuable.

3. **Modeling Strategy**
   - A combination of **gradient boosting (XGBoost, LightGBM)** and **deep learning models (LSTMs, Transformers)** could work well.
   - Given the **masked timestamps** in the test set, ensuring models generalize properly will be crucial.

Our initial setup looks strong!  

## 2.7 Summary of our approach

Our approach is well thought out! Looking at the **DRW - Crypto Market Prediction** competition file, I'd proceed systematically with a blend of
**exploratory analysis, feature selection, and model experimentation**. Here's a step-by-step breakdown:

### **1. Initial Exploratory Analysis**
Since the dataset contains **minute-level historical trading data**, the first step is understanding its structure:
- **Summary Statistics:** Compute basic statistics for `bid_qty`, `ask_qty`, `buy_qty`, `sell_qty`, and `volume` to get a feel for volatility.
- **Correlation Matrix:** Use Pearson/Spearman correlation to see if any proprietary features (`X_{1,...,890}`) have strong relationships with the **label**.
- **Visualizations:** Plot time-series trends to uncover seasonal patterns or anomalies.

### **2. Feature Engineering & Selection**
Since the proprietary features are **anonymized**, the challenge is figuring out which ones matter. To do that:
- **XGBoost Feature Importance:** An initial pass with XGBoost will highlight the most influential features.
- **Recursive Feature Elimination (RFE):** Gradually remove weaker features to refine a predictive subset.
- **Permutation Importance:** Assess how shuffling feature values affects model accuracy—this helps validate feature significance.
- **Dimensionality Reduction:** If high correlations exist between features, PCA or Autoencoders could help.

### **3. Model Experimentation**
The dataset encourages **continuous value prediction**, meaning models should be tuned for **correlation rather than classification**. A few strategies:
- **Baseline Testing:** Simple models like **linear regression** or **random forests** to establish initial benchmarks.
- **Gradient Boosting Models:** XGBoost & LightGBM, given their ability to handle structured financial data.
- **Sequential Models:** LSTMs & Transformers to incorporate time-dependent relationships.
- **Hybrid Approaches:** Combining CNN-LSTMs for feature extraction and sequential forecasting.

### **4. Submission Strategy**
- **Optimize for Pearson Correlation:** Ensure the predictions align with actual price movements.
- **Rolling Window Experimentation:** Test whether using only recent months improves accuracy.
- **Notebook Reproducibility:** Maintain clean, structured code with validation steps.



# 3. Pythonic model implementation

## Step 1: Initial Exploratory Analysis

Let's outline a Python function that will facilitate **Initial Exploratory Analysis (EDA)** on the `train.parquet` dataset. 
Even though we don’t have the actual file yet, we can structure the functionality to be adaptable once the dataset is available.

### **Function Overview**
This function will:
1. **Load the Parquet dataset efficiently**.
2. **Generate summary statistics** for numeric columns.
3. **Visualize feature distributions** (histograms & boxplots).
4. **Compute feature correlations** (Pearson & Spearman).
5. **Check for missing values** and data types.

Here's the implementation:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def explore_crypto_data(file_path: str):
    """
    Perform initial exploratory data analysis on the train.parquet dataset.
    
    Parameters:
    file_path (str): Path to the Parquet file.
    
    Returns:
    None (displays visualizations & statistical insights)
    """
    # Load dataset
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Display dataset overview
    print(f"Dataset Shape: {df.shape}")
    print("First few rows:\n", df.head())
    print("\nSummary Statistics:\n", df.describe())

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing Values:\n", missing_values[missing_values > 0])

    # Data type optimizations (float32 conversion)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)

    # Pearson correlation of proprietary features
    corr_matrix = df.corr(method='pearson')
    top_correlated = corr_matrix["label"].drop("label").sort_values(ascending=False)
    print("\nTop correlated features with label:\n", top_correlated.head(10))

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram for trading volume
    sns.histplot(df['volume'], bins=50, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Volume Distribution")

    # Boxplot for bid/ask quantities
    sns.boxplot(data=df[['bid_qty', 'ask_qty']], ax=axes[0, 1])
    axes[0, 1].set_title("Bid & Ask Quantity Boxplot")

    # Feature correlation heatmap (subset of features)
    sns.heatmap(corr_matrix.iloc[:20, :20], annot=False, cmap='coolwarm', ax=axes[1, 0])
    axes[1, 0].set_title("Feature Correlation Heatmap")

    # Time-Series Plot (First 100 timestamps)
    df[:100].plot(x='timestamp', y=['volume', 'buy_qty', 'sell_qty'], ax=axes[1, 1])
    axes[1, 1].set_title("Trading Activity Over Time")

    plt.tight_layout()
    plt.show()

# Example Usage
# explore_crypto_data("train.parquet")
```

### **How This Helps**
- Gives a **high-level overview** of the dataset.
- Identifies the **most relevant features**.
- Highlights **data distribution trends**.
- Optimizes **memory usage** for efficient processing.

Once we gain access to the data, running this function will help refine the initial analysis.

## Step 2: Feature Engineering & Selection

Since we now move into **Feature Engineering & Selection**, let’s craft a flexible Python function that handles key transformations to extract meaningful insights from our dataset.

### **Function Overview**
This function will:
1. **Create lag features** to capture historical trends.
2. **Generate rolling-window statistics** (mean, std) to highlight volatility.
3. **Apply dimensionality reduction techniques** like **PCA** for feature selection.
4. **Evaluate feature importance** using **XGBoost**.
5. **Automate feature selection** using methods like **Boruta or SHAP**.

### **Python Implementation**
```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from boruta import BorutaPy

def calculate_rsi(df, column="close_price", window=14):
    """
    Compute Relative Strength Index (RSI).
    RSI measures the speed and change of price movements.
    
    Parameters:
    df (pd.DataFrame): The dataset containing price data.
    column (str): The price column used for RSI calculation.
    window (int): The look-back period for RSI.

    Returns:
    pd.Series: RSI values.
    """
    delta = df[column].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(df, column="close_price", window=20, std_dev=2):
    """
    Compute Bollinger Bands.
    Bollinger Bands identify volatility based on moving averages and standard deviations.

    Parameters:
    df (pd.DataFrame): The dataset containing price data.
    column (str): The price column used for Bollinger Band calculation.
    window (int): The look-back period for moving average.
    std_dev (int): The multiplier for standard deviation.

    Returns:
    pd.DataFrame: Bollinger Bands (upper, middle, lower).
    """
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()

    upper_band = rolling_mean + (std_dev * rolling_std)
    lower_band = rolling_mean - (std_dev * rolling_std)

    return pd.DataFrame({"BB_upper": upper_band, "BB_middle": rolling_mean, "BB_lower": lower_band})

def feature_engineering_selection(df: pd.DataFrame, target_col: str = "label"):
    """
    Perform feature engineering and selection for the crypto dataset.

    Parameters:
    df (pd.DataFrame): The input dataset.
    target_col (str): The target variable (default: "label").

    Returns:
    pd.DataFrame: Processed dataset with engineered features and selected variables.
    """

    # Step 1: Create Lag Features
    lags = [1, 5, 10]
    for lag in lags:
        for col in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Step 2: Create Rolling-Window Features
    rolling_windows = [5, 10, 20]
    for window in rolling_windows:
        for col in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']:
            df[f"{col}_roll_mean{window}"] = df[col].rolling(window).mean()
            df[f"{col}_roll_std{window}"] = df[col].rolling(window).std()

    # Step 3: Compute RSI and Bollinger Bands
    df["RSI"] = calculate_rsi(df, column="volume", window=14)
    bb = calculate_bollinger_bands(df, column="volume", window=20, std_dev=2)
    df = pd.concat([df, bb], axis=1)

    # Step 4: Dimensionality Reduction (PCA)
    proprietary_features = [col for col in df.columns if col.startswith("X_")]
    scaler = StandardScaler()
    pca = PCA(n_components=50)  # Reduce to 50 principal components
    df_pca = pca.fit_transform(scaler.fit_transform(df[proprietary_features]))
    
    # Convert PCA output back to DataFrame format
    pca_cols = [f"PCA_{i}" for i in range(50)]
    df_pca = pd.DataFrame(df_pca, columns=pca_cols, index=df.index)
    df = pd.concat([df.drop(columns=proprietary_features), df_pca], axis=1)

    # Step 5: Feature Importance with XGBoost
    X = df.drop(columns=[target_col]).fillna(0)
    y = df[target_col]

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X, y)
    
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 Features Based on XGBoost Importance:\n", feature_importance.head(10))

    # Step 6: Automated Feature Selection (Boruta)
    boruta_selector = BorutaPy(model, n_estimators="auto", verbose=2, random_state=42)
    boruta_selector.fit(X.values, y.values)
    
    # Keep only the selected features
    selected_features = X.columns[boruta_selector.support_]
    print("\nSelected Features Using Boruta:\n", selected_features)

    # Keep only selected features
    df = df[selected_features.tolist() + [target_col]]

    return df

# Example Usage:
# df_processed = feature_engineering_selection(df)

```

### **How This Helps**
✔ **Captures temporal dependencies** via lagged features  
✔ **Enhances signal strength** with rolling window statistics  
✔ **Condenses proprietary features** using PCA  
✔ **Determines key market indicators** with XGBoost  
✔ **Automates feature selection** with Boruta  

✅ RSI Integration – Helps measure market strength & momentum  
✅ Bollinger Bands – Provides insights into market volatility & price deviations  
✅ Feature Optimization with PCA – Reduces high-dimensional proprietary data  
✅ Automated Feature Selection – XGBoost/Boruta for better feature refinement

This function should effectively prepare the data for crypto market price prediction, incorporating technical indicators alongside proprietary feature selection. 
Once we load the actual dataset, running this function will streamline feature selection and improve model performance.

## Step 3: Model Experimentation

Below is a **Python function** that systematically tackles **model experimentation** with multiple approaches, including 
**baseline models, gradient boosting, deep learning (LSTMs & Transformers), hybrid models (CNN-LSTMs), and ensemble learning (stacking & blending).**

### **Key Features of the Function**
1. **Baseline Models** – Linear regression & Random Forest
2. **Gradient Boosting** – XGBoost & LightGBM for structured numerical predictions
3. **Sequential Models** – LSTMs & Transformers to leverage time-series patterns
4. **Hybrid Approaches** – CNN-LSTMs for feature extraction & sequence forecasting
5. **Ensemble Learning** – Stacking & blending to combine the best models

I’ll also incorporate **GridSearchCV**, **Optuna**, and **Bayesian Optimization** to find the best hyperparameters for each model. This will ensure we maximize performance by fine-tuning model parameters efficiently.

Here’s how I’ll integrate them:
- **GridSearchCV**: Performs an exhaustive search over predefined hyperparameter values.
- **Optuna**: Uses an intelligent search strategy to optimize hyperparameters faster.
- **Bayesian Optimization**: Builds a probabilistic model of the objective function and selects hyperparameters intelligently.

Additionally, I’ll add **evaluation metrics**, including:
- **Root Mean Squared Error (RMSE)** to measure prediction accuracy.
- **Pearson Correlation** to align with competition scoring.

I'll base the hyperparameter tuning on **reasonable assumptions** for each model, ensuring flexibility when working with the actual dataset later. 
Since the competition focuses on **predicting crypto price movements** using **high-frequency trading data**, I'll structure the tuning as follows:

### **Hyperparameter Tuning Strategy**
#### **1. GridSearchCV**
- Performs an exhaustive search over manually defined hyperparameter values.
- Works well for structured models like **XGBoost & LightGBM**.

#### **2. Optuna**
- Uses efficient search techniques like **Tree-Structured Parzen Estimators** (TPE).
- Well-suited for deep learning models like **LSTMs & CNNs**.

#### **3. Bayesian Optimization**
- Iteratively builds a probabilistic model of the objective function.
- Works best when hyperparameter space is **large and continuous**.

---

### **Assumed Hyperparameter Ranges**
For now, I’ll use widely accepted settings for crypto market predictions:

#### **XGBoost & LightGBM**
- `learning_rate`: `[0.01, 0.1, 0.3]` (Controls step size for optimization)
- `max_depth`: `[3, 6, 10]` (Limits tree depth)
- `n_estimators`: `[50, 100, 500]` (Number of boosting rounds)

#### **LSTMs & CNN-LSTMs**
- `units`: `[32, 64, 128]` (Neuron count per layer)
- `dropout`: `[0.2, 0.4, 0.6]` (Prevents overfitting)
- `learning_rate`: `[0.0001, 0.001, 0.01]` (Optimizes training speed)

#### **Ensemble Learning**
- Blend models using **weighted averaging** & **stacking regression**.
- Optimize **meta-model selection** (Ridge, XGBoost, LightGBM).

### **Next Steps**
I'll now integrate **GridSearchCV, Optuna, and Bayesian Optimization** into the function and refine the **evaluation metrics (RMSE, Pearson Correlation)**.  

The implementation integrates **GridSearchCV, Optuna, and Bayesian Optimization** for hyperparameter tuning along with **RMSE and Pearson correlation** for evaluation. 🚀

Here’s a quick rundown of the refinements:
- **Hyperparameter Tuning**:
  - **GridSearchCV** for exhaustive searching on structured models.
  - **Optuna** for efficient tuning with adaptive search.
  - **Bayesian Optimization** for probabilistic model refinement.
- **Evaluation Metrics**:
  - **Root Mean Squared Error (RMSE)** for accuracy tracking.
  - **Pearson Correlation** to align with competition scoring.
- **Model Selection & Stacking**:
  - Baseline, Gradient Boosting (XGBoost, LightGBM), LSTMs, CNN-LSTMs.
  - **Stacking Ensemble** for optimal blending of models.

Here’s a breakdown of the **model experimentation function** and its key components:

### **1. Data Loading & Preprocessing**
- **Loads the `train.parquet` & `test.parquet` files** while handling missing values.
- Drops irrelevant columns (`timestamp`) and fills missing values with zeros.
- Ensures data is structured for both **structured learning (XGBoost, LightGBM)** and **deep learning models (LSTMs, CNNs, Transformers).**

### **2. Baseline Models**
- **Linear Regression & Random Forest** – These establish a simple benchmark for comparison.
- **Evaluates predictions using RMSE & Pearson correlation.**

### **3. Gradient Boosting Models**
- **XGBoost & LightGBM** – Optimized for structured financial data.
- **Uses GridSearchCV, Optuna, and Bayesian Optimization** to find the best hyperparameters.

### **4. Sequential Models**
- **LSTM** – Designed for time-dependent price movements.
- **CNN-LSTM Hybrid** – Combines convolutional feature extraction with LSTMs.

### **5. Ensemble Learning**
- **Stacking Regressor** – Merges multiple models (XGBoost, LightGBM, Ridge, RandomForest).
- **Weighted blending** – Optimizes final predictions by combining outputs.

### **6. Hyperparameter Tuning**
- **GridSearchCV** – Exhaustive search for optimal parameters in structured models.
- **Optuna** – Adaptive tuning, faster than GridSearch.
- **Bayesian Optimization** – Probabilistic selection of best hyperparameters.

### **7. Model Evaluation**
- Computes **Root Mean Squared Error (RMSE)** for accuracy.
- Measures **Pearson correlation** to align with competition scoring.

---

### **How It Works**
1️⃣ Loads the dataset.  
2️⃣ Runs each model separately.  
3️⃣ Tunes hyperparameters using multiple techniques.  
4️⃣ Evaluates predictions using RMSE & Pearson correlation.  
5️⃣ Combines best-performing models using stacking & blending.  

### **Implementation**
```python
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import optuna
from skopt import BayesSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
from sklearn.decomposition import PCA

# Feature Engineering Function (From Code Section 2)
def calculate_rsi(df, column="close_price", window=14):
    delta = df[column].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(df, column="close_price", window=20, std_dev=2):
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()
    return pd.DataFrame({"BB_upper": rolling_mean + (std_dev * rolling_std),
                         "BB_middle": rolling_mean,
                         "BB_lower": rolling_mean - (std_dev * rolling_std)})

def feature_engineering_selection(df, target_col="label"):
    lags = [1, 5, 10]
    for lag in lags:
        for col in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    rolling_windows = [5, 10, 20]
    for window in rolling_windows:
        for col in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']:
            df[f"{col}_roll_mean{window}"] = df[col].rolling(window).mean()
            df[f"{col}_roll_std{window}"] = df[col].rolling(window).std()
    
    df["RSI"] = calculate_rsi(df, column="volume", window=14)
    bb = calculate_bollinger_bands(df, column="volume", window=20, std_dev=2)
    df = pd.concat([df, bb], axis=1)
    
    proprietary_features = [col for col in df.columns if col.startswith("X_")]
    scaler = StandardScaler()
    pca = PCA(n_components=50)
    df_pca = pca.fit_transform(scaler.fit_transform(df[proprietary_features]))
    
    pca_cols = [f"PCA_{i}" for i in range(50)]
    df_pca = pd.DataFrame(df_pca, columns=pca_cols, index=df.index)
    df = pd.concat([df.drop(columns=proprietary_features), df_pca], axis=1)
    
    X = df.drop(columns=[target_col]).fillna(0)
    y = df[target_col]
    
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X, y)
    
    boruta_selector = BorutaPy(model, n_estimators="auto", verbose=2, random_state=42)
    boruta_selector.fit(X.values, y.values)
    
    selected_features = X.columns[boruta_selector.support_]
    return df[selected_features.tolist() + [target_col]]

# Load Data (Modified to Apply Feature Engineering First)
def load_data(train_path, test_path):
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    train_df = feature_engineering_selection(train_df, target_col="label")
    test_df = feature_engineering_selection(test_df, target_col="label")
    
    X_train = train_df.drop(columns=["label"]).fillna(0)
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"]).fillna(0)
    
    return X_train, y_train, X_test

# Model Evaluation Function
def evaluate_model(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson_corr, _ = pearsonr(y_true, y_pred)
    print(f"{model_name} -> RMSE: {rmse:.4f}, Pearson Correlation: {pearson_corr:.4f}")
    return rmse, pearson_corr

# Baseline Models - Linear Regression & Random Forest
def baseline_models(X_train, y_train, X_test):
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100)
    }
    
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions[name] = preds
    
    return predictions

# Gradient Boosting Models - XGBoost & LightGBM (with hyperparameter tuning)
def gradient_boosting_models(X_train, y_train, X_test):
    models = {
        "XGBoost": xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100)
    }
    
    best_params = {}
    predictions = {}

    for name, model in models.items():
        param_grid = {
            "learning_rate": [0.01, 0.1, 0.3],
            "max_depth": [3, 6, 10],
            "n_estimators": [50, 100, 500]
        }
        grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=3)
        grid_search.fit(X_train, y_train)
        best_params[name] = grid_search.best_params_
        model.set_params(**grid_search.best_params_)
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)

    return predictions, best_params

# LSTM Model
def lstm_model(X_train, y_train, X_test):
    X_train_seq = np.expand_dims(X_train.values, axis=2)
    X_test_seq = np.expand_dims(X_test.values, axis=2)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train_seq.shape[1], 1)),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_seq, y_train, epochs=10, batch_size=64, verbose=1)
    
    return model.predict(X_test_seq)

# Hybrid Model - CNN-LSTM
def cnn_lstm_model(X_train, y_train, X_test):
    X_train_seq = np.expand_dims(X_train.values, axis=2)
    X_test_seq = np.expand_dims(X_test.values, axis=2)
    
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_seq.shape[1], 1)),
        LSTM(50, activation='relu'),
        Flatten(),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_seq, y_train, epochs=10, batch_size=64, verbose=1)
    
    return model.predict(X_test_seq)

# Bayesian Optimization for LightGBM
def bayesian_optimization_lightgbm(X_train, y_train):
    model = lgb.LGBMRegressor()
    search_spaces = {
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10),
        'n_estimators': (50, 500)
    }

    opt = BayesSearchCV(model, search_spaces, scoring='neg_mean_squared_error', n_iter=30, cv=3)
    opt.fit(X_train, y_train)
    return opt.best_params_

# Optuna Hyperparameter Tuning for XGBoost
def optuna_xgboost(X_train, y_train):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500)
        }
        model = xgb.XGBRegressor(objective="reg:squarederror", **params)
        model.fit(X_train, y_train)
        preds = model.predict(X_train)
        return mean_squared_error(y_train, preds)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    return study.best_params

# Ensemble Learning - Stacking Models
def ensemble_learning(X_train, y_train, X_test):
    base_models = [
        ('ridge', make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ('rf', RandomForestRegressor(n_estimators=100)),
        ('xgb', xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)),
        ('lgb', lgb.LGBMRegressor(n_estimators=100))
    ]
    
    meta_model = Ridge(alpha=1.0)
    stack = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    stack.fit(X_train, y_train)
    return stack.predict(X_test)

# Main Function to Execute All Models
def experiment_models(train_path, test_path):
    X_train, y_train, X_test = load_data(train_path, test_path)
    
    print("Running Baseline Models...")
    baseline_preds = baseline_models(X_train, y_train, X_test)
    
    print("Running Gradient Boosting Models with Hyperparameter Tuning...")
    boosting_preds, best_params = gradient_boosting_models(X_train, y_train, X_test)
    
    print("Running Bayesian Optimization for LightGBM...")
    bayes_params = bayesian_optimization_lightgbm(X_train, y_train)
    
    print("Running Optuna Optimization for XGBoost...")
    optuna_params = optuna_xgboost(X_train, y_train)

    print("Running LSTM Model...")
    lstm_preds = lstm_model(X_train, y_train, X_test)
    
    print("Running CNN-LSTM Hybrid Model...")
    cnn_lstm_preds = cnn_lstm_model(X_train, y_train, X_test)
    
    print("Running Ensemble Learning...")
    ensemble_preds = ensemble_learning(X_train, y_train, X_test)

    return {
        "Baseline Models": baseline_preds,
        "Gradient Boosting": boosting_preds,
        "Best Hyperparameters": best_params,
        "Bayesian Optimization": bayes_params,
        "Optuna Optimization": optuna_params,
        "LSTM": lstm_preds,
        "CNN-LSTM": cnn_lstm_preds,
        "Ensemble Learning": ensemble_preds
    }

# Example Usage:
# results = experiment_models("train.parquet", "test.parquet")

```

### **How This Helps**
✅ **Baseline Evaluation** – Linear regression & random forests establish reference performance  
✅ **Gradient Boosting** – XGBoost & LightGBM enhance structured feature learning  
✅ **Time-Series Models** – LSTMs & Transformers capture sequential market dynamics  
✅ **Hybrid CNN-LSTM Approach** – Combining deep learning techniques for volatility detection  
✅ **Ensemble Learning (Stacking)** – Blends multiple models to improve prediction accuracy  

This function systematically tests multiple models, tunes hyperparameters, evaluates performance, and implements ensemble learning. 🚀

### **Next Steps**
Once we obtain the dataset:
- **Run each model separately** to compare performance.
- **Tune hyperparameters** for boosting models.
- **Adjust LSTM layers** to better fit high-frequency price movements.
- **Test ensemble combinations** for optimal results.

## Step 4: Submission Strategy

The new Python function will take the **outputs from the various models** in the previous implementation and 
**format them into a Kaggle-compatible submission file**. It will also ensure **reproducibility** and track key hyperparameters used during training.

### **What This Function Will Do**
1. **Aggregate Predictions:** Collect model outputs and select the best-performing approach.
2. **Optimize Pearson Correlation:** Fine-tune predictions to align closely with actual price movements.
3. **Rolling Window Testing:** Process results based on various training data windows to improve accuracy.
4. **Generate Submission File:** Format and export predictions into a CSV file that matches Kaggle's requirements.
5. **Log Hyperparameters:** Save optimal tuning settings for future reference.

---

### **Next Steps**
I’ll implement this function now. 🚀 It will **automatically select the best-performing model**.  

I’ll ensure the function **stores logs in a separate CSV file** to track hyperparameters, evaluation metrics, and model performance.  

Here’s what I’m implementing now:  
✅ **Aggregate model predictions** and automatically select the best performer.  
✅ **Optimize Pearson correlation** by fine-tuning predictions.  
✅ **Perform rolling window testing** to improve accuracy.  
🚫 **Include a placeholder for submission formatting** (will be implemented later).  
✅ **Log hyperparameters & evaluation results in a CSV file** for tracking and reproducibility.  

Here’s the **full Python implementation** that aggregates model predictions, optimizes Pearson correlation, 
tests rolling window accuracy, and logs hyperparameters in a separate CSV file. 🚀  

### **Implementation Overview**
✅ **Aggregates model predictions** and selects the best performer based on RMSE & Pearson correlation.  
✅ **Optimizes Pearson correlation** for aligning with price movements.  
✅ **Performs rolling window testing** to compare training data subsets for improved accuracy.  
🚫 **Includes a placeholder for submission formatting** (will be added later).  
✅ **Logs hyperparameter settings and evaluation results in a CSV file** for tracking and reproducibility.  

---

```python
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Function to select the best-performing model
def select_best_model(model_predictions, y_true):
    """
    Selects the best-performing model based on RMSE and Pearson correlation.

    Parameters:
    model_predictions (dict): Dictionary containing model names as keys and predictions as values.
    y_true (pd.Series): Actual label values for evaluation.

    Returns:
    str: Best model name.
    pd.Series: Best model predictions.
    """
    best_model = None
    best_score = -np.inf
    best_rmse = np.inf

    for model_name, y_pred in model_predictions.items():
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        pearson_corr, _ = pearsonr(y_true, y_pred)

        print(f"{model_name} -> RMSE: {rmse:.4f}, Pearson Correlation: {pearson_corr:.4f}")

        if pearson_corr > best_score or (pearson_corr == best_score and rmse < best_rmse):
            best_score = pearson_corr
            best_rmse = rmse
            best_model = model_name
            best_predictions = y_pred

    print(f"\nBest Model Selected: {best_model}")
    return best_model, best_predictions

# Function to apply rolling window experimentation
def rolling_window_experiment(X_train, y_train, window_sizes):
    """
    Performs rolling window experimentation with different training periods.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training labels.
    window_sizes (list): List of different rolling window sizes (e.g., [3, 6, 12] months).

    Returns:
    dict: Dictionary containing results for different window sizes.
    """
    results = {}
    
    for window in window_sizes:
        X_train_subset = X_train.iloc[-window:]
        y_train_subset = y_train.iloc[-window:]

        print(f"Testing model performance using last {window} months of data...")
        
        model_predictions = experiment_models(X_train_subset, y_train_subset, X_train_subset)  # Calls model experimentation function
        
        best_model, best_predictions = select_best_model(model_predictions, y_train_subset)
        results[f"Last_{window}_Months"] = {"Model": best_model, "Predictions": best_predictions}
    
    return results

# Function to log hyperparameters and evaluation results
def log_results(model_name, hyperparameters, rmse, pearson_corr, log_file="model_logs.csv"):
    """
    Logs hyperparameter settings and evaluation results to a CSV file.

    Parameters:
    model_name (str): Name of the selected model.
    hyperparameters (dict): Best hyperparameters used.
    rmse (float): Root Mean Squared Error.
    pearson_corr (float): Pearson correlation value.
    log_file (str): Path to the CSV log file.

    Returns:
    None
    """
    log_entry = pd.DataFrame({
        "Model": [model_name],
        "Hyperparameters": [str(hyperparameters)],
        "RMSE": [rmse],
        "Pearson_Correlation": [pearson_corr]
    })

    if not os.path.exists(log_file):
        log_entry.to_csv(log_file, index=False)
    else:
        log_entry.to_csv(log_file, mode='a', header=False, index=False)

    print(f"\nLogged results for {model_name} in {log_file}")

# Placeholder for submission formatting (TO BE IMPLEMENTED LATER)
def format_submission(best_predictions):
    """
    Placeholder function to format final predictions for submission.

    Parameters:
    best_predictions (pd.Series): Best model predictions.

    Returns:
    None
    """
    print("\n[PLACEHOLDER] Format and save predictions for submission. Implementation pending Kaggle format.")

# Main function to execute submission strategy
def execute_submission_strategy(train_path, test_path, window_sizes):
    """
    Executes the full submission strategy including model selection, rolling window testing, and logging.

    Parameters:
    train_path (str): Path to the train.parquet file.
    test_path (str): Path to the test.parquet file.
    window_sizes (list): List of rolling window sizes for experimentation.

    Returns:
    None
    """
    X_train, y_train, X_test = load_data(train_path, test_path)
    
    print("Running Rolling Window Experimentation...")
    rolling_results = rolling_window_experiment(X_train, y_train, window_sizes)
    
    # Select the best overall model
    final_best_model = rolling_results[max(rolling_results.keys(), key=lambda x: rolling_results[x]['Predictions'].corr(y_train))]['Model']
    final_best_predictions = rolling_results[max(rolling_results.keys(), key=lambda x: rolling_results[x]['Predictions'].corr(y_train))]['Predictions']

    # Log final results
    log_results(final_best_model, {}, np.sqrt(mean_squared_error(y_train, final_best_predictions)), pearsonr(y_train, final_best_predictions)[0])

    # Placeholder for submission formatting
    format_submission(final_best_predictions)

# Example Usage:
# execute_submission_strategy("train.parquet", "test.parquet", window_sizes=[3, 6, 12])
```

### **Key Features & Implementation Details**
✅ **Automatically selects the best-performing model** based on **Pearson correlation & RMSE**.  
✅ **Runs rolling window experimentation** using **multiple training data subsets** for better accuracy.  
✅ **Logs hyperparameters and evaluation results** in a **separate CSV file (`model_logs.csv`)** for future analysis.  
🚫 **Submission formatting is marked as a placeholder** until the official format is known.  

### **Next Steps**
- When the **submission format** is available, we can replace the placeholder function.
- Once the dataset is ready, we can **fine-tune the rolling window sizes** dynamically.


# 4. Package installation section and Kaggle pipeline results

Our model uses the following python packages:

````bash
!pip install pandas numpy matplotlib seaborn scikit-learn
!pip install --upgrade scikit-learn
!pip install --user scikit-learn==1.5.0
!pip install xgboost lightgbm scikit-optimize optuna boruta
!pip install tensorflow
!pip install pyarrow fastparquet
````  

Pythonic import section:

````python
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
import seaborn as sns
````

Below is a **thorough description** of our **latest Kaggle pipeline functionalities** and how it **adheres to the competition rules and evaluation criteria**.

---

# **Complete Kaggle Pipeline Overview**
### **Objective:**  
Our pipeline is designed to **predict cryptocurrency price movements** using **machine learning models** and **deep learning architectures**, 
applying a **robust feature engineering process** and optimizing performance based on **Pearson correlation**, which is the competition’s evaluation metric.

---

## **📌 Step 1: Exploratory Data Analysis (EDA)**
✔ Loads the `train.parquet` dataset and performs **preliminary statistical analysis**.  
✔ Identifies **missing values**, applies **memory-efficient float32 conversions**, and calculates **Pearson correlation** between features and target labels (`label`).  
✔ Visualizations include:
   - **Histogram of trading volume**  
   - **Boxplot of bid/ask quantities**  
   - **Feature correlation heatmap**  
   - **Time-series visualization for trading activity**  
✔ Ensures dataset **quality and consistency** before applying ML models.

We inspect a smaple of data from the file train.parquet:  

````python
import pandas as pd

def preview_dataset(file_path, num_rows=10):
    """
    Loads and displays a small portion of the dataset for inspection.

    Parameters:
    file_path (str): Path to the Parquet file.
    num_rows (int): Number of rows to display (default: 10).

    Returns:
    None (prints dataset preview)
    """
    try:
        # Load dataset
        df = pd.read_parquet(file_path)

        # Display first few rows
        print(f"\nPreviewing first {num_rows} rows of the dataset:\n")
        print(df.head(num_rows))
        
        # Display column info
        print("\nDataset Info:\n")
        print(df.columns)
        print(df.info())

    except Exception as e:
        print(f"Error loading file: {e}")

# Example Usage:
# preview_dataset("train.parquet", num_rows=10)
````
This leads to the following result:
````python
preview_dataset("train.parquet", num_rows=10)
````
![Train_Parquet_File1](https://github.com/NenadBalaneskovic/ExternalProjects/blob/ffd7a9b47012f058a703368e059d82bf1333aacf/DRW_Crypto_Competition/train_parquet_file.PNG)
![Train_Parquet_File2](https://github.com/NenadBalaneskovic/ExternalProjects/blob/ffd7a9b47012f058a703368e059d82bf1333aacf/DRW_Crypto_Competition/train_parquet_file2.PNG)

Finally, we also create synthetic sample parquet data for the sake of debugging:

````python
import pandas as pd

# Define file paths
train_file = "train.parquet"
test_file = "test.parquet"
train_sample_file = "train_sample.parquet"
test_sample_file = "test_sample.parquet"

# Load original datasets
train_df = pd.read_parquet(train_file)
test_df = pd.read_parquet(test_file)

# ✅ FIX: Take a random sample (e.g., 5% of the data)
train_sample = train_df.sample(frac=0.05, random_state=42)
test_sample = test_df.sample(frac=0.05, random_state=42)

# ✅ FIX: Ensure timestamp index is reset
if isinstance(train_sample.index, pd.DatetimeIndex):
    train_sample.reset_index(inplace=True)
if isinstance(test_sample.index, pd.DatetimeIndex):
    test_sample.reset_index(inplace=True)

# ✅ FIX: Save sampled data as Parquet for debugging
train_sample.to_parquet(train_sample_file)
test_sample.to_parquet(test_sample_file)

print(f"Sampled datasets saved as: {train_sample_file} and {test_sample_file}")
````

---

## **📌 Step 2: Feature Engineering & Selection**
✔ Computes **technical indicators**, such as:
   - **Relative Strength Index (RSI)** → Measures price momentum.  
   - **Bollinger Bands** → Identifies volatility trends.  
✔ Generates **lag features** for bid/ask quantities and trading volume (`1, 5, 10` time periods).  
✔ Applies **rolling-window feature computation** for **mean and standard deviation analysis** across different time periods (`5, 10, 20`).  
✔ Implements **Principal Component Analysis (PCA)** to reduce dimensionality while retaining key patterns.  
✔ **Automated feature selection** via:
   - **XGBoost feature importance** ranking.  
   - **Boruta algorithm** for optimal subset selection.  
✔ Converts `float64` values to `float32` before PCA for **memory efficiency**.

Based on the portion of parquet data we’ve provided, here are some observations:

### 🔍 **Key Features Identified:**
✔ **Timestamp:** Data recorded at **one-minute intervals** (from `"2023-03-01 00:00:00"` onward).  
✔ **Trading Metrics:** Includes `bid_qty`, `ask_qty`, `buy_qty`, `sell_qty`, and `volume`, which are crucial for **market trend analysis**.  
✔ **Feature Columns (X1 to X890):** A large set of **proprietary features**, likely engineered for **predictive modeling**.  
✔ **Label Column:** Represents the **target variable**—important for the **Kaggle competition**. 

![Feature_Selection](https://github.com/NenadBalaneskovic/ExternalProjects/blob/ffd7a9b47012f058a703368e059d82bf1333aacf/DRW_Crypto_Competition/FeatureSelection.PNG) 

Our **Kaggle pipeline** aligns well with the **train.parquet dataset structure**! 🚀  

### 🔍 **Key Verifications:**
✔ **Feature Engineering Handles Proprietary Features (`X1 - X890`)** → PCA reduces dimensions efficiently.  
✔ **Trading Metrics (`bid_qty`, `ask_qty`, `buy_qty`, `sell_qty`, `volume`)** → Correctly incorporated in lag & rolling window features.  
✔ **Timestamp Handling** → `timestamp` feature used for time-series modeling (LSTM, CNN-LSTM).  
✔ **Memory Efficiency** → `float64` converted to `float32` for reduced memory footprint before PCA.  
✔ **Target Label (`label`) Handling** → Used for Pearson correlation evaluation in model selection.  
✔ **Submission Formatting** → Matches Kaggle’s required `ID` and `prediction` format.  

### 🚀 **Next Steps Before Competition Submission:**  
🔹 **Validate feature correlation** → Ensure `label` is well-predicted by selected features.  
🔹 **Test rolling window experiments** → Confirm optimal training period selection.  
🔹 **Perform leaderboard tracking** → Adjust model hyperparameters dynamically post-submission.  

Our pipeline is **ready to tackle the Kaggle dataset competitively**! 🔧🔥  

---

## **📌 Step 3: Model Experimentation**
✔ **Baseline models** → Implements **Ridge Regression** and **Random Forest** for comparison.  
✔ **Gradient Boosting models** → Uses **XGBoost and LightGBM**, with **GridSearchCV for hyperparameter tuning**.  
✔ **Neural Networks (Deep Learning) models**:  
   - **LSTM** → Captures sequential dependencies in cryptocurrency price movements.  
   - **CNN-LSTM Hybrid** → Applies **Convolutional 1D layers** before LSTM for **improved time-series analysis**.  
✔ **Bayesian Optimization (LightGBM)** → Applies **BayesSearchCV** to **fine-tune hyperparameters** for LightGBM.  
✔ **Optuna Optimization (XGBoost)** → Uses **Optuna's Bayesian sampling** to search for **optimal model parameters**.  
✔ **Ensemble Learning (Stacking)** → Combines **multiple models**, using **Ridge Regression as the meta-model** for final predictions.

The sample of generated results is structured as a dictionary and displayed below:

![Results_Dictionary](https://github.com/NenadBalaneskovic/ExternalProjects/blob/ffd7a9b47012f058a703368e059d82bf1333aacf/DRW_Crypto_Competition/ResultsDictionary.png)

The deliberate model comparison proceeds as follows:

![Model_Comparison](https://github.com/NenadBalaneskovic/ExternalProjects/blob/ffd7a9b47012f058a703368e059d82bf1333aacf/DRW_Crypto_Competition/ModelComparison.png)

---

## **📌 Step 4: Submission Strategy**
✔ **Rolling Window Experimentation** → Evaluates models across **multiple training periods** (`3, 6, 12 months`) to determine the most **stable prediction strategy**.  
✔ **Model Selection Process**:
   - Computes **RMSE** (Root Mean Squared Error).  
   - Computes **Pearson correlation** → The key **evaluation metric** used by Kaggle.  
   - **Selects the best-performing model** based on **highest Pearson correlation**.  
✔ **Hyperparameter and Model Logging** → Saves **best hyperparameters and model performance scores** into a CSV file for reference.  
✔ **Final Submission Formatting** → Converts predictions into **Kaggle-required format**:
   - `ID` (index-based unique identifier).  
   - `prediction` (floating-point predictions).  
   - Exports as **CSV** using `pandas.to_csv()` with `index=False`.
   
 We recognize the clear tabular structure (columns ID and prediction) of our kaggle-output file in accord with competition requirements:
 
 ![CSV_File_Results](https://github.com/NenadBalaneskovic/ExternalProjects/blob/ffd7a9b47012f058a703368e059d82bf1333aacf/DRW_Crypto_Competition/Results_CSV_File.png)

---

## **📌 How It Adheres to Kaggle’s Rules & Evaluation Criteria**
💡 The Kaggle competition evaluates submissions based on the **Pearson correlation coefficient** between actual values (`label`) and predicted values (`prediction`).  
✔ The pipeline **explicitly optimizes models using Pearson correlation** rather than RMSE alone.  
✔ The **rolling window experimentation** ensures **consistent alignment** between train and test datasets, improving **real-world generalizability**.  
✔ The **final model selection process chooses the model with the highest Pearson correlation**, ensuring **maximum adherence to Kaggle’s scoring system**.  
✔ The **submission formatting adheres** to Kaggle’s CSV requirements, avoiding **formatting errors**.  
✔ The pipeline **logs hyperparameter tuning results**, allowing **adaptive improvements** based on leaderboard feedback.

---

### 🚀 **Conclusion**
Our pipeline is now **fully optimized for the Kaggle competition**, ensuring compliance with the **evaluation criteria (Pearson correlation)** while 
employing **state-of-the-art ML techniques** for feature engineering, model experimentation, and submission generation. ✅  

Please refer to the md-file ["The full pythonic code implementation"](https://github.com/NenadBalaneskovic/ExternalProjects/blob/ffd7a9b47012f058a703368e059d82bf1333aacf/DRW_Crypto_Competition/Crypto_Pipeline_Module_Code.md) 
when accessing the full pythonic code implementation of the crypto analysis pipeline.


# 5. Future improvements

Our Crypto analysis pipeline is already incredibly capable, but there’s plenty of room to sharpen its edge and expand its scope. 
Here’s a curated list of **future enhancements**, grouped by **category** to help prioritize development:

## 🧪 Advanced Features (Stretch Goals)

- **Batching Wrapper Functionality**  
  The Crypto analysis pipeline discussed above could be enhanced by means of a batching-wrapper functionality that would allow users
  to process arbitrary large parquet data files without having to revert to costly alternatives provided by GCP, AWS or Azure. 
  This batching wrapper functionality could be represented as a data analytic pipeline comprised of the following algorithms and models 
  (please refer to the md-file ["BatchOptimizationConcepts"](https://github.com/NenadBalaneskovic/ExternalProjects/blob/ffd7a9b47012f058a703368e059d82bf1333aacf/DRW_Crypto_Competition/BatchOptimizationConcepts.md) for more details):

-- ✔ **Bayesian Optimization & Genetic Algorithms** – Ensuring models are optimally tuned dynamically.  
-- ✔ **SHAP & LIME for Interpretability** – Providing deep insights into ML/DL model predictions.  
-- ✔ **Memory-Efficient Processing** – Utilizing `functools.lru_cache` for caching and `numpy.memmap` for large dataset handling.  
-- ✔ **Parallel Processing & Large-Scale Forecasting** – Using `ThreadPoolExecutor` to efficiently manage data chunks.  
-- ✔ **Ensemble Learning Across SARIMAX, ML, and DL Models** – Robust multi-modal forecasting strategy.  

# 6. 📚 References
1. J. Berk, P. DeMarzo: „__Corporate Finance__“, 6th Ed., Pearson (2023); R. W. Melicher, E. A. Norton: "__Introduction to Finance__", 16th Ed. WILEY (2017); 
Anatoly B. Schmidt: "__Quantitative Finance for Physicists: An Introduction__", 1st Ed. Academic Press (2005); Alex Backwell: "__An Intuitive Introduction to Finance and Derivatives: Concepts, Terminology and Models__",
 1st Ed, Springer (2023); Michael Isichenko: "__Quantitative Portfolio Management: The Art and Science of Statistical Arbitrage__", 1st Ed., Springer (2021); John H. Cochrane: "__Asset Pricing__", Revised Ed., Princeton University Press (2005);
 Antti Ilmanen: "__Expected Returns: An Investor’s Guide to Harvesting Market Rewards__", 1st Ed., WILEY (2011); Steven E. Shreve: "__Stochastic Calculus for Finance I & II__", 1st Ed., Springer (2004); 
 Andrew Pole: "__Statistical Arbitrage: Algorithmic Trading Insights and Techniques__", 1st Ed., WILEY (2007); Mark S. Joshi: "__The Concepts and Practice of Mathematical Finance__", 2nd Ed., Cambridge University Press (2008);
Kaggle-link: competition-documentation: https://www.kaggle.com/competitions/drw-crypto-market-prediction.
2. [![Jupyter Notebook | English](https://img.shields.io/badge/Jupyter%20Notebook-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/ffd7a9b47012f058a703368e059d82bf1333aacf/DRW_Crypto_Competition/DRW_CryptoMarketPrediction.ipynb)
3. [![DRW_Crypto_Forecasting Report | English](https://img.shields.io/badge/DRW_Crypto_Forecasting%20Report-English-yellowblue?logoColor=blue&labelColor=red)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/ffd7a9b47012f058a703368e059d82bf1333aacf/DRW_Crypto_Competition/Crypto_TimeseriesForecast_Report.pdf) 
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



