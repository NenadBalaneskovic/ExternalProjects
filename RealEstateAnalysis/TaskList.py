# Task 0 - package installations
!pip install pymongo
!pip install pandas 
!pip install seaborn
!pip install folium 
!pip install matplotlib
!pip install sompy
!pip show sompy
!pip install numpy scipy matplotlib
!pip install minisom
!pip install somoclu

import pymongo
import pandas as pd
import seaborn as sns
import folium
import matplotlib.pyplot as plt
from minisom import MiniSom
# --------------------

# Task 1 - synthetic data generation  

import json
import random
import datetime

# Define possible values for each category
districts = ["Costal", "Central", "Mercantil", "Faunarium", "Technica", "Academia", "Historica"]
property_types = ["Luxury Villa", "Apartment", "Office", "Stock Exchange Building", "Park Estate", "Tech Campus", "Historical Property"]
buyer_types = ["Individual", "Corporation", "Developer", "Investment Firm"]
seller_types = ["Private Owner", "Government", "Corporation", "Developer"]
conditions = ["New", "Renovated", "Needs Repair", "Abandoned"]
economic_status = ["Boom", "Stable", "Recession"]

def generate_transaction(transaction_id):
    """Generates a single synthetic real estate transaction"""
    district = random.choice(districts)
    date = datetime.datetime.strptime(f"{random.randint(1900, 2025)}-{random.randint(1, 12)}-{random.randint(1, 28)}", "%Y-%m-%d").isoformat()
    property_type = random.choice(property_types)
    price = round(random.uniform(50_000, 10_000_000), 2)  # Randomized realistic price ranges
    buyer_type = random.choice(buyer_types)
    seller_type = random.choice(seller_types)
    square_meters = round(random.uniform(50, 1000), 1)  # Random property size
    condition = random.choice(conditions)
    economic_status_value = random.choice(economic_status)
    historical_significance = district == "Historica"

    return {
        "_id": f"TXN{transaction_id:08d}",
        "date": date,
        "district": district,
        "property_type": property_type,
        "price": price,
        "buyer_type": buyer_type,
        "seller_type": seller_type,
        "square_meters": square_meters,
        "condition": condition,
        "economic_status": economic_status_value,
        "historical_significance": historical_significance
    }

def generate_dataset(num_entries=70000, output_file="flower_hill_real_estate.json"):
    """Generates a dataset of synthetic real estate transactions"""
    dataset = [generate_transaction(i) for i in range(1, num_entries + 1)]
    
    # Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
    
    print(f"✅ Dataset successfully generated with {num_entries} entries and saved to {output_file}")

# Run dataset generation
generate_dataset()

# --------------------
# Task 2 - MongoDB data set extraction

import json
import pandas as pd
from pymongo import MongoClient

# Connection to MongoDB (Update with your own credentials if needed)
client = MongoClient("mongodb://localhost:27017/")  # Local MongoDB server
db = client["flower_hill_db"]  # Database name
collection = db["transactions"]  # Collection name

def load_json_to_mongodb(json_file="flower_hill_real_estate.json"):
    """Loads the real estate dataset from JSON file into MongoDB."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Remove duplicate "_id" fields to avoid insertion errors
    for record in data:
        record.pop("_id", None)  # Allow MongoDB to auto-generate unique IDs
    
    # Insert data while skipping duplicates
    try:
        collection.insert_many(data, ordered=False)  # Insert with duplicate handling
        print(f"✅ Successfully inserted {len(data)} records into MongoDB.")
    except Exception as e:
        print(f"❌ Error inserting data: {e}")

def extract_data_to_pandas(task_number):
    """Extracts dataset subsets based on the predefined 15 analytical tasks."""
    
    queries = {
        1: {"district": {"$exists": True}},  # Extract all districts for price trends
        2: {"economic_status": {"$exists": True}},  # Analyze market peaks & dips
        3: {"buyer_type": {"$exists": True}},  # Buyer segmentation
        4: {"district": {"$exists": True}, "condition": {"$exists": True}},  # Predict property values
        5: {"date": {"$exists": True}},  # Forecast transaction volume
        6: {"square_meters": {"$gte": 50}, "price": {"$gte": 100000}},  # District classification
        7: {"price": {"$exists": True}, "district": {"$exists": True}},  # Growth/stagnation clustering
        8: {"date": {"$exists": True}, "district": {"$exists": True}},  # Urban expansion analysis
        9: {"buyer_type": {"$exists": True}, "district": {"$exists": True}},  # Corporate vs individual buyers
        10: {"seller_type": {"$exists": True}},  # Predict property resale trends
        11: {"district": {"$exists": True}, "price": {"$gte": 500000}},  # Identify booming districts
        12: {"district": "Faunarium"},  # Parks & green spaces impact on prices
        13: {"district": "Technica"},  # Tech district influence on real estate
        14: {"economic_status": {"$exists": True}},  # Global economic cycle impact
        15: {"date": {"$exists": True}},  # Forecast new residential zones
    }
    
    if task_number not in queries:
        print(f"❌ Task {task_number} is not defined.")
        return None

    cursor = collection.find(queries[task_number])  # Execute query
    data = list(cursor)  # Convert query results into a list of dictionaries
    df = pd.DataFrame(data)  # Convert to Pandas DataFrame

    print(f"✅ Extracted data for Task {task_number} ({len(df)} records)")
    return df

# Load dataset into MongoDB
load_json_to_mongodb()

# Extract data subsets and store them in CSV files
for iter in range(15):
    df_task = extract_data_to_pandas(iter + 1)
    
    if df_task is not None and not df_task.empty:  # Ensure valid extraction
        file_name = f"flower_hill_data_task_{iter + 1}.csv"
        df_task.to_csv(file_name, index=False, encoding="utf-8")
        print(f"✅ Saved {file_name} ({len(df_task)} records)")
    else:
        print(f"❌ No data extracted for Task {iter + 1}")
        
# --------------------
# Task 3 - Predict district-wide property price trends using historical sales data

import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Load extracted dataset for Task 1
df = pd.read_csv("flower_hill_data_task_1.csv")

# Convert date column to datetime format
df["date"] = pd.to_datetime(df["date"])

# Group by district and resample by year to analyze yearly price trends
df_grouped = df.groupby(["district"]).resample("Y", on="date").mean()

# Reset index for clarity
df_grouped = df_grouped.reset_index()

# Preview structured data
print(df_grouped.head())

# Load extracted dataset for Task 1
df = pd.read_csv("flower_hill_data_task_1.csv")

# Convert date column to datetime format
df["date"] = pd.to_datetime(df["date"])

# Group data by district and resample yearly to analyze price trends
df_grouped = df.groupby(["district"]).resample("Y", on="date").mean()

# Reset index for clarity
df_grouped = df_grouped.reset_index()

# Plot price trends using Seaborn
plt.figure(figsize=(14, 7))
sns.lineplot(data=df_grouped, x="date", y="price", hue="district", marker="o")

# Customize plot
plt.title("📈 District-Wide Property Price Trends in Flower Hill", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average Price", fontsize=14)
plt.legend(title="District", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.grid(True)

# Show plot
plt.show()

# SARIMAX analysis

# Load district-wise data
df = pd.read_csv("flower_hill_data_task_1.csv")

# Convert date column to datetime format
df["date"] = pd.to_datetime(df["date"])

# Set date as the index
df.set_index("date", inplace=True)

# Select a specific district (e.g., "Costal") for forecasting
district_name = "Costal"  # You can choose other districts dynamically
df_district = df[df["district"] == district_name].resample("Y").mean()

# Preview dataset
print(df_district.head())

# Load district-wise dataset
df = pd.read_csv("flower_hill_data_task_1.csv")

# Convert date to datetime format
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Choose a district to analyze (e.g., "Costal")
district_name = "Costal"
df_district = df[df["district"] == district_name].resample("Y").mean()

# Perform Augmented Dickey-Fuller Test
adf_test = adfuller(df_district["price"])
print(f"ADF Statistic: {adf_test[0]}")
print(f"P-Value: {adf_test[1]}")

# Interpretation
if adf_test[1] > 0.05:
    print("🔴 The time series is NON-STATIONARY (p > 0.05) – Differencing needed!")
else:
    print("🟢 The time series is STATIONARY (p < 0.05) – Ready for modeling!")

# Apply seasonal decomposition
decomposition = seasonal_decompose(df_district["price"], model="additive", period=1)

# Plot components
plt.figure(figsize=(12, 8))
decomposition.plot()
plt.suptitle(f"📊 Seasonal Decomposition of Property Prices in {district_name}", fontsize=16)
plt.show()

# Load district-wise dataset
df = pd.read_csv("flower_hill_data_task_1.csv")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Select a district (e.g., "Costal") for forecasting
district_name = "Costal"
df_district = df[df["district"] == district_name].resample("Y").mean()

# Fit SARIMAX model (without differencing since data is stationary)
sarimax_model = SARIMAX(df_district["price"], order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))
sarimax_fit = sarimax_model.fit()

# Forecast for the next 5 years
forecast_steps = 5
forecast = sarimax_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(df_district.index[-1], periods=forecast_steps + 1, freq="Y")[1:]

# Plot forecast
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df_district.index, df_district["price"], label="Historical Prices", marker="o")
plt.plot(forecast_index, forecast.predicted_mean, label="Forecasted Prices", linestyle="dashed", marker="o", color="red")
plt.fill_between(forecast_index, forecast.conf_int().iloc[:, 0], forecast.conf_int().iloc[:, 1], color="red", alpha=0.2)
plt.title(f"📈 SARIMAX Forecast for {district_name} Property Prices", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average Price", fontsize=14)
plt.legend()
plt.grid()
plt.show()

# --------------------
# Task 4 - Analyze market dips & peaks based on economic conditions

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load task 2 dataset
df = pd.read_csv("flower_hill_data_task_2.csv")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Select district (e.g., "Costal") for analysis
district_name = "Costal"
df_district = df[df["district"] == district_name].resample("Y").mean()

# Detect market peaks & dips based on percentage price change
df_district["price_change"] = df_district["price"].pct_change() * 100
df_district["peak"] = df_district["price_change"] > 5
df_district["dip"] = df_district["price_change"] < -5

# Plot price trends & highlight peaks/dips
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_district, x=df_district.index, y="price", label="Price Trend", marker="o")
plt.scatter(df_district[df_district["peak"]].index, df_district[df_district["peak"]]["price"], color="green", label="Market Peak", marker="^", s=100)
plt.scatter(df_district[df_district["dip"]].index, df_district[df_district["dip"]]["price"], color="red", label="Market Dip", marker="v", s=100)

plt.title(f"📈 Market Peaks & Dips in {district_name} (Updated with Task 2 Data)", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average Price", fontsize=14)
plt.legend()
plt.grid()
plt.show()

# --------------------
# Task 5 - Segment buyers by their purchasing behavior (individual vs. corporate)

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("flower_hill_data_task_3.csv")

# Convert date column to datetime format
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Count transactions by buyer type
buyer_counts = df["buyer_type"].value_counts()

# Plot buyer segmentation
plt.figure(figsize=(8, 6))
buyer_counts.plot(kind="bar", color=["blue", "orange"])
plt.title("📊 Individual vs. Corporate Buyer Segmentation", fontsize=16)
plt.xlabel("Buyer Type", fontsize=14)
plt.ylabel("Transaction Count", fontsize=14)
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.show()

# --------------------
# Task 6 - Predict future property values based on district, condition, and economy

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

# Load dataset
df = pd.read_csv("flower_hill_data_task_4.csv")

# Convert date column to datetime format
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Select relevant features
df_filtered = df[["district", "property_type", "price", "condition", "economic_status"]]

# Check for missing values
print(df_filtered.isnull().sum())

# Load dataset
df = pd.read_csv("flower_hill_data_task_4.csv")

# Convert date column to datetime format
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Select a district for forecasting (e.g., "Costal")
district_name = "Costal"
df_district = df[df["district"] == district_name].groupby("date").agg({"price": "mean", "economic_status": "first", "condition": "first"})  # Using 'first' for categorical data

# Convert categorical economic_status to numeric values
economic_mapping = {"Stable": 0, "Declining": -1, "Growing": 1}
df_district["economic_status"] = df_district["economic_status"].map(economic_mapping)

# Convert categorical condition to numeric values if needed
condition_mapping = {"New": 2, "Renovated": 1, "Old": 0}
df_district["condition"] = df_district["condition"].map(condition_mapping)

# Check for any remaining non-numeric values
print(df_district.dtypes)

# Drop any rows with missing data (optional but helpful)
df_district.dropna(inplace=True)

# Fit SARIMAX model with economic_status & condition as exogenous variables
sarimax_model = SARIMAX(df_district["price"], order=(1, 0, 1), seasonal_order=(1, 0, 1, 12),
                         exog=df_district[["economic_status", "condition"]])
sarimax_fit = sarimax_model.fit()

# Forecast for the next 5 years
forecast_steps = 5

# Expand the last available values for the next 5 years (correcting shape issue)
last_values = df_district[["economic_status", "condition"]].iloc[-1].values.reshape(1, -1)  # Get last row
future_exog = np.repeat(last_values, forecast_steps, axis=0)  # Duplicate it 5 times
print(future_exog.shape)  # Should be (5,2)

# Generate forecast with correctly shaped exogenous variables
forecast = sarimax_fit.get_forecast(steps=forecast_steps, exog=future_exog)
forecast_index = pd.date_range(df_district.index[-1], periods=forecast_steps + 1, freq="Y")[1:]

# Plot forecast results
plt.figure(figsize=(12, 6))
plt.plot(df_district.index, df_district["price"], label="Historical Prices", marker="o")
plt.plot(forecast_index, forecast.predicted_mean, label="Forecasted Prices", linestyle="dashed", marker="o", color="red")
plt.fill_between(forecast_index, forecast.conf_int().iloc[:, 0], forecast.conf_int().iloc[:, 1], color="red", alpha=0.2)
plt.title(f"📈 Property Price Forecast for {district_name}", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average Price", fontsize=14)
plt.legend()
plt.grid()
plt.show()

# --------------------
# Task 7 - Forecast the number of purchased real estate items in the future

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("flower_hill_data_task_5.csv")

# Convert date column to datetime format
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Aggregate number of real estate purchases per year
df_transactions = df.groupby("date").size().to_frame("transactions")

# Plot historical trend of real estate purchases
plt.figure(figsize=(12, 6))
plt.plot(df_transactions.index, df_transactions["transactions"], marker="o", label="Real Estate Transactions")
plt.title("📊 Historical Trend of Real Estate Purchases", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Number of Transactions", fontsize=14)
plt.legend()
plt.grid()
plt.show()

# Load dataset
df = pd.read_csv("flower_hill_data_task_5.csv")

# Convert date column to datetime format
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Aggregate number of real estate purchases per year
df_transactions = df.groupby("date").size().to_frame("transactions")

# Standardize transaction data to prevent scaling issues
scaler = StandardScaler()
df_transactions["transactions"] = scaler.fit_transform(df_transactions[["transactions"]])

# Fit SARIMAX model with optimized parameters
sarimax_model = SARIMAX(df_transactions["transactions"], order=(1, 0, 0), seasonal_order=(1, 0, 1, 12))
sarimax_fit = sarimax_model.fit()

# Forecast for the next 5 years
forecast_steps = 5

# Correct forecast input shape: Expand last value for 5 years
last_values = df_transactions["transactions"].iloc[-1]
future_exog = np.full((forecast_steps, 1), last_values)  # Ensure (5,1) shape
print(future_exog.shape)  # Should be (5,1)

# Generate forecast
forecast = sarimax_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(df_transactions.index[-1], periods=forecast_steps + 1, freq="Y")[1:]

# Plot forecast results
plt.figure(figsize=(12, 6))
plt.plot(df_transactions.index, df_transactions["transactions"], label="Historical Transactions", marker="o")
plt.plot(forecast_index, forecast.predicted_mean, label="Forecasted Transactions", linestyle="dashed", marker="o", color="red")
plt.fill_between(forecast_index, forecast.conf_int().iloc[:, 0], forecast.conf_int().iloc[:, 1], color="red", alpha=0.2)
plt.title("📈 Forecasted Real Estate Transactions", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Number of Transactions (Standardized)", fontsize=14)
plt.legend()
plt.grid()
plt.show()

# --------------------
# Task 8 - Predict to which district a new real estate belongs based on its features

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("flower_hill_data_task_6.csv")

# Select relevant features for prediction
features = ["square_meters", "buyer_type", "property_type", "condition", "economic_status", "historical_significance"]
target = "district"

# Verify column presence
print(df.columns)

# Convert categorical buyer_type to numeric values
buyer_mapping = {"Individual": 0, "Investment Firm": 1, "LLC": 2, "Corporate": 3}  # Adjust as needed
df["buyer_type"] = df["buyer_type"].map(buyer_mapping)

# Apply One-Hot Encoding for categorical features
df_encoded = pd.get_dummies(df, columns=["property_type", "condition", "economic_status", "historical_significance"])

# Ensure encoded columns are correct
print(df_encoded.columns)

# Convert numeric columns to proper float format
numeric_columns = df_encoded.select_dtypes(include=[np.number]).columns
df_encoded[numeric_columns] = df_encoded[numeric_columns].apply(pd.to_numeric, errors="coerce")

# Handle missing values
df_encoded.fillna(df_encoded.median(), inplace=True)

# Remove infinite values
df_encoded.replace([np.inf, -np.inf], np.nan, inplace=True)
df_encoded.dropna(inplace=True)

# Apply clipping only to numeric columns
df_encoded[numeric_columns] = df_encoded[numeric_columns].clip(-1e6, 1e6)  # Prevent extreme values

# Split data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(df_encoded.drop(columns=["district", "_id", "date", "price", "seller_type"]), 
                                                    df_encoded[target], test_size=0.2, random_state=42)

# Standardize numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train district classification model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Example new property for prediction
new_property = pd.DataFrame({
    "square_meters": [120],
    "buyer_type": [1],  # Investment Firm (mapped from categorical)
    "property_type_Apartment": [1],  # One-hot encoded categorical variable
    "condition_New": [1],
    "economic_status_Growing": [1],
    "historical_significance_Low": [1]
}, index=[0])

# Ensure new property columns match training set
missing_cols = set(df_encoded.columns) - set(new_property.columns)
for col in missing_cols:
    new_property[col] = 0  # Fill missing columns with zeros

new_property = new_property[df_encoded.drop(columns=["district", "_id", "date", "price", "seller_type"]).columns]  # Align column order

# Standardize new property features
new_property_scaled = scaler.transform(new_property)

# Predict district for new real estate
predicted_district = model.predict(new_property_scaled)
print(f"🏡 Predicted District: {predicted_district[0]}")

def predict_district(new_property_data, model, scaler, df_encoded):
    """
    Predicts the district of a new real estate property based on its features.

    Parameters:
    - new_property_data (dict): Dictionary containing feature values for a new property.
    - model (RandomForestClassifier): Trained Random Forest model.
    - scaler (StandardScaler): Fitted standardization scaler.
    - df_encoded (DataFrame): Encoded dataset used for training, ensuring correct feature alignment.

    Returns:
    - str: Predicted district for the new property.
    """

    import pandas as pd

    # Convert dictionary input to DataFrame
    new_property = pd.DataFrame(new_property_data, index=[0])

    # Ensure new property includes ALL feature columns from training data
    missing_cols = set(df_encoded.drop(columns=["district", "_id", "date", "price", "seller_type"]).columns) - set(new_property.columns)
    for col in missing_cols:
        new_property[col] = 0  # Fill missing categorical features with zeros

    # Arrange columns in the same order as the training set
    new_property = new_property[df_encoded.drop(columns=["district", "_id", "date", "price", "seller_type"]).columns]

    # Standardize new property features
    new_property_scaled = scaler.transform(new_property)

    # Predict district
    predicted_district = model.predict(new_property_scaled)
    
    return f"🏡 Predicted District: {predicted_district[0]}"
    
# Example usage
new_property_data = {
    "square_meters": [120],
    "buyer_type": [1],  # Investment Firm (mapped from categorical)
    "property_type_Apartment": [1],  # One-hot encoded categorical variable
    "condition_New": [1],
    "economic_status_Growing": [1],
    "historical_significance_Low": [1]
}

predicted_district = predict_district(new_property_data, model, scaler, df_encoded)
print(predicted_district)

# --------------------
# Task 9 - Cluster districts by real estate trends (growth vs. stagnation)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from somoclu import Somoclu  # Install via: `pip install somoclu`

# Load dataset
df = pd.read_csv("flower_hill_data_task_7.csv")

# Aggregate data by district for clustering insights
df_grouped = df.groupby("district").agg({
    "price": "mean", 
    "_id": "count",  # Use row count as an estimate for transaction volume
    "economic_status": "first"  # Take the first entry to avoid concatenated values
}).reset_index()

# Rename column for clarity
df_grouped.rename(columns={"_id": "transactions"}, inplace=True)

# Extract the most relevant economic status using regex
df_grouped["economic_status"] = df_grouped["economic_status"].str.extract(r"(Stable|Recession|Boom|Growing|Declining)")

# Map cleaned categorical values to numeric scores
economic_mapping = {"Stable": 0, "Recession": -2, "Declining": -1, "Growing": 1, "Boom": 2}
df_grouped["economic_status"] = df_grouped["economic_status"].map(economic_mapping)

# Verify conversion
print(df_grouped["economic_status"].unique())  # Should now contain numeric values only

# Standardize features for clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_grouped.drop(columns=["district"]))

# Apply PCA for better visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Apply K-Means clustering
num_clusters = 2  # Growth vs. Stagnation
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df_grouped["Cluster"] = kmeans.fit_predict(df_scaled)

# Visualize clustering results
plt.figure(figsize=(10, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_grouped["Cluster"], cmap="coolwarm", marker="o")
plt.title("🏙️ District Clustering: Growth vs. Stagnation", fontsize=16)
plt.xlabel("PCA Component 1", fontsize=14)
plt.ylabel("PCA Component 2", fontsize=14)
plt.colorbar(label="Cluster Group")
plt.grid()
plt.show()

# SOM-analysis version

# Load dataset
df = pd.read_csv("flower_hill_data_task_7.csv")

# Aggregate data by district
df_grouped = df.groupby("district").agg({
    "price": "mean", 
    "square_meters": "sum",  
    "economic_status": "first"  # First recorded economic status per district
}).reset_index()

# Encode categorical economic status numerically
economic_mapping = {"Stable": 0, "Recession": -2, "Declining": -1, "Growing": 1, "Boom": 2}
df_grouped["economic_status"] = df_grouped["economic_status"].map(economic_mapping)

# Standardize features for clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_grouped.drop(columns=["district"]))

# Convert dataset to float32 to prevent SOM errors
df_scaled = np.array(df_scaled, dtype=np.float32)

# Adjust SOM grid size for optimal clustering
n_rows, n_cols = 10, 10  # Larger grid for better cluster separation
som = Somoclu(n_columns=n_cols, n_rows=n_rows)

# Train SOM model by passing data directly
som.train(data=df_scaled, epochs=1000)

# Force clustering explicitly AFTER training
som.cluster()

# Manually track cluster assignments using BMU indices
cluster_map = {idx: (row, col) for idx, (row, col) in zip(df_grouped.index.to_numpy(), som.bmus)}

# Assign clusters using manually extracted cluster_map
df_grouped["Cluster"] = df_grouped.index.to_series().apply(lambda idx: cluster_map.get(idx, (-1, -1)))

# Debugging: Print verified cluster assignments
print(df_grouped[["district", "Cluster"]].head())

# Plot SOM clusters
plt.figure(figsize=(10, 6))
plt.scatter(df_grouped["price"], df_grouped["square_meters"], c=df_grouped["Cluster"].apply(lambda x: x[0]), cmap="coolwarm", marker="o")
plt.title("🏙️ SOM-Based District Clustering: Growth vs. Stagnation", fontsize=16)
plt.xlabel("Average Price", fontsize=14)
plt.ylabel("Total Square Meters Developed", fontsize=14)
plt.colorbar(label="Cluster Group")
plt.grid()
plt.show()

# --------------------
# Task 10 - Analyze urban expansion & contraction patterns over time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("flower_hill_data_task_8.csv")

# Convert date column to datetime format
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

# Aggregate urban data by district and year
df_grouped = df.groupby(["district", df.index.year]).agg({
    "square_meters": "sum",  # Total built area per district per year
    "price": "mean",  # Average property price per year
    "economic_status": "first"  # Take first available economic status per year
}).reset_index()

# Calculate year-over-year growth in built area
df_grouped["expansion_rate"] = df_grouped.groupby("district")["square_meters"].pct_change() * 100  # Percentage change

# Determine expansion vs. contraction zones
df_grouped["trend"] = np.where(df_grouped["expansion_rate"] > 0, "Expansion", "Contraction")

# Visualize expansion trends over time
plt.figure(figsize=(12, 6))
for district in df_grouped["district"].unique():
    subset = df_grouped[df_grouped["district"] == district]
    plt.plot(subset["district"], subset["expansion_rate"], marker="o", label=district)

plt.title("🏙️ Urban Expansion vs. Contraction Over Time", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Expansion Rate (%)", fontsize=14)
plt.legend()
plt.grid()
plt.show()

# Linear regression forecasting

# Load dataset
df = pd.read_csv("flower_hill_data_task_8.csv")

# Convert date column to datetime format
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year  # Extract correct year values

# Aggregate urban data by district and year
df_grouped = df.groupby(["district", "year"]).agg({
    "square_meters": "sum",  # Total built area per district per year
    "price": "mean",  # Average property price per year
    "economic_status": "first"  # First available economic status per year
}).reset_index()

# Encode categorical features (district & economic_status)
categorical_columns = ["district", "economic_status"]
label_encoders = {}

for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    df_grouped[col] = label_encoders[col].fit_transform(df_grouped[col])

# Calculate year-over-year growth in built area
df_grouped["expansion_rate"] = df_grouped.groupby("district")["square_meters"].pct_change() * 100  # Percentage change

# Fill missing values (e.g., first year with no previous data)
df_grouped["expansion_rate"].fillna(0, inplace=True)

# Standardize numeric features
scaler = StandardScaler()
df_grouped[["square_meters", "price", "expansion_rate"]] = scaler.fit_transform(df_grouped[["square_meters", "price", "expansion_rate"]])

# Train expansion trend prediction model (Linear Regression)
X = df_grouped[["year", "economic_status"]].values  # Correct year format used for predictions
y = df_grouped["expansion_rate"].values

model = LinearRegression()
model.fit(X, y)

# Predict future expansion trends for correct years
future_years = np.array([[2026, 1], [2027, 1], [2028, 1], [2029, 1], [2030, 1]])  # Example future economic status
predicted_expansion = model.predict(future_years)

# Visualize historical & future expansion trends
plt.figure(figsize=(12, 6))
plt.scatter(df_grouped["year"], df_grouped["expansion_rate"], marker="o", label="Historical Expansion Rate")
plt.plot(future_years[:, 0], predicted_expansion, marker="o", linestyle="dashed", color="red", label="Predicted Expansion")
plt.title("🌆 Historical & Predicted Urban Expansion Trends", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Expansion Rate (%)", fontsize=14)
plt.legend()
plt.grid()
plt.show()

# --------------------
# Task 11 - Identify dominant buyer types in different districts (corporations vs. individuals)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from somoclu import Somoclu  # Install via: `pip install somoclu`

# Load dataset
df = pd.read_csv("flower_hill_data_task_9.csv")

# Aggregate buyer types per district
df_grouped = df.groupby(["district", "buyer_type"]).size().reset_index(name="transaction_count")

# Calculate percentage of transactions per buyer type in each district
df_grouped["buyer_share"] = df_grouped.groupby("district")["transaction_count"].transform(lambda x: x / x.sum() * 100)

# Pivot data for better visualization
df_pivot = df_grouped.pivot(index="district", columns="buyer_type", values="buyer_share").fillna(0)

# Verify buyer share distribution
print(df_pivot.head())

# Plot buyer dominance per district
ax = df_pivot.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="coolwarm")
plt.title("🏡 Market Control by Buyer Type in Flower Hill Districts", fontsize=16)
plt.xlabel("District", fontsize=14)
plt.ylabel("Transaction Share (%)", fontsize=14)
plt.legend(title="Buyer Type")
plt.grid()
plt.show()

# SOM-analysis version

# Load dataset for Task 9
df = pd.read_csv("flower_hill_data_task_9.csv")

# Aggregate data by district using available columns
df_grouped = df.groupby("district").agg({
    "price": "mean", 
    "square_meters": "sum",  
    "condition": "first",  # First recorded condition per district
    "economic_status": "first",  # First recorded economic status per district
    "historical_significance": "first"  # Consider historical significance for clustering
}).reset_index()

# Select only numeric columns before applying median fill
numeric_columns = df_grouped.select_dtypes(include=[np.number]).columns
df_grouped[numeric_columns] = df_grouped[numeric_columns].replace([np.inf, -np.inf], np.nan)  # Replace infinities
df_grouped[numeric_columns] = df_grouped[numeric_columns].fillna(df_grouped[numeric_columns].median())  # Fill NaN safely

# Encode categorical economic status numerically
economic_mapping = {"Stable": 0, "Recession": -2, "Declining": -1, "Growing": 1, "Boom": 2}
df_grouped["economic_status"] = df_grouped["economic_status"].map(economic_mapping)

# Encode property condition numerically
condition_mapping = {"Poor": -2, "Fair": -1, "Good": 0, "Very Good": 1, "Excellent": 2}
df_grouped["condition"] = df_grouped["condition"].map(condition_mapping)

# Encode historical significance numerically (if applicable)
df_grouped["historical_significance"] = df_grouped["historical_significance"].astype("category").cat.codes

# Standardize features for clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_grouped[numeric_columns])  # Apply scaling only to numeric columns

# Convert dataset to float32 to prevent SOM errors
df_scaled = np.array(df_scaled, dtype=np.float32)

# Adjust SOM grid size and initialization method for optimal clustering
n_rows, n_cols = 20, 20  # Increased grid size for better distribution
som = Somoclu(n_columns=n_cols, n_rows=n_rows, initialization="random", compactsupport=False)

# Train SOM model with corrected function call
som.train(data=df_scaled, epochs=1500)

# Debugging: Check BMU assignments
if som.bmus is None:
    raise ValueError("BMU assignment failed—check training parameters!")

# Assign clusters directly from BMU array
df_grouped["Cluster_Row"] = np.array(som.bmus)[:, 0]  # Extract row indices
df_grouped["Cluster_Col"] = np.array(som.bmus)[:, 1]  # Extract column indices

# Verify extracted cluster coordinates
print(df_grouped[["district", "Cluster_Row", "Cluster_Col"]].head())

# Improved cluster color mapping for visualization
plt.figure(figsize=(10, 6))
plt.scatter(df_grouped["price"], df_grouped["square_meters"], 
            c=df_grouped["Cluster_Row"] * n_cols + df_grouped["Cluster_Col"], 
            cmap="coolwarm", marker="o")
plt.title("🏙️ SOM-Based District Clustering for Task 9", fontsize=16)
plt.xlabel("Average Price", fontsize=14)
plt.ylabel("Total Square Meters Developed", fontsize=14)
plt.colorbar(label="Cluster Group")
plt.grid()
plt.show()

# --------------------
# Task 12 - Predict whether a property will be resold within a certain timeframe 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("flower_hill_data_task_10.csv")

# Convert date column to datetime format
df["date"] = pd.to_datetime(df["date"])
df.sort_values(by="date", inplace=True)  # Ensure chronological order

# Identify resale likelihood using district and property type as tracking proxy
df["resale_flag"] = df.groupby(["district", "property_type"])["date"].diff().dt.days <= (5 * 365)  # Resale within 5 years

# Convert resale_flag to binary (0 = No Resale, 1 = Resold)
df["resale_flag"] = df["resale_flag"].fillna(False).astype(int)

# Select relevant features for prediction
features = ["square_meters", "buyer_type", "property_type", "condition", "economic_status", "price"]
target = "resale_flag"

# Apply One-Hot Encoding for categorical features
df_encoded = pd.get_dummies(df, columns=["property_type", "condition", "economic_status", "buyer_type"])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_encoded.drop(columns=["resale_flag", "_id", "date", "district", "seller_type"]), 
                                                    df_encoded[target], test_size=0.2, random_state=42)

# Standardize numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train resale likelihood prediction model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Example new property for resale prediction
new_property = pd.DataFrame({
    "square_meters": [120],
    "buyer_type_Corporate": [1],  # One-hot encoded categorical variable
    "property_type_Apartment": [1],
    "condition_Renovated": [1],
    "economic_status_Growing": [1],
    "price": [450000]  # Example property price
}, index=[0])

# Ensure new property columns match training set
missing_cols = set(df_encoded.columns) - set(new_property.columns)
for col in missing_cols:
    new_property[col] = 0  # Fill missing categorical features with zeros

# Align column order with training data
new_property = new_property[df_encoded.drop(columns=["resale_flag", "_id", "date", "district", "seller_type"]).columns]

# Standardize new property features
new_property_scaled = scaler.transform(new_property)

# Predict resale likelihood
resale_probability = model.predict_proba(new_property_scaled)[:, 1]
print(f"🔄 Resale Probability within 5 years: {resale_probability[0]:.2f}")

# --------------------
# Task 13 - Predict which district will see the highest appreciation in the next decade
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def analyze_time_series_regression(df, time_column, target_column, features=None, model_type="linear"):
    """
    Perform time-series analysis, regression modeling, and comparative analysis.

    Parameters:
    df (pd.DataFrame): The dataset containing time-series and feature columns.
    time_column (str): The column representing time-series data.
    target_column (str): The dependent variable for regression.
    features (list): List of independent variables (optional, auto-selected if None).
    model_type (str): Model selection ("linear", "ridge", "lasso", "random_forest").

    Returns:
    dict: Results including time-series trends, regression model evaluation, and comparative analysis.
    """
    results = {}

    # Print available columns
    print("Available columns:", df.columns)

    # Ensure target_column exists
    if target_column not in df.columns:
        raise KeyError(f"Column '{target_column}' not found! Please choose from: {df.columns}")

    # Auto-select numeric features if none are provided
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in features:
            features.remove(target_column)  # Exclude target variable from predictors

    # Ensure selected features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise KeyError(f"Features {missing_features} not found in dataset!")

    # Convert time column to datetime & sort dataset
    df[time_column] = pd.to_datetime(df[time_column])
    df.sort_values(by=time_column, inplace=True)

    # Convert categorical economic status to numeric values
    economic_mapping = {"Stable": 0, "Recession": -2, "Declining": -1, "Growing": 1, "Boom": 2}
    if "economic_status" in df.columns:
        df["economic_status"] = df["economic_status"].map(economic_mapping)

    # Convert property condition to numeric values
    condition_mapping = {"Poor": -2, "Fair": -1, "Good": 0, "Very Good": 1, "Excellent": 2}
    if "condition" in df.columns:
        df["condition"] = df["condition"].map(condition_mapping)

    # Ensure missing values are handled before integer conversion
    df["economic_status"].fillna(0, inplace=True)  # Replace NaN with default 0
    df["condition"].fillna(0, inplace=True)  # Replace NaN with default 0

    # Convert categorical variables to integer format
    df["economic_status"] = df["economic_status"].astype(int)
    df["condition"] = df["condition"].astype(int)

    # Apply smoothing using a rolling average to reduce overload
    df[target_column] = df[target_column].rolling(window=30).mean()

    # Time-Series Decomposition (after smoothing)
    decomposition = seasonal_decompose(df[target_column].dropna(), model="additive", period=12)
    results["time_series_trend"] = decomposition.trend.dropna()
    results["time_series_seasonality"] = decomposition.seasonal.dropna()
    results["time_series_residual"] = decomposition.resid.dropna()

    # Plot optimized decomposition
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax[0].plot(decomposition.trend[:300], label="Trend", alpha=0.8, linestyle="--", color="blue")
    ax[1].plot(decomposition.seasonal[:300], label="Seasonality", alpha=0.8, linestyle="--", color="green")
    ax[2].plot(decomposition.resid[:300], label="Residuals", alpha=0.6, linestyle="--", color="red")

    ax[0].set_title("📈 Improved Time-Series Decomposition")
    for a in ax: 
        a.legend()
        a.grid()

    plt.show()

    # Prepare data for regression
    X = df[features]
    y = df[target_column]
    
    # Handle missing values in numeric columns only
    X.fillna(X.select_dtypes(include=[np.number]).median(), inplace=True)
    y.fillna(y.median(), inplace=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select regression model
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.1),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    model = models.get(model_type, LinearRegression())
    model.fit(X_train, y_train)
    
    # Predictions & Evaluation
    y_pred = model.predict(X_test)
    results["MAE"] = mean_absolute_error(y_test, y_pred)
    results["MSE"] = mean_squared_error(y_test, y_pred)
    results["R2_Score"] = r2_score(y_test, y_pred)

    # Plot actual vs predicted
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Model Comparison: {model_type.capitalize()}")
    plt.grid()
    plt.show()

    # Comparative Analysis: Feature Importance
    if model_type == "random_forest":
        feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        results["feature_importance"] = feature_importances
        
        plt.figure(figsize=(10, 5))
        feature_importances.plot(kind="bar", title="Feature Importance (RandomForest)")
        plt.grid()
        plt.show()

    return results

# Example Usage:
df = pd.read_csv("flower_hill_data_task_11.csv")
result = analyze_time_series_regression(df, "date", "price", features=["square_meters", "economic_status"], model_type="random_forest")

# --------------------
# Task 14 - How does the presence of parks & lakes (Faunarium district) influence real estate prices?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def analyze_environmental_impact(df, district_column, price_column, features):
    """
    Assess the impact of environmental factors (historical significance, size) on real estate prices in the Faunarium district.

    Parameters:
    df (pd.DataFrame): The dataset containing district and property data.
    district_column (str): Column representing district names.
    price_column (str): Column representing real estate prices.
    features (list): Real estate factors affecting property values (e.g., 'square_meters', 'historical_significance').

    Returns:
    dict: Results including comparative price analysis, statistical significance, and regression insights.
    """
    results = {}

    # Extract Faunarium district data
    df_faunarium = df[df[district_column] == "Faunarium"]
    df_other = df[df[district_column] != "Faunarium"]

    # Compare average prices
    avg_price_faunarium = df_faunarium[price_column].mean()
    avg_price_other = df_other[price_column].mean()
    results["average_price_faunarium"] = avg_price_faunarium
    results["average_price_other_districts"] = avg_price_other

    # Perform statistical significance test (T-test)
    t_stat, p_value = ttest_ind(df_faunarium[price_column], df_other[price_column], nan_policy="omit")
    results["price_difference_p_value"] = p_value  # If p-value < 0.05, impact is statistically significant

    # Prepare data for regression analysis
    X = df[features]
    y = df[price_column]

    # Handle missing values
    X.fillna(X.median(), inplace=True)
    y.fillna(y.median(), inplace=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate regression model
    results["MAE"] = mean_absolute_error(y_test, y_pred)
    results["R2_Score"] = r2_score(y_test, y_pred)

    # Feature importance analysis
    feature_importances = pd.Series(model.coef_, index=features).sort_values(ascending=False)
    results["feature_importance"] = feature_importances

    # Visualization of price trends
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df[district_column], y=df[price_column])
    plt.title("Comparative Price Analysis: Faunarium vs. Other Districts")
    plt.ylabel("Real Estate Prices")
    plt.xlabel("Districts")
    plt.grid()
    plt.show()

    # Feature importance visualization
    plt.figure(figsize=(10, 5))
    feature_importances.plot(kind="bar", title="Impact of Historical Significance & Size on Price")
    plt.grid()
    plt.show()

    return results

# Example Usage:
df = pd.read_csv("flower_hill_data_task_12.csv")
result = analyze_environmental_impact(df, "district", "price", ["square_meters", "historical_significance"])

# --------------------
# Task 15 - Does technology investment (Technica district) correlate with real estate appreciation?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def analyze_tech_investment_impact(df, district_column, price_column, investment_column):
    """
    Analyze the correlation between economic status (growth proxy) and real estate appreciation.

    Parameters:
    df (pd.DataFrame): The dataset containing district, real estate, and investment data.
    district_column (str): Column representing district names.
    price_column (str): Column representing real estate prices.
    investment_column (str): Column representing economic status as a proxy for investment.

    Returns:
    dict: Results including correlation metrics, regression insights, and visualizations.
    """
    results = {}

    # Convert economic_status to numeric format
    economic_mapping = {"Stable": 0, "Recession": -2, "Declining": -1, "Growing": 1, "Boom": 2}
    if investment_column in df.columns:
        df[investment_column] = df[investment_column].map(economic_mapping).fillna(0).astype(float)

    # Extract Technica district data
    df_technica = df[df[district_column] == "Technica"]
    df_other = df[df[district_column] != "Technica"]

    # Compare average prices
    avg_price_technica = df_technica[price_column].mean()
    avg_price_other = df_other[price_column].mean()
    results["average_price_technica"] = avg_price_technica
    results["average_price_other_districts"] = avg_price_other

    # Correlation analysis between economic status & price appreciation
    correlation_coeff, p_value = pearsonr(df[investment_column], df[price_column])
    results["correlation_coefficient"] = correlation_coeff
    results["correlation_p_value"] = p_value  # Significance check (p-value < 0.05 indicates strong correlation)

    # Regression Analysis: How much does economic status influence real estate appreciation?
    X = df[[investment_column]]
    y = df[price_column]

    # Handle missing values
    X.fillna(X.median(), inplace=True)
    y.fillna(y.median(), inplace=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model evaluation
    results["MAE"] = mean_absolute_error(y_test, y_pred)
    results["R2_Score"] = r2_score(y_test, y_pred)

    # Visualization of price trends vs. economic status
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=df[investment_column], y=df[price_column], alpha=0.7)
    plt.title("Economic Growth (Proxy for Tech Investment) vs. Real Estate Prices")
    plt.xlabel("Economic Status")
    plt.ylabel("Real Estate Prices")
    plt.grid()
    plt.show()

    return results

# Example Usage:
df = pd.read_csv("flower_hill_data_task_13.csv")
result = analyze_tech_investment_impact(df, "district", "price", "economic_status")


# --------------------
# Task 16 - How do global economic cycles affect Flower Hill's real estate market?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def analyze_economic_cycles(df, date_column, price_column, economic_factors, visualize=True, sample_size=1000):
    """
    Analyze how local economic conditions impact Flower Hill's real estate market during global downturns.

    Parameters:
    df (pd.DataFrame): Dataset containing historical real estate pricing and local economic data.
    date_column (str): Column representing time-series data.
    price_column (str): Column representing real estate prices.
    economic_factors (list): Columns representing economic indicators (e.g., 'economic_status', 'historical_significance').
    visualize (bool): Whether to generate graphs (set to False for faster execution).
    sample_size (int): Limit dataset to speed up processing (default: 1000 rows).

    Returns:
    dict: Results including correlation metrics, regression insights, and crisis recovery patterns.
    """
    results = {}

    # Use a smaller dataset sample to reduce computation time
    df_sampled = df.sample(min(sample_size, len(df)), random_state=42)

    # Convert date to datetime format & sort dataset
    df_sampled[date_column] = pd.to_datetime(df_sampled[date_column])
    df_sampled.sort_values(by=date_column, inplace=True)

    # Convert economic_status to numeric values
    economic_mapping = {"Stable": 0, "Recession": -2, "Declining": -1, "Growing": 1, "Boom": 2}
    if "economic_status" in df_sampled.columns:
        df_sampled["economic_status"] = df_sampled["economic_status"].map(economic_mapping).fillna(0).astype(float)

    # Optimized Correlation Analysis (uses `.corrwith()` for speed)
    results["correlations"] = df_sampled[economic_factors].corrwith(df_sampled[price_column])

    # Regression Analysis: How does economic status affect prices?
    X = df_sampled[["economic_status"]]  # Only using key feature to speed up execution
    y = df_sampled[price_column]

    # Handle missing values
    X.fillna(X.median(), inplace=True)
    y.fillna(y.median(), inplace=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  # Reduced test size

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Model evaluation
    results["MAE"] = mean_absolute_error(y_test, y_pred)
    results["R2_Score"] = r2_score(y_test, y_pred)

    # Visualization (Only if enabled)
    if visualize:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=df_sampled[date_column], y=df_sampled["economic_status"], label="Economic Status", alpha=0.7)
        sns.lineplot(x=df_sampled[date_column], y=df_sampled[price_column], label="Real Estate Prices", linewidth=2, color="black")
        plt.title("Local Economic Cycles vs. Real Estate Market Trends")
        plt.xlabel("Year")
        plt.ylabel("Economic Indicators & Real Estate Prices")
        plt.legend()
        plt.grid()
        plt.show()

    return results

# Example Usage:
df = pd.read_csv("flower_hill_data_task_14.csv")
result = analyze_economic_cycles(df, "date", "price", ["economic_status"], visualize=True)

# --------------------
# Task 17 - Will new residential zones emerge over the next 50 years based on current trends?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def forecast_residential_zones(df, date_column, price_column, economic_column, size_column, visualize=True, sample_size=1000):
    """
    Forecast the emergence of new residential zones based on real estate trends and economic shifts.

    Parameters:
    df (pd.DataFrame): Dataset containing urban expansion and real estate trends.
    date_column (str): Column representing historical time-series data.
    price_column (str): Column representing real estate prices.
    economic_column (str): Column representing economic status (growth proxy).
    size_column (str): Column representing property size trends.
    visualize (bool): Whether to generate graphs (set to False for faster execution).
    sample_size (int): Limit dataset to speed up processing (default: 1000 rows).

    Returns:
    dict: Results including price predictions, economic trends, and zoning forecasts.
    """
    results = {}

    # Use a smaller dataset sample for faster execution
    df_sampled = df.sample(min(sample_size, len(df)), random_state=42)

    # Convert date column to datetime and sort
    df_sampled[date_column] = pd.to_datetime(df_sampled[date_column])
    df_sampled.sort_values(by=date_column, inplace=True)

    # Convert economic_status to numeric values
    economic_mapping = {"Stable": 0, "Recession": -2, "Declining": -1, "Growing": 1, "Boom": 2}
    if economic_column in df_sampled.columns:
        df_sampled[economic_column] = df_sampled[economic_column].map(economic_mapping).fillna(0).astype(float)

    # Forecast Economic Growth Using Exponential Smoothing (with optimized settings)
    model_economic = ExponentialSmoothing(df_sampled[economic_column], trend="add").fit()
    future_economic_trend = model_economic.forecast(50)
    results["economic_forecast"] = future_economic_trend

    # Predict Real Estate Prices Based on Trends (Optimized Index Conversion)
    X = np.arange(len(df_sampled)).reshape(-1, 1)  # Convert date indices to numeric format
    y = df_sampled[price_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  # Reduced test size

    model_price = LinearRegression()
    model_price.fit(X_train, y_train)
    future_prices = model_price.predict(np.arange(len(df_sampled), len(df_sampled) + 50).reshape(-1, 1))
    results["price_forecast"] = future_prices

    # Visualization (Only if enabled)
    if visualize:
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=df_sampled[date_column], y=df_sampled[economic_column], label="Historical Economic Status", alpha=0.7)
        plt.plot(pd.date_range(df_sampled[date_column].max(), periods=50, freq="Y"), future_economic_trend, label="Forecasted Economic Growth", linestyle="--")
        plt.title("Projected Economic Growth Over 50 Years")
        plt.xlabel("Year")
        plt.ylabel("Economic Status")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 5))
        sns.lineplot(x=df_sampled[date_column], y=df_sampled[price_column], label="Historical Prices", alpha=0.7)
        plt.plot(pd.date_range(df_sampled[date_column].max(), periods=50, freq="Y"), future_prices, label="Forecasted Prices", linestyle="--", color="red")
        plt.title("Real Estate Price Trends Over 50 Years")
        plt.xlabel("Year")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        plt.show()

    return results

# Example Usage:
df = pd.read_csv("flower_hill_data_task_15.csv")
result = forecast_residential_zones(df, "date", "price", "economic_status", "square_meters", visualize=True)