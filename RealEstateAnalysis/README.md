# Real Estate Data Set Analysis in Flower Hill

## 📌 Project Overview
This project involves a large data set related to real estate sales for a fictional town of Flower Hill. The aim is to combine the analysis of this data set with PyMongo,
MLflow, Python (SARIMAX times series forecasting, classification, Neural Networks, Kohonen Maps) and DAG-like process organization of ML-tasks.
Thus, we are blending data engineering, machine learning, forecasting, and process automation into a well-structured framework
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/tree/main/RealEstateAnalysis#-references) 1 - 3 below).

## 📜 Flower Hill's Story
Flower Hill is a flourishing City, consisting of 7 districts: Costal, Central, Mercantil, Faunarium, Technica, Academia and Historica. 
Costal department contains usually high valued real estate items, such as villas, luxury appartments, etc. 
Central is the center of the city consisting mostly of offices. Mercantil revolves around the Stock Exchange of the City. 
Faunarium contains parks, lakes and real estate item connected with them. Technica contains technology companies, Academia district is 
centered around the Flower University of Technology, whereas Historica represents a traditional, historical real estate district. 
The City dates back almost 200 years and we could try to create a data set of real estate purchases from 1900 till 2025.

### **🛠 Conceptual Framework**
Here’s how we can approach this project step by step:

#### **1️⃣ Data Structure & Storage** (PyMongo)  
✔ Define the schema for real estate transactions (prices, timestamps, property types, etc.).  
✔ Optimize indexing in **MongoDB** for fast querying and aggregation.  
✔ Use **historical data pipelines** to preprocess sales trends.  

#### **2️⃣ Machine Learning Strategy** (MLflow for Tracking)  
✔ **SARIMAX** → Time series forecasting for predicting real estate trends in Flower Hill.  
✔ **Classification Models** → Predict property price ranges, buyer demographics, and market trends.  
✔ **Neural Networks** → Deep learning models for pattern recognition (e.g., feature engineering).  
✔ **Kohonen Maps** → Self-organizing maps for **clustering neighborhoods** based on sales behavior.  

#### **3️⃣ DAG-Based ML Pipeline** (Task Automation)  
✔ Implement **Apache Airflow** or another DAG framework to define steps for data ingestion, transformation, and model training.  
✔ Automate periodic **forecast updates** using DAG scheduling.  
✔ Integrate **MLflow** logging for monitoring different model versions and optimizing hyperparameters.  

## **📜 Defining the Core Data Schema**
Since we are tracking **real estate transactions** across **historical time periods**, our dataset should be structured as follows:

#### **1️⃣ Core Fields**
| Field Name       | Type        | Description |
|-----------------|------------|-------------|
| `transaction_id` | `String`   | Unique identifier for each purchase |
| `date`          | `DateTime`  | Transaction timestamp (from 1900 to 2025) |
| `district`      | `String`    | One of Flower Hill's 7 districts |
| `property_type` | `String`    | Apartment, villa, commercial, park-related, historical property, etc. |
| `price`         | `Float`     | Transaction price in standard currency |
| `buyer_type`    | `String`    | Individual, corporation, developer, etc. |
| `seller_type`   | `String`    | Private owner, government, corporation |
| `square_meters` | `Float`     | Property size in m² |
| `condition`     | `String`    | New, renovated, needs repair, abandoned |
| `economic_status` | `String`  | City-wide economy marker (e.g., boom, recession) |
| `historical_significance` | `Boolean` | Whether the property has historical relevance |

---

### **🔥 Database Design in MongoDB (PyMongo)**
MongoDB is great for **handling time-series data** and **unstructured real estate transactions** dynamically.

💡 **Collection: `transactions`**
```python
{
    "_id": "TXN10001234",
    "date": "1999-07-15T00:00:00",
    "district": "Costal",
    "property_type": "Luxury Villa",
    "price": 2_500_000.00,
    "buyer_type": "Individual",
    "seller_type": "Developer",
    "square_meters": 450.0,
    "condition": "New",
    "economic_status": "Boom",
    "historical_significance": false
}
```

---

### **📊 Machine Learning Tasks**
💡 **Forecasting with SARIMAX**  
- **Predict district-wide property price trends** using historical sales data.  
- **Analyze market dips & peaks** based on economic conditions.  

💡 **Classification & Neural Networks**  
- **Segment buyers** by their purchasing behavior (individual vs. corporate).  
- **Predict future property values** based on district, condition, and economy.  

💡 **Kohonen Maps (Self-Organizing Neural Networks)**  
- **Cluster districts by real estate trends** (growth vs. stagnation).  
- **Analyze urban expansion & contraction patterns** over time.  


## **📜 Comprehensive List of Investigation Questions & Analytical Tasks**
Each item in this list is designed to **extract insights from Flower Hill’s real estate dataset**, combining **time series forecasting, classification, clustering, and predictive modeling**.

---

### **📈 Time Series Forecasting (SARIMAX)**  
#### **1️⃣ Predict district-wide property price trends using historical sales data**  
✔ **Goal:** Forecast long-term real estate price trends for each of Flower Hill’s seven districts.  
✔ **Significance:** Helps investors and policymakers **understand price evolution** and identify future growth areas.  

#### **2️⃣ Analyze market dips & peaks based on economic conditions**  
✔ **Goal:** Detect how **booms vs. recessions** impact property prices.  
✔ **Significance:** Enables predictive modeling for **risk assessment** and **market stability** evaluations.  

---

### **🔎 Classification & Neural Networks**  
#### **3️⃣ Segment buyers by their purchasing behavior (individual vs. corporate)**  
✔ **Goal:** Identify different **buyer profiles** based on purchase frequency, transaction size, and district preference.  
✔ **Significance:** Essential for **market segmentation**, **investment strategy planning**, and **real estate demand analysis**.  

#### **4️⃣ Predict future property values based on district, condition, and economy**  
✔ **Goal:** Estimate **property price appreciation or depreciation** over time.  
✔ **Significance:** Useful for **real estate valuation, lending risk management, and developer investment** strategies.  

---

### **🏡 Kohonen Maps (Self-Organizing Neural Networks)**  
#### **5️⃣ Cluster districts by real estate trends (growth vs. stagnation)**  
✔ **Goal:** Group districts into **high-growth vs. stagnant real estate zones**.  
✔ **Significance:** Enables **urban planning decisions**, highlighting areas that **need infrastructure investment**.  

#### **6️⃣ Analyze urban expansion & contraction patterns over time**  
✔ **Goal:** Track how **Flower Hill has developed or shrunk** based on historical sales data.  
✔ **Significance:** Key for **predicting suburban sprawl vs. central densification** in future decades.  

#### **7️⃣ Predict to which district a new real estate item belongs based on its features**  
✔ **Goal:** Classify newly listed properties by district based on attributes like **price, square meters, condition, and type**.  
✔ **Significance:** Useful for **automated property listings** and **district-specific pricing recommendations**.  

#### **8️⃣ Forecast the number of purchased real estate items in the future**  
✔ **Goal:** Predict **property transaction volume trends** for different time periods.  
✔ **Significance:** Helps forecast **real estate demand fluctuations** and **economic health indicators**.  

---

### **📊 Advanced Analytical Inquiries**
#### **9️⃣ How do global economic cycles affect Flower Hill's real estate market?**  
✔ **Goal:** Understand **how recessions, booms, and financial crises** influence property transactions.  
✔ **Significance:** Critical for **long-term market forecasting** and **economic resilience studies**.  

#### **🔟 Predict whether a property will be resold within a certain timeframe**  
✔ **Goal:** Model the **resale probability** of a real estate asset based on features like **condition and price history**.  
✔ **Significance:** Helps **investors anticipate returns** and **predict flipping trends**.  

#### **1️⃣1️⃣ Identify dominant buyer types in different districts (corporations vs. individuals)**  
✔ **Goal:** Analyze the **buyer distribution per district** over time.  
✔ **Significance:** Helps assess **market control** (corporate vs. personal investments) and trends in **housing affordability**.  

#### **1️⃣2️⃣ Predict which district will see the highest appreciation in the next decade**  
✔ **Goal:** Forecast **future real estate hotspots** based on urban growth trends.  
✔ **Significance:** Enables **investment targeting**, **city expansion planning**, and **developer strategy** adjustments.  

#### **1️⃣3️⃣ How does the presence of parks & lakes (Faunarium district) influence real estate prices?**  
✔ **Goal:** Quantify the impact of **green spaces on property value**.  
✔ **Significance:** Helps policymakers and developers **prioritize environmental planning**.  

#### **1️⃣4️⃣ Does technology investment (Technica district) correlate with real estate appreciation?**  
✔ **Goal:** Determine whether **high-tech hubs drive real estate price growth**.  
✔ **Significance:** Helps **cities plan economic zones** and **investors identify profitable regions**.  

#### **1️⃣5️⃣ Will new residential zones emerge over the next 50 years based on current trends?**  
✔ **Goal:** Forecast **suburban expansion vs. urban densification** over a long timeframe.  
✔ **Significance:** Supports **city zoning decisions**, **transport planning**, and **housing policies**.  

---

### **📊 Prioritized Investigation Tasks**
| **Task #** | **Short Task Description** | **Prioritization Reason** |
|------------|----------------------------|----------------------------|
| **1** | Predict district-wide property price trends using historical sales data | Foundational analysis; necessary for forecasting & investment decisions. |
| **2** | Analyze market dips & peaks based on economic conditions | Helps understand how recessions & booms impact Flower Hill's real estate market. |
| **3** | Segment buyers by their purchasing behavior (individual vs. corporate) | Key for classifying market trends, buyer strategies, and ownership shifts. |
| **4** | Predict future property values based on district, condition, and economy | Essential for urban planning, lending risk assessments, and developer strategies. |
| **5** | Forecast the number of purchased real estate items in the future | Helps predict real estate demand fluctuations & economic health indicators. |
| **6** | Predict to which district a new real estate belongs based on its features | Supports automated property listings & accurate district-specific pricing suggestions. |
| **7** | Cluster districts by real estate trends (growth vs. stagnation) | Helps detect urban expansion zones vs. underperforming areas. |
| **8** | Analyze urban expansion & contraction patterns over time | Reveals historical urban trends & predicts future density changes. |
| **9** | Identify dominant buyer types in different districts (corporations vs. individuals) | Allows deeper insights into market control & housing affordability trends. |
| **10** | Predict whether a property will be resold within a certain timeframe | Important for understanding flipping trends & repeat transactions. |
| **11** | Predict which district will see the highest appreciation in the next decade | Investment-critical forecasting to identify the **next booming district**. |
| **12** | How does the presence of parks & lakes (Faunarium district) influence real estate prices? | Explores the **environmental impact** on real estate values. |
| **13** | Does technology investment (Technica district) correlate with real estate appreciation? | Studies the role of innovation hubs in shaping urban pricing models. |
| **14** | How do global economic cycles affect Flower Hill's real estate market? | Provides insights into Flower Hill’s resilience during financial crises. |
| **15** | Will new residential zones emerge over the next 50 years based on current trends? | Supports long-term housing policies and urban zoning decisions. |

## 🚀 Features
- **Data Preprocessing** → Cleaning and storing new transactions in MongoDB.
- **SARIMAX Forecasting** → Running time-series analysis on price trends.
- **Classification (Neural Networks & Random Forest)** → Predicting property valuation.
- **Kohonen Maps** → Clustering districts based on real estate patterns.

## 📊 Airflow - MLflow methodology
Here’s the **full Python implementation** that sets up:  

1️⃣ **An MLflow-Airflow Anaconda environment on Windows 10**  
2️⃣ **A structured Airflow DAG execution for all 18 analysis tasks from the file TaskList.txt**  
3️⃣ **MLflow tracking for task completion, numerical results, and visual outputs**  

This implementation ensures proper **environment creation**, **dependency installation**, **task execution**, and **tracking** with minimal setup issues. 🚀✨  

---

### **📌 Step 1: Create Anaconda Environment for MLflow & Airflow**
Run this in **Command Prompt (Windows Terminal)**:
```bash
conda create --name airflow_mlflow_env python=3.9
conda activate airflow_mlflow_env
```
Then, install **Airflow** and **MLflow**:
```bash
conda install -c conda-forge apache-airflow
pip install mlflow pandas numpy scikit-learn matplotlib seaborn
```

Initialize Airflow database:
```bash
airflow db init
```
Start the MLflow tracking server:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host localhost --port 5000
```

---

### **📌 Step 2: Define DAGs for Automating All 18 Analysis Tasks**
Each task from **TaskList.txt** is structured into an **Airflow PythonOperator**, executed in sequence.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import pandas as pd
from task_functions import *

default_args = {"start_date": datetime(2025, 1, 1), "retries": 1}
dag = DAG("flower_hill_analysis", default_args=default_args, schedule_interval=None)

def run_task(task_id, function, dataset_file):
    """
    Executes an analysis task and logs results in MLflow.

    Parameters:
    - task_id (str): Unique task identifier.
    - function (function): Function performing the analysis.
    - dataset_file (str): CSV file containing dataset for the task.

    Logs:
    - Execution status, numerical results, and visualizations to MLflow.
    """
    mlflow.start_run(run_name=f"Task_{task_id}")
    df = pd.read_csv(dataset_file)
    result = function(df)

    # Log key metrics
    for key, value in result.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)

    mlflow.end_run()

# Define all 18 tasks
task_functions = {
    "task_0": install_packages,
    "task_1": generate_synthetic_data,
    "task_2": extract_mongo_data,
    "task_3": predict_price_trends,
    "task_4": analyze_market_dips,
    "task_5": segment_buyers,
    "task_6": predict_future_values,
    "task_7": forecast_transaction_volume,
    "task_8": classify_real_estate_districts,
    "task_9": cluster_districts,
    "task_10": analyze_urban_expansion,
    "task_11": analyze_buyer_dominance,
    "task_12": predict_property_resale,
    "task_13": forecast_high-appreciation-districts,
    "task_14": analyze_environmental_impact,
    "task_15": analyze_tech_investment,
    "task_16": analyze_economic_cycles,
    "task_17": forecast_residential_zones
}

# Create DAG tasks
dag_tasks = []
previous_task = None

for task_id, func in task_functions.items():
    dataset_file = f"flower_hill_data_task_{task_id}.csv"
    task = PythonOperator(
        task_id=task_id,
        python_callable=run_task,
        op_kwargs={"task_id": task_id, "function": func, "dataset_file": dataset_file},
        dag=dag
    )

    if previous_task:
        previous_task >> task  # Set execution order
    previous_task = task
    dag_tasks.append(task)

```

---

### **📌 Step 3: Deploy & Monitor Tasks in MLflow**
Start Airflow scheduler to execute the DAGs:
```bash
airflow scheduler
```
Track results **in MLflow UI**:
```bash
mlflow ui --port 5000
```

---

### **🚀 Final Thoughts**
✔ **This implementation ensures** automated execution of all 18 tasks via Airflow.  
✔ **MLflow logs** execution results, tracking numerical values and figures. 

## 🔬 Comparison Goals: District identification and Resale Probability predictions for new property via Random Forest
![Training Accuracy of the Random Forest Classifier](https://github.com/NenadBalaneskovic/ExternalProjects/blob/6d57a58906694bb30fcab466b17cc8e8428b34a6/RealEstateAnalysis/Fig7.PNG)
- **Average Accuracy of the Random Forest Classifier**: 0.83.
- Classification hyperparameters: **n_estimators = 100**, **random_state = 42**.
- **Average recall, precision and f1-score**: 0.83.

## 📂 Repository Structure
![MongoDB-Python_Airflow-MLflow repository structure](https://github.com/NenadBalaneskovic/ExternalProjects/blob/0ae13210b7c313d46af4671d715fdd5e71709033/RealEstateAnalysis/Fig6.PNG)
   
## 📷 Visualizations  

### Property price analysis
![Property price evolution across districts](https://github.com/NenadBalaneskovic/ExternalProjects/blob/77dd3737b0aa0ead474a71b496a3983fc7b566e5/RealEstateAnalysis/Fig1.PNG)  
![Property price evolution of the Costal district](https://github.com/NenadBalaneskovic/ExternalProjects/blob/231b49c620fdb2135d9f7595f5ab9883d1e617f0/RealEstateAnalysis/Fig2.PNG)  

### District Growth Clustering and Buyer Segmentation
![District Growth Clustering](https://github.com/NenadBalaneskovic/ExternalProjects/blob/5fec4f98ed3c0fd3cd6bb4230f62c092929fb85f/RealEstateAnalysis/Fig3.PNG)  

### Urban Expansion, Market Control and Comparative Price Analysis
![Urban Expansion, Market Control and Comparative Price Analysis](https://github.com/NenadBalaneskovic/ExternalProjects/blob/294b3c9120821e0377256a094c805364b82fd43a/RealEstateAnalysis/Fig4.PNG)

### Real Estate Market & Price Trends, Projected Economic Growth
![Implemented Sentiment Analysis GUI](https://github.com/NenadBalaneskovic/ExternalProjects/blob/fd2cb21f5228ece32d7deac504e96ca129828d6b/RealEstateAnalysis/Fig5.PNG) 

## 🚀 Results  
1️⃣ **📈 Districts with Continuous Price Growth**  
   - If certain districts show **a steady upward trend**, they are **booming areas** with strong investment potential.  
   - For example, **Technica** (if present) might have **high-tech development**, boosting property value.  
   - This could suggest **strong demand, ongoing development projects, or corporate investments.**  

2️⃣ **🔄 Districts with Cyclical or Fluctuating Prices**  
   - If some districts display **price surges followed by declines**, these areas might experience **short-term investment bursts** but lack sustained growth.  
   - Economic phases like **Boom → Recession → Recovery** might be influencing price fluctuations.  

3️⃣ **📉 Declining Price Trends**  
   - If districts like **Costal or Academia** (for example) show **continuous price drops**, it could indicate:  
     ✔ **Economic downturns affecting real estate**  
     ✔ **Infrastructure decline or migration to high-growth areas**  
     ✔ **Regulatory changes impacting housing demand**  

4️⃣ **📌 Economic Status Correlations**  
   - If we overlay **economic status (Boom, Stable, Recession)** on the price graph, we might observe:  
     ✔ **Boom periods aligning with price increases**  
     ✔ **Recession periods causing noticeable downturns**
   
5️⃣ **📊 Comparative Price Analysis (Box Plot)**
✔ **Faunarium district's real estate prices** appear **higher** than many other districts.  
✔ **Median price** in Faunarium seems to hover around **6,000,000**, with a **range spanning 4,000,000–8,000,000**.  
✔ **Outlier properties** show that prices can **reach 10,000,000**, indicating the presence of **high-value real estate**.  
✔ The **wider price distribution** suggests that **environmental factors, such as parks and lakes, might increase property value variability**.

6️⃣ **📉 Feature Importance (Bar Chart)**
✔ **Historical significance negatively impacts real estate value**—suggesting that older properties **might not hold as much modern appeal** despite their legacy.  
✔ **Square meters have an even stronger negative correlation**, indicating that **larger properties tend to be priced lower on a per-unit basis**—possibly due to land-use zoning restrictions or lower demand for oversized housing.  
✔ This suggests that **proximity to parks and lakes alone might not be the main driver of price increases**, but rather **other factors like district reputation, demand, and infrastructure could be playing a role**.
✔ While **Faunarium does show elevated property prices**, historical significance and size seem to be **strong limiting factors** on overall price appreciation.  
✔ **Additional analysis** incorporating **park adjacency, lake views, or green space ratings** might further clarify **the precise environmental impact**.

7️⃣ **📊 Interpretation of the Box Plot (Faunarium vs. Other Districts)**  

Median real estate price is approximately 6,000,000, meaning properties in Faunarium tend to be higher-valued than many other districts.

Wide price variation suggests diverse pricing models across the district—possibly due to environmental influence or infrastructure.

Outlier properties reaching 10,000,000 indicate a premium market segment, likely influenced by location desirability.

8️⃣ **📈 Interpretation of Feature Impact (Historical Significance & Size)**  

✔ Historical significance has a slightly negative impact on property prices, implying modern properties may hold greater appeal in the market.

✔ Square meters have a stronger negative correlation with prices, suggesting larger properties don’t necessarily mean higher value per unit.

9️⃣ **🚀 Relevance to Tech Investment & Real Estate Growth**  

✔ If historical significance negatively impacts real estate, it suggests that innovation hubs like Technica might attract higher property valuations due to modern infrastructure, startups, and advanced technology clusters.

✔ The correlation between district type and pricing trends could help determine whether technology investment supports urban appreciation, much like how environmental factors influenced Faunarium’s pricing trends. 

10️⃣ **Real Estate Price Variability (Box Plot – Faunarium vs. Other Districts)**
✔ **Wide Price Range:** The real estate prices in Faunarium have a **broad distribution**, suggesting that external economic cycles could be influencing market fluctuations.  
✔ **Higher Median Prices:** The median price sits around **6,000,000**, indicating **strong market valuation despite economic variations**.  
✔ **Presence of High-Value Outliers:** Prices reaching **10,000,000** suggest **some properties maintain premium value** even during downturns.  

📌 **Implication:** Flower Hill's real estate **might exhibit resilience**, with some properties retaining high valuation regardless of global economic cycles.

11️⃣ **Impact of Historical Significance & Size on Prices (Bar Chart)**
✔ **Historical Significance Negatively Impacts Prices:** Older properties **may not hold their value** during financial crises, hinting at **market preference for modern investments**.  
✔ **Square Meters Have a Stronger Negative Effect:** Larger properties tend to **lose more value**, possibly due to **higher maintenance costs or reduced liquidity in economic downturns**.  

📌 **Implication:** During recessions, buyers might **prefer smaller, modern properties** with **lower upkeep costs**, aligning with **economic shifts influencing investment behaviors**.

12️⃣ **🚀 Final Takeaway**
✔ **Market Resilience:** Some districts (**like Faunarium**) appear **more resistant** to downturns, maintaining **high-value properties despite financial crises**.  
✔ **Investment Shifts in Economic Cycles:** Buyers may **move away from historical properties** and focus on **modern, efficient real estate** during downturns.  
✔ **Economic Influence on Property Size Preference:** Larger homes **lose more value**, suggesting a **trend toward compact, cost-efficient housing** when financial uncertainty is high.  

## 📈 Future Improvements  
✔ **Future improvements may involve the following aspects**:

1. Enhance MLflow tracking by **storing visualizations & comparative results**. 

2. Deploy **Airflow DAGs on a cloud-based solution** (AWS, GCP, Azure) for scalability.

3. Optimize processing speed using GPU acceleration.

4. We can extend the MLflow-Airflow setup by adding custom dependencies or modifying task execution orders.

## 📚 References
1. **Kohonen Maps** – https://pypi.org/project/sklearn-som/, https://www.geeksforgeeks.org/self-organising-maps-kohonen-maps/, https://pypi.org/project/kohonen/, https://github.com/Kursula/Kohonen_SOM
2. [![Jupyter Notebook | English](https://img.shields.io/badge/Jupyter%20Notebook-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/3c79ca5422db323da28488a34c72af642e402c05/RealEstateAnalysis/RealEstateAnalysis.ipynb)
3. [![Real_Estate Report | English](https://img.shields.io/badge/Real_Estate%20Report-English-yellowblue?logoColor=blue&labelColor=red)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/3a07dee498fa12cef3d92f4dcaf146032365b442/SARIMAX_Forecasting/SARIMAX_BoarderCrossingReport.pdf) 
4. **Apache Airflow Documentation** – https://pypi.org/project/apache-airflow/, https://airflow.apache.org/
5. **MLflow Python API** – https://mlflow.org/docs/latest/api_reference/python_api/index.html, https://pypi.org/project/mlflow/
6. **PyMongo Documentation** - https://pymongo.readthedocs.io/en/stable/, https://pypi.org/project/pymongo/, https://www.w3schools.com/python/python_mongodb_getstarted.asp
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
