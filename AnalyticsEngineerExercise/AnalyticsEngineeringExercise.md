# 1. 🚀 Project Introduction: Exercise Analytics Engineer (Gaming)

## Objective  
Imagine you are working as an Analytics Engineer for a company offering a mobile app. This app collects millions of data points from users every day. 
Your goal is to transform and organize this data so that not only you can derive valuable insights from it, but also product managers and marketers 
can easily query it. 

### 🎯 **Primary Aim**

Attached to this exercise you can find an SQLite database containing two tables. In the following, we give a brief description of the data.
This first table (events) contains 43,479 telemetric events for one week from a hypothetical mobile app. All users contained in the database 
are new users. Each row in this table describes a single event by a user and contains the following columns:  

● user_id  

● event_name  

● event_timestamp (unix timestamp measured in microseconds)  

● platform  

● os (= operating system)  

● country  

● ad_revenue (only set for particular events related to monetization)  

● tracker_name (the name of the campaign if a user was acquired via paid
marketing or “Unattributed“ if this is an organic user).  

The second table (user_acquisition) contains information about user
acquisition campaigns. These campaigns were run to acquire new users via
different ad networks. The data contains for each day and campaign a tracker
name (i.e., the name of the campaign) and the amount spent on this day for this
campaign. To be more precise, the table contains the following columns:  

● date  

● tracker_name  

● costs  
 
### 🧠 **Tasks**

Please answer the following questions and implement the necessary tasks in a
programming language of your choice (hint: the Python standard library offers a
module sqlite3 which can be easily used for the task to read a file based database. 
Of course, you are also free to choose other packages or programming
languages).

1. What’s the total number of users present in the dataset?
2. List the number of installs per country.
3. In this exercise, you will calculate the retention for a specific cohort.
○ How many users installed the app on August 2, 2022 in Germany on
Android?
○ How many of these users are active on the first, third, and fourteenth
day after the install respectively? (I.e., count users for all three days
separately)
○ How much are those in percent? These are called day 1, day 3, and
day 14 retention.
4. Create a view named marketing that provides the following columns per
day and per campaign:
○ day
○ tracker_name
○ number_of_installs
○ costs (costs spent on this day for the specific campaign)
○ total_revenue (revenue from the users acquired on this day from
this campaign. Make use of the column ad_revenue from events)
5. Query the view marketing and report the Costs per Install (CPI) on August
6, 2022, for campaign “google_campaign1”?

### Final Remarks  

● Please submit your documented code along with instructions on how to
run the code and the answers to the questions above.  

● Please only spend 120 minutes on the exercise. We know that it is
challenging to complete all tasks but please respect the time limit.

Attachment
- exercise.db

(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/AnalyticsEngineerExercise/AnalyticsEngineeringExercise.md#6--references) 1 - 3 below).

# 2. 🔐 Ideas

## 2.1 Initial remarks - packages

This is a clean list of `!pip install` commands we can run in our Jupyter Notebook to cover everything needed for the above **Analytics Engineer Coding Exercise** using **Python** and **SQLite**:

```python
# Core packages
!pip install pandas
!pip install matplotlib  # Optional: for visualizing retention or CPI trends

# For file watching or hot reloads (optional)
!pip install watchdog

# If you plan to use SQL magic in Jupyter
!pip install ipython-sql
```

There is no need to install `sqlite3` — it is a part of Python’s standard library.

## 2.2 Coding Concepts  

Based on the provided task list  I would approach the task using **Python (Jupyter Notebook)** and the **sqlite3** module step by step in the following manner:

---

### 🧰 Setup: Load and Explore the Database

```python
import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect("exercise.db")

# Preview tables
pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
```

### 1️⃣ Total Number of Users

```python
query = "SELECT COUNT(DISTINCT user_id) AS total_users FROM events;"
pd.read_sql(query, conn)
```

### 2️⃣ Number of Installs per Country

Assuming "install" is represented by a specific `event_name` (e.g., `"install"` or similar):

```python
query = """
SELECT country, COUNT(DISTINCT user_id) AS installs
FROM events
WHERE event_name = 'install'
GROUP BY country;
"""
pd.read_sql(query, conn)
```

### 3️⃣ Retention for Cohort: Germany, Android, August 2, 2022

#### a. Users who installed on 2022-08-02

```python
from datetime import datetime, timedelta

# Convert date to microseconds
install_day = datetime(2022, 8, 2)
start_ts = int(install_day.timestamp() * 1_000_000)
end_ts = int((install_day + timedelta(days=1)).timestamp() * 1_000_000)

query = f"""
/* Select Germany, Android, August 2, 2022 */
SELECT DISTINCT user_id
FROM events
WHERE event_name = 'install'
AND country = 'Germany'
AND platform = 'Android'
AND event_timestamp BETWEEN {start_ts} AND {end_ts};
"""
cohort_users = pd.read_sql(query, conn)
```

#### b. Retention on Day 1, 3, 14

```python
def retention_day(day_offset):
    day_start = install_day + timedelta(days=day_offset)
    ts_start = int(day_start.timestamp() * 1_000_000)
    ts_end = int((day_start + timedelta(days=1)).timestamp() * 1_000_000)

    query = f"""
    SELECT DISTINCT user_id
    FROM events
    WHERE event_timestamp BETWEEN {ts_start} AND {ts_end}
    AND user_id IN ({','.join(map(str, cohort_users['user_id']))});
    """
    return pd.read_sql(query, conn)

day1 = retention_day(1)
day3 = retention_day(3)
day14 = retention_day(14)

# Retention percentages
total = len(cohort_users)
retention = {
    "Day 1": len(day1) / total * 100,
    "Day 3": len(day3) / total * 100,
    "Day 14": len(day14) / total * 100
}
```

### 4️⃣ Create View `marketing`

A view in SQL is a virtual table which does not store data itself, but instead presents the result of a stored query as if it were a table. 
We can query it just like a regular table, however in reality, it dynamically pulls data from the underlying tables.

```python
query = """
-- Create a view
CREATE VIEW IF NOT EXISTS marketing AS
SELECT
    ua.date AS day,
    ua.tracker_name,
    COUNT(DISTINCT e.user_id) AS number_of_installs,
    ua.costs,
    SUM(e.ad_revenue) AS total_revenue
FROM user_acquisition ua
LEFT JOIN events e
    ON e.tracker_name = ua.tracker_name
    AND DATE(e.event_timestamp / 1000000, 'unixepoch') = ua.date
    AND e.event_name = 'install'
GROUP BY ua.date, ua.tracker_name;
"""
conn.execute(query)
```

### 5️⃣ Query CPI for August 6, 2022, "google_campaign1"

```python
query = """
/* Select August 6, 2022 */
SELECT
    costs,
    number_of_installs,
    ROUND(costs * 1.0 / number_of_installs, 2) AS CPI
FROM marketing
WHERE day = '2022-08-06' AND tracker_name = 'google_campaign1';
"""
pd.read_sql(query, conn)
```

### 🧾 Final Notes

- We can wrap each step in Markdown cells to document our logic.
- We should use `conn.close()` at the end to cleanly close the connection.
- If needed, we can export results using `df.to_csv()` or `df.to_excel()`.

---

# 3. Bonus-ideas (Visualizations)

Now we can extend our Jupyter Notebook to include **visual dashboards** and **PDF export** of the results. This will make our analysis both insightful and presentation-ready.

---

## 📊 3.1. Retention Curve Visualization

```python
import matplotlib.pyplot as plt

# Retention dictionary from earlier
retention_days = ["Day 1", "Day 3", "Day 14"]
retention_values = [retention["Day 1"], retention["Day 3"], retention["Day 14"]]

plt.figure(figsize=(8, 5))
plt.plot(retention_days, retention_values, marker='o', linestyle='-', color='teal')
plt.title("Retention Curve – Germany / Android / 2022-08-02")
plt.xlabel("Day")
plt.ylabel("Retention (%)")
plt.grid(True)
plt.ylim(0, 100)
plt.show()
```

## 📈 3.2. CPI Trend Visualization

```python
# Load CPI data from marketing view
query = """
SELECT day, tracker_name, costs, number_of_installs,
       ROUND(costs * 1.0 / number_of_installs, 2) AS CPI
FROM marketing
WHERE tracker_name = 'google_campaign1'
ORDER BY day;
"""
cpi_df = pd.read_sql(query, conn)

# Plot CPI over time
plt.figure(figsize=(10, 5))
plt.plot(cpi_df["day"], cpi_df["CPI"], marker='o', color='darkorange')
plt.title("CPI Trend – google_campaign1")
plt.xlabel("Date")
plt.ylabel("Cost Per Install (CPI)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
```

## 🧾 3.3 Export Results to PDF

We can use `matplotlib.backends.backend_pdf` to export plots and tables:

```python
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages("analytics_dashboard.pdf") as pdf:
    # Retention plot
    plt.figure(figsize=(8, 5))
    plt.plot(retention_days, retention_values, marker='o', color='teal')
    plt.title("Retention Curve – Germany / Android / 2022-08-02")
    plt.xlabel("Day")
    plt.ylabel("Retention (%)")
    plt.grid(True)
    plt.ylim(0, 100)
    pdf.savefig()
    plt.close()

    # CPI plot
    plt.figure(figsize=(10, 5))
    plt.plot(cpi_df["day"], cpi_df["CPI"], marker='o', color='darkorange')
    plt.title("CPI Trend – google_campaign1")
    plt.xlabel("Date")
    plt.ylabel("Cost Per Install (CPI)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Optional: Export table as image
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=cpi_df.values, colLabels=cpi_df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    pdf.savefig()
    plt.close()
```

## ✅ Conclusion

- We now have a PDF file `analytics_dashboard.pdf` with retention and CPI visuals.

---

# 4. Results  

### Run instructions

1. Download the folder !["AnalyticsEngineerExercise"](https://github.com/NenadBalaneskovic/ExternalProjects/tree/dd28dbf8cddf9f1aa26eae6479bca049c936c180/AnalyticsEngineerExercise) which has the following structure:
   <img src="https://github.com/NenadBalaneskovic/ExternalProjects/blob/4d44e2bfac2bc3832651d539b633f9b3c258485a/AnalyticsEngineerExercise/FolderStructure.png" width="400" height="200"/>
2. Run all cells below marked with a comment "# --- EXECUTE CELL ---" in succession, all other cells are morkdown-cells containing interpretations (discussions) of code implementations and their results.
3. Results to posed exercise questions are displayed in subsection 5.4 below and are also repeated here for the sake of convenience:

### Questions and Answers

1. What’s the total number of users present in the dataset?  **1586**
2. List the number of installs per country. Austria	**185**, Germany **1199**, Switzerland **119**. 
3. How many users installed the app on August 2, 2022 in Germany on Android? **117** 
4. How many of these users are active on the first, third, and fourteenth day after the install respectively? (I.e., count users for all three days separately) Number of users active on **day 1**: **63**, Number of users active on **day 3**: **19**, Number of users active on **day 14**: **0**
5. How much are those in percent? These are called day 1, day 3, and day 14 retention.
   {'Day 1': **53.85 %**, 'Day 3': **16.24 %**, 'Day 14': **0.0 %**}
6. Query the view marketing and report the Costs per Install (CPI) on August 6, 2022, for campaign “google_campaign1”? **costs** (113.199537); **number_of_installs** (58); **CPI** (1.95).

## 📊 4.1. Retention Curve Visualization

![Retention_Curve_Visualization](https://github.com/NenadBalaneskovic/ExternalProjects/blob/3486e0e261412886f5ea9ebfc35185dde311067e/AnalyticsEngineerExercise/bild1.PNG)

Above is a line chart showing how retention drops over time for the selected cohort — a key visual for understanding user engagement and campaign effectiveness. 

The plot above titled **"Retention Curve – Germany / Android / 2022-08-02"** offers a clear visual of how user engagement declines over time for a specific cohort.

### 📈 Interpretation of the Retention Curve

#### 🧩 Cohort Definition
- Users who **installed the app on August 2, 2022**
- Located in **Germany**
- Using the **Android** platform

#### 🔍 Data Points

| Day     | Retention (%) | Insight |
|---------|----------------|--------|
| Day 1   | ~60%           | Strong initial engagement — users returned the next day |
| Day 3   | ~20%           | Sharp drop — many users disengaged quickly |
| Day 14  | ~0%            | Near-total churn — very few users remained active |

### 📊 What This Suggests

- **Day 1 retention** is relatively healthy, indicating that onboarding or initial experience may be effective.
- The **steep decline by Day 3** suggests a possible issue with short-term value delivery — users may not find enough reason to return.
- **Day 14 near-zero retention** implies long-term engagement is weak, and the app may need better hooks, content, or incentives to sustain interest.

### 🧠 Strategic Follow-ups

We might consider:
- Segmenting retention by **tracker_name** to see which campaigns yield stickier users.
- Comparing this curve to other countries or platforms (e.g., iOS) to identify structural differences.
- Investigating **event sequences** between install and churn to pinpoint drop-off triggers.

## 📈 4.2. CPI Trend Visualization

![CPI_Trend_Visualization](https://github.com/NenadBalaneskovic/ExternalProjects/blob/d2d500963c8ef30a776adf91e5f5761d2d61e371/AnalyticsEngineerExercise/bild2.PNG)

The plot above titled **"CPI Trend – google_campaign1"** offers a valuable look into the **cost efficiency** of a specific marketing campaign over time.

### 📈 Interpretation of the CPI Trend

#### 🧩 Campaign Context
- Tracker: `google_campaign1`
- Time window: **2022-08-02 to 2022-08-08**
- Metric: **Cost Per Install (CPI)** — calculated as `costs / number_of_installs`

#### 🔍 Observed Pattern

| Date        | CPI (approx.) | Insight |
|-------------|----------------|--------|
| Aug 2       | ~2.0           | Baseline cost level |
| Aug 3–4     | ~1.3–1.4       | Noticeable dip — possibly more efficient targeting or higher install volume |
| Aug 5       | >2.2           | Spike — cost surge or drop in installs |
| Aug 6       | ~1.9           | Slight recovery |
| Aug 7       | ~1.3           | Another dip — potentially strong performance |
| Aug 8       | ~1.4           | Stabilization at lower CPI |

### 📊 What This Suggests

- The **volatility** in CPI indicates that campaign performance is not consistent day-to-day.
- **Aug 5 spike** may warrant investigation — was there a budget change, creative swap, or platform issue?
- **Lower CPI on Aug 3–4 and Aug 7–8** suggests those days were more cost-efficient — possibly due to better audience matching or organic uplift.

### 🧠 Strategic Follow-ups

We might consider:
- Comparing CPI against **retention or revenue** to assess true ROI.
- Segmenting by **country or platform** to isolate cost drivers.
- Reviewing **install volume** and **ad spend** on high-CPI days to identify inefficiencies.

Together with the retention plot above, this CPI trend helps evaluate not just how much one is spending to acquire users, but whether those users are sticking around.

## 4.3 Conclusions:

- We now have a PDF file `analytics_dashboard.pdf` with retention and CPI visuals.

**Questions and Answers**

1. What’s the total number of users present in the dataset?  **1586**
2. List the number of installs per country. Austria	**185**, Germany **1199**, Switzerland **119**. 
3. How many users installed the app on August 2, 2022 in Germany on Android? **117** 
4. How many of these users are active on the first, third, and fourteenth day after the install respectively? (I.e., count users for all three days separately)  
Number of users active on **day 1**: **63**, Number of users active on **day 3**: **19**, Number of users active on **day 14**: **0**
5. How much are those in percent? These are called day 1, day 3, and day 14 retention. {'Day 1': **53.85 %**, 'Day 3': **16.24 %**, 'Day 14': **0.0 %**}
6. Query the view marketing and report the Costs per Install (CPI) on August 6, 2022, for campaign “google_campaign1”? **costs** (113.199537); **number_of_installs** (58);	**CPI** (1.95).

# 5. Future improvements

The datasets in the **Analytics Engineer Coding Exercise** open up a rich field of exploratory and causal analysis. 
Beyond the core metrics like installs, retention, and CPI, we could flesh out a deeper set of **investigable questions and correlations**, grouped by theme:

## 5.1 📊 User Behavior & Retention

### 🔍 Questions
- What is the average session frequency per user by country or platform?
- How does retention vary by acquisition channel (e.g. tracker_name)?
- Are users acquired via paid campaigns more likely to generate ad revenue?

### 🔗 Correlations
- **Retention vs. install date**: Are certain days of the week or months associated with higher retention?
- **Retention vs. platform**: Does Android vs. iOS show different retention curves?
- **Retention vs. ad revenue**: Do retained users contribute more to monetization?

## 5.2 💰 Marketing Efficiency

### 🔍 Questions
- Which tracker yields the best ROI (revenue vs. cost)?
- How does CPI vary over time for each campaign?
- Are there diminishing returns for high-cost campaigns?

### 🔗 Correlations
- **CPI vs. installs**: Is lower CPI associated with higher install volume?
- **Campaign cost vs. retention**: Do expensive campaigns yield more loyal users?
- **Ad revenue vs. tracker**: Which campaigns drive the most monetizable users?

## 5.3 🌍 Geographic & Platform Insights

### 🔍 Questions
- Which countries have the highest install-to-retention conversion?
- Is ad revenue per user higher in certain regions?
- Do platform-specific behaviors differ in session depth or frequency?

### 🔗 Correlations
- **Country vs. ad revenue**: Are some markets more lucrative?
- **Platform vs. CPI**: Is it cheaper to acquire users on Android vs. iOS?
- **Country vs. retention**: Are cultural or regional patterns observable?

## 5.4 🧠 Advanced Causal Inference (if time-series or user-level granularity is available)

### 🔍 Questions
- Does early engagement (e.g. first 24h activity) predict long-term retention?
- Can we identify churn predictors based on event sequences?
- What is the causal impact of tracker cost on downstream revenue?

### 🔗 Techniques
- **Propensity score matching**: Compare similar users across campaigns.
- **Survival analysis**: Model time-to-churn.
- **Granger causality**: Test if one time series (e.g. installs) predicts another (e.g. revenue).  

# 6. 📚 References
1. R. Nystrom: "__Game Programming Patterns__", 1st Ed. genever benning (2014); A. A. Stepanov, D. E. Rose: "__From Mathematics to Generic Programming__", 1st Ed. Addison-Wesley (2015); 
K. Webel, D. Wied: "__Stochastische Prozesse__", 2. Auflage Springer (2016); L. Held: "__Methoden der statistischen Inferenz__", 1. Auflage Spektrum (2008);
[Link to the Exercise:](https://github.com/NenadBalaneskovic/ExternalProjects/blob/5821cffe9960af6739066d3c5a1e2a7da1946bf5/AnalyticsEngineerExercise/Analytics%20Engineer%20Coding%20Exercise.pdf).
2. [![Jupyter Notebook | English](https://img.shields.io/badge/Jupyter%20Notebook-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/bed77a87db8dcaca580bca2da1b6d81d16cdb150/AnalyticsEngineerExercise/AnalyticsEngineerTask.ipynb)
3. [![Analytics_Engineering_Gaming_Analysis Report | English](https://img.shields.io/badge/Analytics_Engineering_Gaming_Analysis%20Report-English-yellowblue?logoColor=blue&labelColor=red)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/7b91e2a516ed3974a15aae5d083e0332947e65e6/AnalyticsEngineerExercise/analytics_dashboard.pdf) 
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










