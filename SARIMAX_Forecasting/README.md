# Boarder Crossing Forecasting - SARIMAX Modeling

## 📌 Introduction
This project attempts at **forecasting the number of boarder crossings** between USA and Canada based on the corresponding kaggle data set of The Bureau of 
Transportation Statistics (BTS) containing entries from 1996 to 2024, employing Python's SARIMAX forecasing scheme to optimize predictive accuracy and improve future security strategies 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/DZ_bank_DataSet_classification/README.md#-references) 1 - 3 below).

## 📂 Dataset Overview
We utilized boarder crossing data, including:
- **Port Locations** → Different crossing points across states like Maine, Texas, California, and Montana.
- **Border Type** → Specifies whether the crossing is at the U.S.-Canada or U.S.-Mexico border (here we concentrate on the U.S. -Canada border).
- **Date** → Covers crossings from January 1996 to December 2024.
- **Traffic Type** → Various measures like trucks, personal vehicles, pedestrians, trains, buses, and containers (loaded or empty).
- **Entry Volume** → The "Value" column indicates the number of crossings per category at each location (this is the target variable that needs to be forecasted).
- **Geographical Coordinates** → Latitude & Longitude included for mapping purposes.

## 🔄 Data Preprocessing
✔ **Extraction of relevant columns** – Extract relevant data set columns by means of DuckDB: 'Date Stamp' and 'Total Entry Volume per Month' and their upload into a pandas Data Frame. 
✔ **Aggregation** – Aggregate the extracted data set by summing the Entry Volume values within each month. This yields a Data Frame containing columns '1996' to '2024',  
with each column containing aggregated (summed) Entry Volumes for months '0' (January) to '11' (December).  
✔ **Train/Test Split** – Split the data set into an 80 % - 20 % train/test portions.  
✔ **SARIMAX-GridSearch** – Implement a SARIMAX grid search functionality.  

## 🤖 Model Implementation
### **Individual Models**
- **SARIMAX forecasting (including FFT period estimation)**
- - **Customized cross validation**
- - **Automated cross validation**

### **Ensemble Learning Techniques**
- **Automated SARIMAX Grid Search**

## 📊 Evaluation Metrics
| Model | RMSE score | MAPE | Duration [sec] |Tuned Hyperparameters |
|-------|---------|----------|--------|-----------------------|
| Customized SARIMAX Grid Search | 3622052 | - | 191 | `p=2.0`, `d=1.0`, `q=2.0`, `P=2.0`, `D=2.0`,`Q=2.0`,`s=12` |
| **Automated SARIMAX Grid Search**| - | **0.06** | **261** | `p=4.0`, `d=1.0`, `q=2.0`, `P=2.0`, `D=2.0`,`Q=2.0`,`s=12` |
| Abbreviations | __RMSE__ | Root Mean Squared Error | __MAPE__ |Mean Absolute Percentage Error|

## 📈 Results & Insights
✔ **An automated SARIMAX cross-validated grid search improved forecasting accuracy significantly**.  
✔ **Lower MAPE-values** in predictions, even with respect to unexpected data series patterns.

## 📷 Visualizations
### 1. Train/Test Split of the data set
![Train/Test Split](https://github.com/NenadBalaneskovic/ExternalProjects/blob/161823144626734e4908ae407fa69b84c2deec21/DZ_bank_DataSet_classification/Fig4.PNG)  
### 2. Customized cross-validated SARIMAX grid search
![Customized SARIMAX](https://github.com/NenadBalaneskovic/ExternalProjects/blob/f2a05f9a2998e0b9992a2c8856a61f4a05c53a3e/DZ_bank_DataSet_classification/Fig12.PNG)  
### 3. Automated cross-validated SARIMAX grid search
![Automated SARIMAX](https://github.com/NenadBalaneskovic/ExternalProjects/blob/735f0d2547281074c02e432d3615e20cbf2197b9/DZ_bank_DataSet_classification/Fig3.PNG)  

## 🚀 Model Parameter Storage
✔ **Parameters of the trained best models can be stored as dictionaries within the jupyter notebook**. 

## 📚 References
1. **Boarder Crossings Dataset** – https://www.kaggle.com/datasets/akhilv11/border-crossing-entry-data
2. [![Jupyter Notebook | English](https://img.shields.io/badge/Jupyter%20Notebook-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/03d304db3daf8b6e50d33c4706835dbf9eefa9c5/DZ_bank_DataSet_classification/DZ_Bank_HomeAssignment.ipynb)
3. [![Forecasting Report | English](https://img.shields.io/badge/SARIMAX%20Report-English-yellowblue?logoColor=blue&labelColor=red)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/dd2cfa231369855a7f61906c3cf3fb1ed9825042/DZ_bank_DataSet_classification/MarketingDataSetAnalysis.pdf) 
4. **Scikit-Learn Documentation** – https://scikit-learn.org  
5. **XGBoost Library** – https://xgboost.readthedocs.io
6. **SARIMAX statsmodel** - https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html
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
---
