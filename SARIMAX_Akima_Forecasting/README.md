# Boarder Crossing Forecasting - SARIMAX Modeling

## 📌 Introduction
This project attempts at **forecasting the temporal stock prices evolution** based on a ficticious csv file containing daily stock prices by means of Akima interpolated stock price data subject to SARIMAX and
critical point modeling implemented via the physical theory of critical phenomena (see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/DZ_bank_DataSet_classification/README.md#-references)
 1 - 3 below). It introduces pythonic functions for Akima interpolation and critical point extraction from stock price time series, which allow the user to characterise critical inflection points of
 stock price evolution and aid the customized SARIMAX forecasting functions in reliably estimating volatility ranges of unknown stock price changes. The project also compares the regular customized Pythonic SARIMAX 
 functionality with its batched version and delves into its conceptual intricacies. 

## 📂 Dataset Overview
We utilized a ficticious stock price data, including:
- **Date** → Date information formatted as "YYY-MM-DD" from January 1st 2025 until June 30th 2025.
- **Price** → Closing price of a stock for the particular trading day (float).

In addition, the Pythonic customized Akima interpolator function creates an intermediary data set structure which is then used by the (batched) SARIMAX algorithm for further evaluation:
- **Date** → Date information formatted as "YYY-MM-DD" from January 1st 2025 until June 30th 2025.
- **First Derivative** → First derivative of stock prices indicating their rates of change.
- **Second Derivative** → Second derivative of stock prices indicating their acceleration and sudden shifts in stock trends.
- **Trend Type** → Characterises the type of trend shifts as "Bullish Surge" or "Sharp Decline".

## 🔄 Data Preprocessing  
✔ **Generation of a stock price data set** – Generate an artificial stock price data set. 
✔ **Akima interpolation of stock price time series** – Perform the Akima-interpolation of stock price time series. 
✔ **Extraction of critical points of the stock price time series** – Extract critical points from the stock price time series (first, second and third derivative).   
✔ **Storage of critical points** – Store extracted critical points from a stock price data set into a separate csv file for further processing.  

## 🤖 Model Implementation
### **Individual Models**
- **Critical Point Estimation and Trend Characterisation of Time Series**
- - **Akima-interpolation of stock price time series**
- **SARIMAX forecasting**
- - **Customized forecasting of time series via critical point extraction**
- - **Batched customized forecasting of time series via critical point extraction**

### **Ensemble Learning Techniques**
- **Automated SARIMAX Grid Search**

## 📊 Evaluation Metrics
| Model | RMSE | MAE | $$R^{2}$$ | Duration [sec] |
|-------|---------|----------|--------|--------|
| Customized SARIMAX Forecasting | 18.4792 | 17.8311 | -7.7315 | 2.13 |
| **Batched SARIMAX Forecasting**| **18.4792** | **17.8311** | **-7.7315** | **2.18** |

Abbreviations:  
1. __RMSE__ : Root Mean Squared Error  
2. __MAE__ : Mean Absolute Error  
3. __$$R^{2}$$__ : R-squared

## 📈 Results & Insights
✔ **An Akima-interpolated, batched SARIMAX modeling enables reliable estimation of volatility bands for forecasted stock price trends**.  
✔ **Critical Point trend analysis** allows correct characterisations of past stock price behavior and enables robust predictions of future time series patterns.

## 📷 Visualizations
### 1. Akima interpolated stock price data set
![Train/Test Split](https://github.com/NenadBalaneskovic/ExternalProjects/blob/161823144626734e4908ae407fa69b84c2deec21/DZ_bank_DataSet_classification/Fig8.PNG)  
### 2. Adaptive Trend Detection of Stock Prices
![Customized SARIMAX](https://github.com/NenadBalaneskovic/ExternalProjects/blob/f2a05f9a2998e0b9992a2c8856a61f4a05c53a3e/DZ_bank_DataSet_classification/Fig5.PNG)  
### 3. SARIMAX-forecasted Stock Prices
![Automated SARIMAX](https://github.com/NenadBalaneskovic/ExternalProjects/blob/735f0d2547281074c02e432d3615e20cbf2197b9/DZ_bank_DataSet_classification/Fig5.PNG)  

## 🚀 Storage of Forecast results
✔ **Trend forecasts are stored in a separate file "critical_trends.csv"**. 

## 📚 References
1. **Boarder Crossings Dataset** – https://www.kaggle.com/datasets/akhilv11/border-crossing-entry-data
2. [![Jupyter Notebook | English](https://img.shields.io/badge/Jupyter%20Notebook-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/5836d4c0485ce1bdae77807c6d70b178d4082e5f/SARIMAX_Akima_Forecasting/AkimaInterpolationStocks.ipynb)
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
26. Michał Chorowski, Tomasz Gubiec, Ryszard Kutner: "__Anomalous Stochastics: A Comprehensive Guide to Multifractals, Random Walks, and Real-World Applications__", Springer (2025).
27. Cyril Domb, Joel Lebowitz: "__Phase Transitions and Critical Phenomena__" (20-volume series), Academic Press (1971-2001).
28. Chris Chatfield: "__Time-Series Forecasting__", Chapman and Hall/CRC (2000).
29. Juliane T. Moraes, Silvio C. Ferreira: "__Strong Localization Blurs the Criticality of Time Series for Spreading Phenomena on Networks__",Phys. Rev. E 111, 044302 – Published 8 April, 2025.
30. Klaus Lehnertz: "__Time-Series-Analysis-Based Detection of Critical Transitions in Real-World Non-Autonomous Systems__", Chaos: An Interdisciplinary Journal of Nonlinear Science, https://doi.org/10.1063/5.0214733.
31. Saad Zafar Khan et al.: "__Quantum Long Short-Term Memory (QLSTM) vs. Classical LSTM in Time Series Forecasting__", Front. Phys., 09 October 2024, Sec. Quantum Engineering and Technology, Volume 12 - 2024, https://doi.org/10.3389/fphy.2024.1439180.
32. Hiroshi Akima: "__A New Method of Interpolation and Smooth Curve Fitting Based on Local Procedures__", Journal of the ACM. 17: 589–602 (1970).
---
