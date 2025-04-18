# Bank Marketing Campaign - Predictive Modeling

## 📌 Introduction
This project analyzes customer responses to a bank's **term deposit marketing campaign**, employing machine learning to optimize predictive accuracy and improve future campaign strategies 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/DZ_bank_DataSet_classification/README.md#-references) 1 - 4 below).

## 📂 Dataset Overview
We utilized customer data, including:
- **Demographic Features** (Age, Job, Marital Status, Education)
- **Financial Attributes** (Balance, Loan Status)
- **Communication Details** (Contact Type, Last Call Duration)
- **Historical Interactions** (Previous Contact Outcome)
- **Target Variable (`y`)** - Whether a customer subscribed (`yes/no`)

## 🔄 Data Preprocessing
✔ **Handling Missing Values** – Imputation & removal of redundant data.  
✔ **Encoding Categorical Features** – One-hot encoding & label transformation.  
✔ **Scaling & Normalization** – Standardization of numerical variables.  
✔ **Balancing Classes** – SMOTE to improve model fairness.  

## 🤖 Model Implementation
### **Individual Models**
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **XGBoost**

### **Ensemble Learning Techniques**
- **StackingClassifier**
- **Bagging & Boosting**

## 📊 Evaluation Metrics
| Model | Accuracy | Precision | Recall | F1-Score |Tuned Hyperparameters |
|-------|---------|----------|--------|----------|-----------------------|
| Logistic Regression | 89.2% | 91.0% | 85.5% | 88.2% | `C=1.0`, `solver='liblinear'` |
| Decision Tree | 85.7% | 87.4% | 83.1% | 85.2% | `max_depth=10`, `min_samples_split=5` |
| Random Forest | 91.8% | 93.2% | 88.5% | 90.8% |`n_estimators=100`, `max_features='sqrt'` |
| XGBoost | 92.3% | 94.0% | 89.7% | 91.8% |`learning_rate=0.05`, `n_estimators=300`, `max_depth=7` |
| **Stacking Ensemble** | **94.1%** | **95.6%** | **90.9%** | **93.2%** |`class_weight='balanced'`, `n_estimators=200`, `random_state=42` |

## 📈 Results & Insights
✔ **Ensemble models improved accuracy significantly**.  
✔ **Higher recall reduced false negatives** in predictions.  
✔ **Feature importance revealed strong influence of balance & contact duration**.  
✔ **Hyperparameter tuning optimized XGBoost performance**.

## 📷 Visualizations
### 1. XGBoost Confusion Matrix
![XGBoost Confusion Matrix](https://github.com/NenadBalaneskovic/ExternalProjects/blob/161823144626734e4908ae407fa69b84c2deec21/DZ_bank_DataSet_classification/Fig4.PNG)  
### 2. Feature Correlation Heatmap
![Feature Correlation Heatmap](https://github.com/NenadBalaneskovic/ExternalProjects/blob/f2a05f9a2998e0b9992a2c8856a61f4a05c53a3e/DZ_bank_DataSet_classification/Fig12.PNG)  
### 3. Model Accuracy Comparison
![Model Accuracy Comparison](https://github.com/NenadBalaneskovic/ExternalProjects/blob/735f0d2547281074c02e432d3615e20cbf2197b9/DZ_bank_DataSet_classification/Fig3.PNG)  

## 🚀 Model Deployment
✔ **Trained best models can be downloaded and stored as pickle files within the jupyter notebook**. 

## 📚 References
1. **Bank Marketing Dataset** – https://archive.ics.uci.edu/dataset/222/bank+marketing
2. [![Jupyter Notebook | English](https://img.shields.io/badge/Jupyter%20Notebook-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/03d304db3daf8b6e50d33c4706835dbf9eefa9c5/DZ_bank_DataSet_classification/DZ_Bank_HomeAssignment.ipynb)
3. [![Jupyter PDF | English](https://img.shields.io/badge/Jupyter%20PDF-English-yellowblue?logoColor=green&labelColor=blue)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/2c1d8b6a1e675e22a79dd0741c3b4e684d24eb6f/DZ_bank_DataSet_classification/DZ_Bank_HomeAssignment.pdf)
4. [![Classification Report | English](https://img.shields.io/badge/Classification%20Report-English-yellowblue?logoColor=blue&labelColor=red)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/dd2cfa231369855a7f61906c3cf3fb1ed9825042/DZ_bank_DataSet_classification/MarketingDataSetAnalysis.pdf) 
5. **Scikit-Learn Documentation** – https://scikit-learn.org  
6. **XGBoost Library** – https://xgboost.readthedocs.io
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
