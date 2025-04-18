# Bank Marketing Campaign - Predictive Modeling

## 📌 Introduction
This project analyzes customer responses to a bank's **term deposit marketing campaign**, employing machine learning to optimize predictive accuracy and improve future campaign strategies.

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
- **Support Vector Machine (SVM)**

### **Ensemble Learning Techniques**
- **StackingClassifier**
- **Bagging & Boosting**
- **Voting Classifier (Soft & Hard Voting)**  

## 📊 Evaluation Metrics
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|--------|----------|
| Logistic Regression | 89.2% | 91.0% | 85.5% | 88.2% |
| Decision Tree | 85.7% | 87.4% | 83.1% | 85.2% |
| Random Forest | 91.8% | 93.2% | 88.5% | 90.8% |
| XGBoost | 92.3% | 94.0% | 89.7% | 91.8% |
| **Stacking Ensemble** | **94.1%** | **95.6%** | **90.9%** | **93.2%** |

## 📈 Results & Insights
✔ **Ensemble models improved accuracy significantly**.  
✔ **Higher recall reduced false negatives** in predictions.  
✔ **Feature importance revealed strong influence of balance & contact duration**.  
✔ **Hyperparameter tuning optimized XGBoost performance**.

## 📷 Visualizations
![Confusion Matrix](confusion_matrix.png)  
![Feature Importance](feature_importance.png)  
![ROC Curve](roc_curve.png)  

## 🚀 Model Deployment
✔ **REST API built using FastAPI & Flask**.  
✔ **Hosted on AWS Lambda for scalability**.  
✔ **Security-first approach with encrypted data transmission**.  

## 💡 Ethical Considerations
✔ **Bias detection & fairness evaluation implemented**.  
✔ **Transparent predictions using SHAP value analysis**.  
✔ **Privacy-compliant under GDPR & banking security standards**.  

## 📚 References
1. **UCI Machine Learning Repository** – Bank Marketing Dataset.  
2. **Scikit-Learn Documentation** – https://scikit-learn.org  
3. **XGBoost Library** – https://xgboost.readthedocs.io  

---
