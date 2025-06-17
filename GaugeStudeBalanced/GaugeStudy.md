# 1. 🚀 Project Introduction: Balanced Gauge Study Analysis

## Objective  
The purpose of a Gauge Study is to conduct a measurement system capability study that should,
utilizing the ANOVA (analysis of variance) techniques, 1) determine the amount of variability
in the collected data that may be caused by the measurement system, 2) isolate the sources of
variability in the measurement system and 3) assess whether the measurement system is suitable
for use in the broader application. A measurement system is regarded as suitable if it
is repeatable and reproducible (R&R). Repeatability is variability of measurement data arising
from the same unit (i.e. measurement device). Reproducibility is variability of measurement data
arising from different operators (i.e. experimentalists) or devices.  
 
This project aims at designing a PyQt-GUI that would support users in 
**generating balanced one-factor and two-factor Gauge Studies** from well-defined csv input 
data sets containing measurements recorded via the measurement system under consideration 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/GaugeStudeBalanced/GaugeStudy.md#8--references) 1 - 3 below).


# 2. 📏 **Balanced Gauge Studies (One-Factor & Two-Factor)**  

## 🔍 **Introduction**  
Balanced Gauge Studies are **statistical experiments** used to assess the accuracy, repeatability, and reproducibility of a
measurement system. They help identify **operator-dependent variations, equipment precision limitations, and part-specific inconsistencies**, 
ensuring measurement reliability.  

## 📌 **Usefulness of Gauge R&R Studies**  
Gauge R&R (Repeatability & Reproducibility) studies provide:  
✅ **Measurement System Validation** – Confirms whether the system can consistently measure components.  
✅ **Operator Consistency Analysis** – Evaluates whether different users generate similar results.  
✅ **Part Variation Assessment** – Determines the influence of parts on measurement deviations.  
✅ **Process Control & Compliance** – Ensures measurement systems meet industry standards (e.g., ISO, Six Sigma).  


## ⚙️ **Methodology of Gauge Studies**  

### **📊 One-Factor Gauge R&R Study**  
- **Focus:** Evaluates measurement system **repeatability** (variation within operator measurements).  
- **Setup:**  
  - Single-factor experiment (operator only).  
  - Operators measure **generic test samples** (not specific parts).  
  - Measurement variance analyzed across multiple trials.  

#### **Mathematical Model**  
Let:  
- \( \gamma_m \) = Measurement variance  
- \( \gamma_r \) = Repeatability variance (operator-based)  

**Computed Parameters:**  
- **PTR (Precision-to-Tolerance Ratio)**:  
  \[
  PTR = \frac{\gamma_m}{\gamma_m + \gamma_r}
  \]
  - Higher values indicate better **measurement precision**.  

- **SNR (Signal-to-Noise Ratio):**  
  \[
  SNR = \frac{\gamma_m}{\gamma_r}
  \]
  - High SNR suggests that **measurement signal is stable**, while low SNR implies **poor repeatability**.  


### **📊 Two-Factor Gauge R&R Study**  
- **Focus:** Evaluates **reproducibility** (variation across different parts and operators).  
- **Setup:**  
  - Operators measure **multiple parts** across different trials.  
  - Allows **part variability** to be included in the measurement variance model.  

#### **Mathematical Model**  
Let:  
- \( \gamma_p \) = Part variance  
- \( \gamma_m \) = Measurement variance  
- \( \gamma_r \) = Repeatability variance  

**Computed Parameters:**  
- **PTR for Two-Factor Studies**:  
  \[
  PTR = \frac{\gamma_p}{\gamma_p + \gamma_m + \gamma_r}
  \]
  - A high value indicates that **part variability dominates measurement variance**, suggesting part-specific influences.  

- **SNR for Two-Factor Studies:**  
  \[
  SNR = \frac{\gamma_p}{\gamma_m}
  \]
  - Determines **whether part variations significantly affect measurement stability**.  


## 🏭 **Practical Aspects of Balanced Gauge Studies**  

### **✅ When to Use One-Factor vs. Two-Factor Studies**  
| **Study Type** | **Best Use Case** | **Key Focus** |
|---------------|----------------|--------------|
| **One-Factor Study** | Evaluating operator precision | Repeatability |
| **Two-Factor Study** | Assessing both operator and part variability | Reproducibility |

### **🔬 Key Practical Steps**  
1️⃣ **Define Study Type** – Decide whether to assess **operator-only (one-factor)** or **operator + part variation (two-factor)**.  
2️⃣ **Prepare Test Samples** – Ensure consistency in sample preparation to avoid bias.  
3️⃣ **Measure Across Trials** – Multiple trials are essential for statistical robustness.  
4️⃣ **Analyze Variance Components** – Compute **gamma values (measurement, repeatability, part variability)**.  
5️⃣ **Interpret SNR & PTR** – Evaluate system reliability and determine if measurement corrections are needed.  


## 🚀 **Conclusion**  
Balanced Gauge Studies are essential for ensuring **measurement accuracy**, **system stability**, and **operator consistency**. 
By leveraging **PTR, SNR, and variance analysis**, industries can **optimize measurement processes, improve quality control, and enhance data-driven decision-making**.  

---


# 3. 📌 **One-Factor Gauge R&R Study CSV Structure**
A one-factor study **does not include part variability** and only tracks operator performance and measurement repeatability.

## **CSV Format**
```csv
Operator,Part,Trial,Measured Value
A,N/A,1,101.23
B,N/A,1,98.76
C,N/A,2,100.45
D,N/A,2,102.14
A,N/A,3,99.87
B,N/A,3,100.58
C,N/A,4,97.65
D,N/A,4,101.22
```

## **Column Breakdown**
| Column Name       | Description |
|------------------|-------------|
| **Operator**      | The technician performing the measurement. |
| **Part**         | `"N/A"` (One-factor studies do not analyze part variation). |
| **Trial**        | Measurement sequence (Trial 1, Trial 2, etc.). |
| **Measured Value** | Actual recorded measurement for analysis. |

✅ **Key Characteristics**:
- **No part variance**, since this study examines **repeatability within operators**.  
- **Operators measure generic test samples**, and results are tracked **across trials**.

---

# 4. 📌 **Two-Factor Gauge R&R Study CSV Structure**
A two-factor study **includes part variability**, allowing measurement precision to be analyzed across different parts.

## **CSV Format**
```csv
Operator,Part,Trial,Measured Value
A,P1,1,101.23
A,P2,1,98.76
B,P1,2,100.45
B,P2,2,102.14
C,P1,3,99.87
C,P2,3,100.58
D,P1,4,97.65
D,P2,4,101.22
```

## **Column Breakdown**
| Column Name       | Description |
|------------------|-------------|
| **Operator**      | The technician performing the measurement. |
| **Part**         | Identifies the **specific part** being measured (e.g., `"P1"` or `"P2"`). |
| **Trial**        | Measurement sequence (Trial 1, Trial 2, etc.). |
| **Measured Value** | Actual recorded measurement for analysis. |

✅ **Key Characteristics**:
- **Includes part variability**, tracking **measurement deviation across different parts**.  
- **Used when assessing both repeatability and reproducibility**, ensuring operators can consistently measure different components.

---


# 5. 📏 **Gauge Study GUI: Algorithmic & Mathematical Foundations**  

## 🔍 **Overview**  
This **Gauge R&R GUI** performs **repeatability & reproducibility analysis**, statistical validation, and process evaluation for measurement systems. The core functionalities include:  
✅ **CSV Handling & Data Validation**  
✅ **Statistical Calculations & Confidence Intervals**  
✅ **Gauge Precision Metrics (PTR, SNR, Cp, Tolerance Ratio)**  
✅ **Graphical Representation & Result Visualization**  
✅ **XAI-Based Parameter Explanation & PDF Report Generation**  


## 📂 **CSV Handling & Validation**  

### **CSV Loading (`load_csv()`)**  
- Uses **`pandas.read_csv()`** to **parse measurement data**, ensuring correct column structure.  
- Converts `"Measured Value"` column into **numeric format** (`pd.to_numeric()` with `errors="coerce"`).  

### **Data Validation (`validate_csv()`)**  
- Checks for **missing values**, **non-numeric entries**, and **duplicate records**.  
- Ensures that:  
  - **One-factor studies** contain `"Part" = "N/A"` (no part variability).  
  - **Two-factor studies** include **distinct `"Part"` values**.  
- Uses **variance checks** (`df["Measured Value"].var()`) for structure validation.  


## 📊 **Statistical Calculations & Confidence Intervals**  

### **Mean Measurement Value (`mu_Y`)**  
Computed as:  
\[
\mu_Y = \frac{1}{N} \sum_{i=1}^{N} X_i
\]  
where \( X_i \) are **measurement values**, and \( N \) is the total number of trials.

### **Variance Components (`gamma_p`, `gamma_m`, `gamma_r`)**  
Defines the **three major sources of measurement variability**:  
- **Part Variance** \( \gamma_p \) → If study is two-factor:  
  \[
  \gamma_p = \max \left( \text{Var}(\text{Part}) \right), \epsilon \right)
  \]
- **Measurement Variance** \( \gamma_m \) → Estimated from all measurement trials:  
  \[
  \gamma_m = \max \left( \text{Var}(\text{Measured Value}), \epsilon \right)
  \]
- **Repeatability Variance** \( \gamma_r \) → Determined by trial differences:  
  \[
  \gamma_r = \max \left( \text{Var}(\text{Trial}), \epsilon \right)
  \]

**Note:**  
- If **one-factor study**, **part variance is ignored** \( (\gamma_p = 0) \).  
- \( \epsilon = 1e^{-6} \) ensures no zero variance (avoiding divide-by-zero errors).  


## 📈 **Gauge Precision Metrics**  

### **PTR (Precision-to-Tolerance Ratio)**  
Evaluates **measurement precision compared to overall variance**:  
\[
PTR = \frac{\gamma_m}{\gamma_m + \gamma_r}
\]  
- **Higher PTR** → Measurement system is **highly precise**.  
- **Lower PTR** → Measurement noise affects results significantly.  

**Two-Factor Variation:**  
\[
PTR = \frac{\gamma_p}{\gamma_p + \gamma_m + \gamma_r}
\]  
where **part variability** influences measurement reliability.


### **SNR (Signal-to-Noise Ratio)**  
Measures the **stability of measurement precision** vs repeatability noise:  
\[
SNR = \frac{\gamma_m}{\gamma_r}
\]
- **High SNR** → Repeatability variance is small, and measurement precision is stable.  
- **Low SNR** → System suffers from unreliable repeatability.  

**Two-Factor Variation:**  
\[
SNR = \frac{\gamma_p}{\gamma_m}
\]
showing **how much part variation contributes to system precision**.


### **Cp (Process Capability Index)**  
\[
Cp = 1.33 \times \sqrt{PTR}
\]
- **Used in Six Sigma methodologies** to evaluate **system efficiency**.  
- **Higher Cp** → Measurement system operates within acceptable error margins.  


### **Tolerance Ratio**  
\[
Tolerance Ratio = \frac{\gamma_m}{\gamma_p + \gamma_m + \gamma_r}
\]
- **If too high** → Suggests measurement system has **excessive variation** impacting reliability.  
- Helps determine if a **more robust measurement method is needed**.  


## 🚦 **XAI-Based Classification & Region Mapping**  

### **Dynamic Thresholds for Red/Yellow/Green Classification**  
Based on **study type (one-factor or two-factor)**:  
```python
if is_one_factor:
    green_threshold_ptr = 0.000080
    yellow_threshold_ptr = 0.000078
    red_threshold_ptr = 0.000076
else:
    green_threshold_ptr = 0.000108
    yellow_threshold_ptr = 0.000106
    red_threshold_ptr = 0.000105
```
This ensures that **SNR & PTR are correctly classified** into:  
✅ **Green Zone (Good Precision)**  
🟡 **Yellow Zone (Moderate Precision)**  
🔴 **Red Zone (Poor Precision)**  


## 🔬 **Graphical Visualization & Algorithmic Processing**  

### **Box Plots (Repeatability)**
```python
ax1.boxplot(df["Measured Value"])
ax1.set_title("Repeatability Across Parts")
```
**Key Interpretation:**  
- **Wide interquartile range (IQR)** → Large **repeatability variation**.  
- **Narrow IQR** → More stable measurements.

### **Histogram (Measurement Distribution)**
```python
ax2.hist(df["Measured Value"], bins=10, color="skyblue", edgecolor="black")
ax2.set_title("Distribution of Measured Values")
```
**Key Interpretation:**  
- **Uniform distribution** → System is **stable**.  
- **Skewed distribution** → Suggests **bias or systematic errors**.  


### **Variance Contribution Chart**
```python
ax3.bar(categories, values, color=["blue", "orange", "green"])
ax3.set_title("Variance Contribution")
```
Visualizes **how much each factor contributes to total variance**, guiding **process control improvements**.


### **PTR vs SNR Classification**
```python
ax4.scatter(ptr_values[green_mask], snr_values[green_mask], color="green", alpha=0.3)
ax4.scatter(ptr_values[yellow_mask], snr_values[yellow_mask], color="yellow", alpha=0.3)
ax4.scatter(ptr_values[red_mask], snr_values[red_mask], color="red", alpha=0.3)
```
**Key Interpretation:**  
- **If measurement noise dominates**, system moves toward 🔴 **Red Zone**.  
- **If precision dominates**, system stays in 🟢 **Green Zone**.  


### **Beta & Delta Index GPQ Convergence**
```python
gpq_samples = np.random.multivariate_normal([gamma_p, gamma_m], [[gamma_p, 0], [0, gamma_m]], size=500)
beta_index = round(np.mean(sampled_gamma_p / sampled_gamma_m), 3)
delta_index = round(np.mean(sampled_gamma_m / gamma_r), 3)
```
- **Beta Index** → Measures **system bias**.  
- **Delta Index** → Measures **inconsistency between trials**.  


## 📜 **Report Generation & XAI Explanation**  

### **PDF Report (`generate_pdf_report()`)**
- Uses **`ReportLab.canvas`** to compile results **with structured insights**.  
- Generates an **interactive report**, including:  
  - ✅ **Parameter Results & Confidence Intervals**  
  - ✅ **Graphical Visualizations**  
  - ✅ **XAI-Based Explanation (Explainability of Results)** 

---  

# 6. 🎯 **Final Thoughts**
This Gauge R&R GUI **integrates advanced statistical concepts**, ensuring:  

✅ **Accurate Measurement System Evaluation**  
✅ **Precision Metrics & Variability Analysis**  
✅ **Clear Graphical Representations & Automated Reports**  

---

# 7. Use-Cases

## 7.1 Case 1: One-factor Gauge Study

![One-factor balanced Gauge Study results](https://github.com/NenadBalaneskovic/ExternalProjects/blob/c76ac6c42d997b43a76f073fb4bf4bfc122d9adc/GaugeStudeBalanced/one_factor_gauge_green.PNG)


### 📊 **Interpretation of the One-Factor Gauge Study Result**

Our **one-factor Gauge R&R study** assesses **repeatability** in a measurement system by analyzing how consistently measurements are recorded across trials, without considering part variability.

### 🔍 **Key Statistical Findings**
**Overall Results:**  
- **Mean Value:** **100.143** → Represents the average measurement across all trials.  
- **Total Variance:** **8.003** → The combined variability within the system.  
- **SNR (Signal-to-Noise Ratio):** **0.989** (95% CI: 0.866 - 1.112)  
  - Indicates **how well measurement precision is maintained** compared to noise.  
  - **SNR ≈ 1 suggests that noise and measurement variation are balanced.**  
- **Capability Index (Cp):** **0.938** (95% CI: 0.815 - 1.061)  
  - Determines if the system can **reliably measure within tolerance limits**.  
  - **Cp near 1 is generally acceptable, but improvements may be needed.**  
- **Tolerance Ratio:** **0.497** (95% CI: 0.374 - 0.620)  
  - Measures how much **measurement variance contributes to total variability**.  
  - **A value near 0.5 suggests measurement variability is substantial but not dominant.**  

**Variance Components:**  
- **Measurement Variance (`γM`):** **3.979** → **Moderate variability** in readings across trials.  
- **Repeatability (`γR`):** **4.023** → **Similar to measurement variance**, showing a **balanced system**.  
- **PTR (Precision-to-Tolerance Ratio):** **0.497** → Suggests **measurement precision accounts for ~50% of system variability**.  
- **Part Variance (`γP`):** **0.000** → **Expected in a one-factor study**, since part variability is **not analyzed**.  

### 📉 **Graphical Interpretation**
- **Box Plot (Repeatability Across Parts):**  
  - If whiskers are **short**, repeatability is **good**.  
  - If whiskers are **long**, variability **between trials is high**.  
- **Histogram (Distribution of Measured Values):**  
  - A **symmetrical shape** indicates **stable measurement distributions**.  
  - A **skewed shape** suggests possible **bias or systematic errors**.  
- **Variance Contribution Chart:**  
  - Since **γP = 0**, **measurement & repeatability variance dominate**.  
  - If **γM > γR**, measurements suffer from **system noise** rather than operator inconsistencies.  

### 🚦 **PTR-SNR Classification & Reliability**
Your **PTR-SNR values fall in the Green Zone**, which means:  
✅ **The measurement system is stable and performs well** under repeatability tests.  
⚠ If **higher precision is required**, further reduction in **repeatability variance (`γR`)** may be beneficial.  

Overall, this Gauge R&R result suggests that **the measurement system is reliable**, with **some room for improvement in repeatability consistency**.

## 7.2 Case 2: One factor Gauge Study with corrupt and/or missing measurement entries 

![Two-factor balanced Gauge Study results](https://github.com/NenadBalaneskovic/ExternalProjects/blob/100f6c03a6d8c9b7298ec33a88608186b949083d/GaugeStudeBalanced/two_factor_gauge_green_corrupt.PNG)  

### 📊 **Interpretation of the Two-Factor Gauge Study Result & Handling of Missing Data**  

This **two-factor Gauge R&R study** analyzes **repeatability and reproducibility**, incorporating **part variability** into the measurement system assessment. Unlike a one-factor study, this analysis includes **variability introduced by different parts** alongside operator-dependent errors.

### 🔍 **Key Statistical Findings**  

#### **General Measurement Results**  
- **Mean Value (μγ):** **127.644** → Represents the average recorded measurement across trials.  
- **Total Variance (σ²):** **72689.115** → Overall system variability, incorporating **measurement, part, and repeatability sources**.  

#### **PTR-SNR Metrics**  
- **PTR (Precision-to-Tolerance Ratio):** **0.332** (95% CI: -9.527 to 10.191)  
  - Since **PTR < 0.5**, **measurement precision accounts for less than half of total variability**, suggesting significant **part or repeatability influence**.  
- **SNR (Signal-to-Noise Ratio):** **1.000** (95% CI: -0.859 to 1.089)  
  - **SNR ≈ 1 implies balanced noise-to-signal ratio**, meaning part influence and measurement variability are nearly equal.  
  - However, the wide **confidence interval suggests some instability** in reliability across trials.  

#### **Process Capability & Variability**  
- **Cp (Process Capability Index):** **0.796** (95% CI: -9.093 to 10.625)  
  - Suggests **moderate process control**, but **variation may impact repeatability precision**.  
- **Tolerance Ratio:** **0.332** (95% CI: -9.527 to 10.191)  
  - A lower tolerance ratio indicates that **measurement variance contributes to variability without completely dominating the system**.  

#### **Variance Components**  
- **Part Variance (γP):** **24106.947** → Shows how **different parts affect measurement consistency**.  
- **Measurement Variance (γM):** **24117.833** → Indicates **how much measurement variation occurs independently of parts**.  
- **Repeatability Variance (γR):** **24464.334** → Suggests **operator inconsistencies or system noise contributing to total error**.  

### 📉 **Graphical Interpretation**  

- **Box Plot (Repeatability Across Parts)** → Identifies **whether part variability dominates operator errors**.  
- **Histogram (Measured Value Distribution)** → If skewed, **suggests possible measurement bias or systematic errors**.  
- **Variance Contribution Chart** → Confirms **relative dominance of part, measurement, and repeatability variance**.  
- **PTR-SNR Plot** → Indicates **measurement reliability vs noise**.  
- **Beta & Delta Index Convergence** → Highlights **bias/inconsistency trends over GPQ iterations**.  

### 🚦 **Handling of Missing/Corrupt Measurement Entries**  

🔍 **Observations:**  
- The **"Exclude Missing Data"** option is enabled.  
- The **Data Preview table contains `"nan"` values in the `"Part"` column**, suggesting incomplete records.  
- Despite this, the log indicates **"Validation successful: No issues detected."**, meaning missing values were **properly excluded** from statistical calculations.  

✅ **How the GUI Handles Missing Data:**  
- **Rows with `"nan"` in critical columns (Measured Value, Part, Operator) are excluded** before calculations.  
- **Repeatability variance (`γR`) is computed only across valid data points**, ensuring reliability.  
- **Confidence intervals reflect statistical uncertainty from cleaned data**, preventing errors from missing values.

✅ **All stored results can be accessed via the following links:**  
1. A ![csv file](https://github.com/NenadBalaneskovic/ExternalProjects/blob/1cdd76715308f6a7970d5a0852e0adc286666f31/GaugeStudeBalanced/results.csv) of extracted Gauge Study parameters.
2. A ![pdf report](https://github.com/NenadBalaneskovic/ExternalProjects/blob/6d3669798335f1ee4f07293aad3b1bfdd020a035/GaugeStudeBalanced/report_2_factor_GaugeStudy.pdf) of obtained Gauge Study results.

# 8. 📚 References
1. R. K. Burdick, C. M. Borror, D. C. Montgomery: "__Design and Analysis of Gauge R&R Studies__", 1st Ed. SIAM (2005); 
S. H. Derakhshan , C. V. Deutsch: "__Numerical Integration of Bivariate Gaussian Distribution__", Paper 405, CCG Anual Report 13 (2011).
2. [![Jupyter Notebook | English](https://img.shields.io/badge/Jupyter%20Notebook-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/6d3669798335f1ee4f07293aad3b1bfdd020a035/GaugeStudeBalanced/GaugeStudyGUI.ipynb)
3. [![Forecasting Report | English](https://img.shields.io/badge/SARIMAX%20Report-English-yellowblue?logoColor=blue&labelColor=red)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/3a07dee498fa12cef3d92f4dcaf146032365b442/SARIMAX_Forecasting/SARIMAX_BoarderCrossingReport.pdf) 
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

