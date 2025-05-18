# Sentiment Analysis Project - News Sentiment Evaluation

## 📌 Introduction
This project analyzes sentiments of news headlines contained within the Groundnews website by means of a customized Pythonic GUI 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/tree/main/SentimentAnalysis_NewsHeadlines#-references) 1 - 3 below).

### **Overview**
The **News Sentiment Analysis GUI** is a **PyQt5-based application** designed to analyze sentiment trends of **news headlines fetched from Ground News**. 
By leveraging **natural language processing (NLP)** tools like **NLTK and SpaCy**, alongside **web scraping via BeautifulSoup**, the project aims to 
provide **valuable insights into Python’s NLP capabilities and real-world text analysis**.  

The goal of this project is to **experiment with and understand the inner workings** of Python's **NLTK** and **SpaCy** libraries in conjunction with
 **web scraping using BeautifulSoup**. Through sentiment analysis of **real-time headlines**, the tool demonstrates how different NLP techniques can be
 used to quantify **public sentiment, keyword extraction, and trend visualization** in an interactive interface.  

## **Key Features**
✔ **Automated News Fetching** → Fetches headlines from `Ground News` dynamically.  
✔ **Sentiment Analysis** → Uses `NLTK VADER` and `SpaCy` to determine sentiment polarity scores.  
✔ **Keyword Extraction** → Identifies named entities from text for analysis.  
✔ **Word Cloud Generation** → Visual representation of frequently occurring keywords.  
✔ **Sentiment Trend Plot** → Displays sentiment changes over multiple headlines.  
✔ **Data Export Options** → Saves results as `CSV` or `PNG`. 

## **Technical Components**  

### **1️⃣ NewsFetcher (Threaded Data Fetching)**
✔ **Asynchronously retrieves headlines from an online source** (`Ground News`)  
✔ Uses **BeautifulSoup** for parsing HTML content  
✔ Extracts **news categories and headlines**  

### **2️⃣ SentimentAnalysis (Text Processing)**
✔ Uses **NLTK VADER** to calculate **sentiment polarity scores** (`Positive, Negative, Neutral`)  
✔ Implements **SpaCy Named Entity Recognition (NER)** for keyword extraction  

### **3️⃣ NewsSentimentGUI (GUI Interface)**
✔ **PyQt5-based user interface** for interaction  
✔ Displays **news headlines dynamically based on selected categories**  
✔ **Sentiment analysis activation through button interaction**  
✔ Generates **Word Cloud** using `WordCloud`  
✔ Visualizes **sentiment trends** (**Real-time trend visualization**) using `PyQtGraph` 

## **Functional Workflow**
1️⃣ **User selects a news category** → Headlines are fetched dynamically  
2️⃣ **User clicks a headline** → Sentiment analysis is performed  
3️⃣ **Sentiment Score is displayed** → Positive/Negative/Neutral sentiment  
4️⃣ **Word Cloud is generated** → Extracted keywords visualized  
5️⃣ **Sentiment Trend updates** → Graph dynamically tracks sentiment changes  
6️⃣ **Results can be exported** → CSV for sentiment scores or PNG for word clouds  

## **Libraries Used**
- `PyQt5` → GUI framework for interactive visualization  
- `NLTK & SpaCy` → NLP-based sentiment analysis & keyword extraction  
- `WordCloud` → Text-based keyword visualization  
- `BeautifulSoup` → HTML parsing for fetching news  
- `PyQtGraph` → Real-time trend visualization  
- `Pandas` → Data processing & CSV export functionality  

## 📷 Visualizations
![Sketch of the Sentiment Analysis GUI](confusion_matrix.png)  
![Implemented Sentiment Analysis GUI](feature_importance.png)  
![A csv file with the Latest Sentiment Result](roc_curve.png)  

## 🚀 GUI Deployment & Improvements
✔ **The Pythonic GUI for news sentiment analysis can be packaged into an exe-file via autopy2exe if needed**.  
✔ **Future GUI-improvements may involve the following aspects**:  
  
### **1️⃣ Performance Optimization**
🔹 **Use Async Requests for News Fetching** → Replace `urlopen()` with **`requests` + `asyncio`** to improve speed & prevent UI freezing.  
🔹 **Optimize Sentiment Analysis Execution** → Cache previously analyzed headlines to prevent redundant computations.  
🔹 **Reduce Memory Usage for WordCloud** → Instead of saving temporary PNGs, directly render images into `QPixmap` using `BytesIO`.

### **2️⃣ UI Enhancements & User Experience**
🔹 **Real-time Sentiment Updates** → Add a **live refresh button** to re-fetch current news for updated analysis.  
🔹 **Interactive Graph Elements** → Allow users to **hover over sentiment points** to display details (e.g., related news headline).  
🔹 **Better Word Cloud Visibility** → Add **zoom functionality** for clearer keyword visualization.

### **3️⃣ Advanced NLP Features**
🔹 **Sentiment Comparison Across Categories** → Users could compare **multiple news categories** side by side.  
🔹 **Deeper Text Analysis** → Implement **topic modeling (LDA)** to detect themes within headlines.  
🔹 **Multi-language Support** → Expand sentiment analysis to work with **non-English headlines** using SpaCy’s multilingual models.

### **4️⃣ Data Storage & Export**
🔹 **Database Integration** → Store analyzed headlines in **SQLite/PostgreSQL** for historical trend tracking.  
🔹 **Generate Reports Automatically** → Users could export **summarized sentiment trends** to PDFs.  
🔹 **Advanced Filtering** → Enable search/filter functionality to analyze **specific keywords across headlines**.

## **Conclusion**
The **News Sentiment Analysis GUI** provides an intuitive and **data-driven approach** to analyzing news sentiment in real-time. 
By integrating **advanced NLP techniques, graphical visualization, and dynamic data retrieval**, it serves as a powerful tool for extracting **emotional insights from online news**.
The **News Sentiment Analysis GUI** serves as a hands-on exploration into **Python's NLP capabilities** using **NLTK and SpaCy**, combined with **web scraping via BeautifulSoup**. 
By analyzing **real-world news headlines**, the project demonstrates how **machine learning and NLP techniques** can quantify sentiment and **extract meaningful insights** from textual data in an interactive framework.

## 📚 References
1. **Ground News Website** – https://ground.news/
2. [![Jupyter Notebook | English](https://img.shields.io/badge/Jupyter%20Notebook-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/3a07dee498fa12cef3d92f4dcaf146032365b442/SARIMAX_Forecasting/CargoDataSet_Analysis.ipynb)
3. [![Forecasting Report | English](https://img.shields.io/badge/SARIMAX%20Report-English-yellowblue?logoColor=blue&labelColor=red)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/3a07dee498fa12cef3d92f4dcaf146032365b442/SARIMAX_Forecasting/SARIMAX_BoarderCrossingReport.pdf) 
4. **NLTK Documentation** – https://www.nltk.org/
5. **SpaCy Documentation** – https://pypi.org/project/spacy/, https://spacy.io/
6. **Beautiful Soup & WordCloud Documentation** - https://pypi.org/project/beautifulsoup4/, https://www.crummy.com/software/BeautifulSoup/bs4/doc/, https://pypi.org/project/wordcloud/, https://amueller.github.io/word_cloud/
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
---
