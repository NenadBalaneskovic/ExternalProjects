<p align="center">
    <img src="https://github.com/NenadBalaneskovic/ExternalProjects/blob/409b16096842f543f5683752248f8227b1db3891/Rastergrafik.png" width="1000" height="800">
</p>    

# External Projects  [![GitHub Wiki](https://img.shields.io/badge/GitHub-External_Wiki-blue?style=flat&logo=github)](https://github.com/NenadBalaneskovic/ExternalProjects/wiki)  [![Static Badge](https://img.shields.io/badge/Main%20Profile%20-%20yellow?logo=github&labelColor=green&color=violet)](https://github.com/NenadBalaneskovic) [![GitHub Main Wiki](https://img.shields.io/badge/GitHub-Main_Wiki-green?style=flat&logo=github)](https://github.com/NenadBalaneskovic/NenadBalaneskovic/wiki/External-ML-AI-Projects)

This is a public folder containing some of my external Pythonic ML projects and analyses.  

## 🌟 [Dataset Analysis Links](https://github.com/NenadBalaneskovic/NenadBalaneskovic/wiki/External-ML-AI-Projects)

- ### 1. **Bank Marketing DataSet (Classification) - Feb 2025**

> ## Executive Summary: Bank Marketing Campaign Predictive Modeling
> 
> **Business Problem**  
> Banks face low conversion rates in term deposit marketing campaigns because customer targeting is often imprecise. Inefficient targeting increases campaign costs and reduces ROI.  
> 
> **AI/ML Solution**  
> This project applies machine learning models (Logistic Regression, Decision Trees, Random Forest, XGBoost, and ensemble methods) to predict customer responses. Data preprocessing techniques (encoding, scaling, SMOTE balancing) and hyperparameter tuning ensure robust and fair models. Feature importance and correlation analysis provide transparency into key drivers of customer decisions.  
> 
> **Business Impact**  
> The solution improves campaign efficiency by identifying high‑probability customers, reducing wasted marketing spend, and increasing deposit uptake. It enables data‑driven decision‑making for campaign managers, directly supporting revenue growth and customer engagement strategies.  
> 
> **Consulting Relevance**  
> This project demonstrates how advanced analytics can be embedded into financial services marketing. It provides a replicable framework for advisory work in customer segmentation, predictive targeting, and marketing ROI optimization — highly relevant for consulting engagements in banking and financial services.  
> 
> **Compliance / ESG / Risk Management**  
> By using balanced datasets and transparent evaluation metrics (precision, recall, F1‑score), the project mitigates bias and supports compliance with fair marketing practices. It also aligns with ESG principles by promoting responsible customer engagement and reducing unnecessary resource use in campaigns.


  #### Abstract:
  This project analyzes customer responses to a bank's **term deposit marketing campaign**, employing machine learning to optimize predictive accuracy and improve future campaign strategies (see [References](https://github.com/NenadBalaneskovic/ExternalProjects/tree/main/Bank_MarketingDataSet_classification#-references) 1 - 4 below). [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links)
    
  [![Bank Marketing_DataSet](https://img.shields.io/badge/Bank%20Marketing_DataSet%20(Classification)-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/0b628164a6e5e62fbfef9919acf1fb9cc307d7a8/Bank_MarketingDataSet_classification/README.md)

- ### 2. **Boarder Crossings DataSet (SARIMAX Forecasting) - Feb 2025**

> ## Executive Summary: Border Crossing Forecasting – SARIMAX Modeling
> 
> **Business Problem**  
> Accurate forecasting of cross‑border traffic between the USA and Canada is critical for resource allocation, infrastructure planning, and security strategy. Traditional methods often fail to capture seasonal and exogenous factors, leading to inefficiencies in border management.  
> 
> **AI/ML Solution**  
> This project applies SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) to forecast monthly border crossings using Bureau of Transportation Statistics data (1996–2024). FFT analysis identifies periodic patterns, while automated grid search and cross‑validation optimize model parameters. DuckDB and pandas streamline data engineering, ensuring reproducibility and scalability.  
> 
> **Business Impact**  
> The forecasting model enables border authorities and policymakers to anticipate traffic surges, optimize staffing, and reduce wait times. It supports strategic planning for infrastructure investment and enhances operational efficiency in customs and immigration services.  
> 
> **Consulting Relevance**  
> This project illustrates how advanced time‑series modeling can be leveraged in public sector consulting. It provides a framework for advisory work in transportation analytics, resource optimization, and policy design — directly relevant to engagements in government, logistics, and risk advisory.  
> 
> **Compliance / ESG / Risk Management**  
> Forecasting supports compliance with international agreements by ensuring adequate resource allocation for safe and lawful crossings. It contributes to ESG goals by reducing congestion and emissions through better traffic management. From a risk perspective, the model strengthens resilience against unexpected surges, enhancing border security and mitigating operational disruptions.

 
  #### Abstract:
  This project attempts at **forecasting the number of boarder crossings** between USA and Canada based on the corresponding kaggle data set of The Bureau of 
Transportation Statistics (BTS) containing entries from 1996 to 2024, employing Python's SARIMAX forecasing scheme to optimize predictive accuracy and improve future security strategies 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/tree/main/SARIMAX_Forecasting#-references) 1 - 3 below). [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links)
  
  [![Boarder Crossings_DataSet](https://img.shields.io/badge/Boarder_Crossings_DataSet%20(SARIMAX_Forecasting)-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/SARIMAX_Forecasting/README.md)

- ### 3. **Stock Price DataSet (SARIMAX Stock Forecasting) - Mar 2025**

> ## Executive Summary: Stock Price Forecasting – SARIMAX Modeling with Akima Interpolation
> 
> **Business Problem**  
> Financial institutions and investors require accurate forecasting of stock price movements to manage risk, optimize trading strategies, and ensure regulatory compliance. Traditional models often struggle with volatility and non‑linear market dynamics, leading to unreliable predictions.  
> 
> **AI/ML Solution**  
> This project combines SARIMAX forecasting with Akima interpolation to smooth daily stock price trends and applies critical point analysis inspired by physical theories of phase transitions. Synthetic datasets are generated and analyzed with derivative‑based feature extraction to classify market behaviors (e.g., bullish surges, sharp declines). Grid search optimization ensures robust parameter tuning, while visualization tools provide transparency into forecasted trends.  
> 
> **Business Impact**  
> The solution enhances predictive accuracy for stock price evolution, enabling better portfolio management and trading decisions. By identifying critical points of volatility, it supports proactive risk mitigation and improves investor confidence. The methodology can be adapted for scenario analysis and stress testing in financial services.  
> 
> **Consulting Relevance**  
> This project demonstrates how advanced time‑series modeling and interpolation techniques can be applied in financial consulting. It provides a framework for advisory services in capital markets, investment risk analysis, and quantitative strategy development — directly relevant to engagements with banks, asset managers, and regulators.  
> 
> **Compliance / ESG / Risk Management**  
> Forecasting models that incorporate volatility detection strengthen compliance with financial risk disclosure requirements. They also support ESG principles by promoting transparent and responsible investment practices. From a risk management perspective, the approach helps institutions anticipate market shocks and align trading strategies with regulatory standards.


  #### Abstract:
  This project attempts at **forecasting the temporal stock prices evolution** based on a ficticious csv file containing daily stock prices by means of Akima interpolated stock price data subject to SARIMAX and
critical point modeling implemented via the physical theory of critical phenomena (see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/SARIMAX_Akima_Forecasting/README.md#-references)
 1 - 3 below). It introduces pythonic functions for Akima interpolation and critical point extraction from stock price time series, which allow the user to characterise critical inflection points of
 stock price evolution and aid the customized SARIMAX forecasting functions in reliably estimating volatility ranges of unknown stock price changes. The project also compares the regular customized Pythonic SARIMAX 
 functionality with its batched version and delves into its conceptual intricacies. [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links)
  
  [![Stock_Price_DataSet](https://img.shields.io/badge/Stock_Price_DataSet%20(SARIMAX_Stock_Forecasting)-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/1d5b93d956f277d937ef2aa66841608f5b60ba7b/SARIMAX_Akima_Forecasting/README.md)

- ### 4. **Sentiment Analysis DataSet (NLTK, SpaCy, BeautifulSoup) - Mar 2025**

> ## Executive Summary: Sentiment Analysis – News Sentiment Evaluation
> 
> **Business Problem**  
> Organizations and analysts need to understand public sentiment in real time to anticipate market reactions, reputational risks, and policy impacts. Manual monitoring of news headlines is inefficient and fails to capture sentiment trends at scale.  
> 
> **AI/ML Solution**  
> This project implements a customized Python GUI that scrapes headlines from Ground News and applies NLP techniques for sentiment analysis. NLTK VADER provides polarity scoring, SpaCy enables named entity recognition, and keyword extraction plus word clouds highlight dominant themes. Real‑time visualization with PyQtGraph and asynchronous threading ensures interactive and efficient sentiment tracking.  
> 
> **Business Impact**  
> The solution empowers decision‑makers to monitor sentiment shifts across news cycles, enabling proactive communication strategies and risk mitigation. It supports investor relations, corporate communications, and policy analysis by providing actionable insights into public perception.  
> 
> **Consulting Relevance**  
> This project demonstrates how NLP‑driven sentiment analysis can be embedded into advisory services. It provides a replicable framework for consulting engagements in media monitoring, reputational risk assessment, and stakeholder management — directly relevant to clients in finance, government, and corporate sectors.  
> 
> **Compliance / ESG / Risk Management**  
> Automated sentiment evaluation supports compliance with transparency and disclosure requirements by providing unbiased monitoring of public narratives. It contributes to ESG goals by enabling responsible communication strategies and fostering trust with stakeholders. From a risk perspective, the tool helps organizations anticipate reputational threats and align responses with governance frameworks.


  #### Abstract:
  This project analyzes sentiments of news headlines contained within the Groundnews website by means of a customized Pythonic GUI 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/tree/main/SentimentAnalysis_NewsHeadlines#-references) 1 - 3 below).
The **News Sentiment Analysis GUI** is a **PyQt5-based application** designed to analyze sentiment trends of **news headlines fetched from Ground News**. 
By leveraging **natural language processing (NLP)** tools like **NLTK and SpaCy**, alongside **web scraping via BeautifulSoup**, the project aims to 
provide **valuable insights into Python’s NLP capabilities and real-world text analysis**. The goal of this project is to **experiment with and understand the inner workings** of Python's **NLTK** and **SpaCy** libraries in conjunction with
 **web scraping using BeautifulSoup**. Through sentiment analysis of **real-time headlines**, the tool demonstrates how different NLP techniques can be
 used to quantify **public sentiment, keyword extraction, and trend visualization** in an interactive interface. [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links)  
 
  [![Sentiment_Analysis_DataSet](https://img.shields.io/badge/Sentiment_Analysis_DataSet%20(NLTK_SpaCy_BeautifulSoup)-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/5092e58e2599919c1bd9d909be27321b06f80375/SentimentAnalysis_NewsHeadlines/README.md)

- ### 5. **GAN & LLM Chessboard Image DataSet (GAN & LLM) - Apr 2025**

> ## Executive Summary: GAN vs OpenCV Chessboard Reconstruction
> 
> **Business Problem**  
> Image reconstruction is critical in domains such as quality control, surveillance, and digital archiving. Traditional computer vision methods often struggle with obscured or incomplete visual data, leading to inaccuracies in automated inspection and monitoring systems.  
> 
> **AI/ML Solution**  
> This project compares conventional OpenCV techniques (edge detection, contour detection, perspective transformation) with deep learning approaches using Generative Adversarial Networks (GANs). GANs are trained to reconstruct missing or obscured chessboard sections, with adversarial loss optimization ensuring realistic outputs. GPU acceleration via Google Colab enables efficient training and deployment.  
> 
> **Business Impact**  
> The GAN‑based approach demonstrates superior reconstruction accuracy compared to traditional methods, highlighting the potential of deep learning in complex image completion tasks. This translates into improved reliability for automated inspection systems, reduced manual intervention, and enhanced efficiency in industries relying on visual data integrity.  
> 
> **Consulting Relevance**  
> The project showcases how AI can augment or replace traditional computer vision pipelines, offering consulting opportunities in manufacturing, logistics, and digital forensics. It provides a replicable framework for advising clients on when to adopt deep learning solutions versus conventional methods, balancing accuracy, cost, and computational efficiency.  
> 
> **Compliance / ESG / Risk Management**  
> Enhanced image reconstruction supports compliance in regulated industries where accurate visual records are mandatory (e.g., aerospace, automotive). It contributes to ESG goals by reducing waste through automated quality assurance and minimizing resource use in manual inspections. From a risk management perspective, the solution strengthens resilience against data loss or corruption in visual monitoring systems.


  #### Abstract:
  This project aims to compare traditional **OpenCV-based** methods for chessboard image reconstruction with
**Generative Adversarial Network (GAN)-driven** approaches. The goal is to evaluate the effectiveness of deep learning 
in reconstructing **secluded or obscured chessboard sections** more accurately than conventional techniques 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/LLM_GAN_ImageReconstruction/README.md#-references) 1 - 3 below). [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links)
  
  [![LLM_GAN_Chessboard_Image_DataSet](https://img.shields.io/badge/GAN_LLM_Chessboard_Image_DataSet%20(GAN_LLM)-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/LLM_GAN_ImageReconstruction/README.md)

- ### 6. **Real Estate DataSet (SOM, SARIMAX, RandomForest) - Apr 2025**

> ## Executive Summary: Real Estate Data Set Analysis in Flower Hill
> 
> **Business Problem**  
> Real estate markets are complex, with property values influenced by economic cycles, buyer behavior, and urban expansion. Without advanced analytics, municipalities, developers, and investors struggle to forecast price trends, identify growth areas, and manage risk in property transactions.  
> 
> **AI/ML Solution**  
> This project integrates MongoDB for transaction storage, Apache Airflow for DAG‑based automation, and MLflow for experiment tracking. Machine learning models — including SARIMAX for time‑series forecasting, Random Forest for classification, Neural Networks for price prediction, and Kohonen Maps for district clustering — are orchestrated into a structured pipeline. Automated updates and feature engineering ensure scalability and reproducibility.  
> 
> **Business Impact**  
> The framework enables accurate forecasting of property prices, buyer segmentation, and district‑level clustering. It supports better investment decisions, optimized urban planning, and improved resource allocation. By automating workflows, the solution reduces manual effort and accelerates insights for stakeholders in real estate and municipal governance.  
> 
> **Consulting Relevance**  
> This project demonstrates how end‑to‑end ML pipelines can be applied in real estate advisory services. It provides a replicable model for consulting engagements in property valuation, urban development strategy, and investment risk analysis — directly relevant to clients in real estate, banking, and public sector planning.  
> 
> **Compliance / ESG / Risk Management**  
> Transparent forecasting and clustering support compliance with housing market regulations and fair valuation practices. ESG relevance is reflected in sustainable urban expansion planning and equitable buyer segmentation. From a risk management perspective, the system strengthens resilience against market volatility and economic downturns by providing early warning signals and scenario analyses.


  #### Abstract:
  This project involves a large data set related to real estate sales for a fictional town of Flower Hill. The aim is to combine the analysis of this data set with PyMongo,
MLflow, Python (SARIMAX times series forecasting, classification, Neural Networks, Kohonen Maps) and DAG-like process organization of ML-tasks.
Thus, we are blending data engineering, machine learning, forecasting, and process automation into a well-structured framework
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/tree/main/RealEstateAnalysis#-references) 1 - 3 below). [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links)
  
  [![Real_Estate_DataSet](https://img.shields.io/badge/Real_Estate_DataSet%20(SOM_SARIMAX_RandomForest)-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/561392eb414897e5ef3e3a6b36b148ac2fc42153/RealEstateAnalysis/README.md)

- ### 7. **Signal Analysis DataSet (Scipy, Statsmodel, Numpy) - May 2025**

> ## Executive Summary: Advanced Signal Denoising Framework
> 
> **Business Problem**  
> Modern industries such as telecommunications, healthcare, and aerospace rely on high‑fidelity signals for critical operations. Noise interference reduces accuracy, increases operational risk, and can compromise safety in real‑time monitoring systems. Traditional single‑method filtering often fails to adapt to diverse and dynamic noise environments.  
> 
> **AI/ML Solution**  
> This project develops an ensemble‑based noise suppression framework that integrates statistical filtering, variance estimation, autocorrelation‑based denoising, adaptive resampling, and multi‑stage fusion strategies. The system dynamically selects optimal denoising techniques in real time, ensuring robust signal integrity without reliance on deep learning models. RMSE benchmarking and clustering methods provide transparent performance evaluation.  
> 
> **Business Impact**  
> The framework enhances signal fidelity across diverse environments, reducing error rates and improving reliability in mission‑critical applications. It supports cost savings by minimizing false alarms, improving diagnostic accuracy, and reducing downtime in systems dependent on clean signal processing.  
> 
> **Consulting Relevance**  
> This project demonstrates how adaptive signal processing can be embedded into consulting engagements for industries requiring resilient monitoring systems. It provides a replicable model for advisory work in telecommunications optimization, medical device reliability, and aerospace safety — directly relevant to clients seeking robust, real‑time solutions.  
> 
> **Compliance / ESG / Risk Management**  
> Improved signal accuracy supports compliance with regulatory standards in healthcare diagnostics, aviation safety, and telecom quality assurance. ESG relevance is reflected in sustainable resource use by reducing redundant recalibration and manual intervention. From a risk management perspective, the framework strengthens resilience against operational failures caused by noisy or corrupted signals.


  #### Abstract:
  Noise contamination is a fundamental challenge in **signal processing**, affecting the accuracy and reliability of measurements across various domains, 
from biomedical signals to communications and industrial data analysis. This project aims to **experiment with different noise mitigation techniques**, 
particularly **non-deep-learning approaches**, and optimize them to identify the **most flexible noise suppression strategy** 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/SignalNoiseMitigation/Signal_Denoising.md#-references) 1 - 3 below). [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links)
  
  [![Signal Analysis_DataSets](https://img.shields.io/badge/Signal_Analysis_DataSet%20(Scipy_Statsmodel_Numpy)-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/SignalNoiseMitigation/Signal_Denoising.md)

- ### 8. **Qiskit Algorithmic Implementations (Qiskit, VQE, QAOA) - May 2025**

> ## Executive Summary: Quantum Optimization with Qiskit
> 
> **Business Problem**  
> Complex combinatorial optimization problems — such as portfolio allocation, logistics routing, and resource scheduling — are computationally intensive and often exceed the scalability of classical methods. Organizations need faster, more efficient approaches to remain competitive in data‑driven decision environments.  
> 
> **AI/ML Solution**  
> This project leverages Qiskit to implement quantum optimization techniques, including the Variational Quantum Eigensolver (VQE), Quantum Approximate Optimization Algorithm (QAOA), Grover’s search, and Quantum Fourier Transform (QFT). Hybrid quantum‑classical workflows combine classical preprocessing with quantum execution on IBM Quantum hardware and simulators, ensuring practical applicability. Benchmarking against classical optimization methods highlights efficiency gains, while error mitigation techniques improve reliability.  
> 
> **Business Impact**  
> Quantum optimization accelerates decision‑making in domains where classical algorithms are bottlenecked, enabling faster portfolio risk balancing, supply chain optimization, and scheduling. The framework demonstrates how quantum computing can reduce computational costs and unlock new levels of scalability for enterprise operations.  
> 
> **Consulting Relevance**  
> This project provides a replicable model for advisory services in emerging technology adoption. It illustrates how consulting firms can guide clients in evaluating quantum readiness, integrating hybrid workflows, and identifying high‑value use cases in finance, logistics, and risk management.  
> 
> **Compliance / ESG / Risk Management**  
> Quantum optimization supports compliance by enabling transparent, auditable decision processes in regulated industries. ESG relevance is reflected in resource efficiency, as quantum methods reduce energy consumption compared to brute‑force classical approaches. From a risk management perspective, the framework strengthens resilience by providing faster scenario analysis and more robust optimization under uncertainty.


  #### Abstract:
  This Kaggle course notebook provides an **introductory guide to Qiskit**, focusing on **quantum computing fundamentals, hands-on coding, and interactive exercises**. 
The aim is to combine the analysis of quantum circuits with their practical applications (see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/Quantum_Computing_Qiskit/QuantumComputingIntro.md#-references) 1 - 3 below). [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links)
  
  [![Qiskit_Algorithmic_Implementations](https://img.shields.io/badge/Qiskit_Algorithmic_Implementations%20(Qiskit_VQE_QAOA)-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/a8f3db4a3eddc6dd88ea0840c118666e24bdf5bd/Quantum_Computing_Qiskit/QuantumComputingIntro.md)

 - ### 9. **Balanced Gauge Study Implementations (ANOVA) - Jun 2025**

> ## Executive Summary: Balanced Gauge Study Analysis
> 
> **Business Problem**  
> Manufacturing and engineering organizations depend on accurate measurement systems to ensure product quality and compliance with industry standards. Without systematic evaluation of repeatability and reproducibility, measurement errors can propagate, leading to costly defects, regulatory issues, and reduced customer trust.  
> 
> **AI/ML Solution**  
> This project develops a PyQt‑based GUI for conducting Balanced Gauge Studies using ANOVA variance decomposition. The tool evaluates repeatability (same operator/device) and reproducibility (different operators/devices), providing interactive data handling, statistical visualization, and automated report generation. Metrics such as Precision‑to‑Tolerance Ratio (PTR), Signal‑to‑Noise Ratio (SNR), and Process Capability Index (Cp) benchmark measurement reliability.  
> 
> **Business Impact**  
> The framework enables organizations to validate measurement systems quickly and consistently, reducing variability in production processes. It supports cost savings by minimizing rework, improving quality assurance, and ensuring that measurement systems meet industrial standards. Automated reporting enhances transparency and accelerates decision‑making in quality control environments.  
> 
> **Consulting Relevance**  
> This project illustrates how statistical quality control can be embedded into advisory services. It provides a replicable model for consulting engagements in manufacturing, engineering, and industrial risk management — helping clients strengthen measurement reliability, optimize processes, and meet compliance requirements.  
> 
> **Compliance / ESG / Risk Management**  
> Balanced Gauge Studies support compliance with ISO and Six Sigma quality standards by ensuring measurement accuracy. ESG relevance is reflected in sustainable production practices, as improved measurement reduces waste and resource inefficiency. From a risk management perspective, the tool mitigates operational risks by identifying sources of measurement error early and ensuring consistent product quality across operators and devices.


   #### Abstract:
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
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/GaugeStudeBalanced/GaugeStudy.md#8--references) 1 - 3 below). [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links)  

   [![Balanced_Gauge_Study_Implementations](https://img.shields.io/badge/Gauge_Study_Implementations%20(ANOVA)-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/effb0dc46f8337d1581acc7b31683be826d72ec6/GaugeStudeBalanced/GaugeStudy.md)

- ### 10. **Dagster-driven Encryption-Decryption Pipeline (Pycryptodome) - Jun 2025**

> ## Executive Summary: Encryption-Decryption Pipeline with Dagster
> 
> **Business Problem**  
> Organizations handling sensitive data require secure, auditable, and reproducible encryption workflows. Traditional cryptographic implementations often lack modularity, observability, and integration with enterprise data pipelines, making them difficult to validate and scale in compliance‑driven environments.  
> 
> **AI/ML Solution**  
> This project builds a layered encryption‑decryption pipeline orchestrated with the Dagster framework. Classical (Vigenère) and modern cryptographic techniques (AES‑256 in CBC mode, RSA‑OAEP key encapsulation, optional SHA‑256 signing) are combined into a defense‑in‑depth architecture. Modular Dagster ops provide observability, retry capability, and CI‑friendly execution. The pipeline ensures reversible transformations, byte‑level traceability, and secure key management.  
> 
> **Business Impact**  
> The framework enables enterprises to integrate secure encryption into ETL flows and data pipelines, reducing risk of data breaches and ensuring reproducibility in testing environments. Automated orchestration improves developer productivity, while layered cryptography strengthens resilience against attacks. The solution is scalable for deployment in cloud environments and adaptable to audit‑ready workflows.  
> 
> **Consulting Relevance**  
> This project demonstrates how cryptographic best practices can be embedded into advisory services for clients in finance, healthcare, and compliance‑heavy industries. It provides a replicable model for consulting engagements focused on secure data handling, workflow automation, and governance alignment.  
> 
> **Compliance / ESG / Risk Management**  
> Layered encryption supports compliance with GDPR, HIPAA, and financial data protection standards by ensuring confidentiality and integrity. ESG relevance is reflected in transparent, auditable processes that foster trust in digital ecosystems. From a risk management perspective, the pipeline mitigates operational risks by enabling retriable execution, secure key generation, and modular validation of encryption steps.


  #### Abstract:
  The purpose of a Dagster Encryption Pipeline is to test and showcase dagsters DAG-chaining capabilities,
resulting in a self-contained encryption-decryption pipeline which 1) connects different encryption methods
of  a plain text prior to 2) decrypting (and thus reversing the performed chain of encryption operations 
performed on an) encrypted message.
This project aims at designing a encryption-decryption pipeline via Dagsterr's website that would support users in 
**generating robustly encrypted messages** from a well-defined plain text input 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/Dagster_Encryption_Pipeline/DagsterEncryptionPipeline.md#7--references) 1 - 3 below). [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links)  

  [![Dagster_Cryptographic_Pipeline_Implementations](https://img.shields.io/badge/Dagster_Cryptographic_Pipeline_Implementations%20(Dagster_Cryptography)-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/f8f1273a20b5e5fdd495bd8ac23743687f0d92d9/Dagster_Encryption_Pipeline/DagsterEncryptionPipeline.md)

- ### 11. **FOREX Arbitrage Seeker GUI (ccxt) - Jul 2025**

> ## Executive Summary: FOREX Arbitrage Seeker GUI
> 
> **Business Problem**  
> Traders and analysts in cryptocurrency and foreign exchange markets need efficient tools to identify arbitrage opportunities across venues and trading pairs. Without real‑time visualization and simulation, market inefficiencies remain hidden, limiting profit potential and increasing exposure to risk.  
> 
> **AI/ML Solution**  
> This project delivers a modular PyQt5‑based GUI that integrates live exchange data via the ccxt library, simulates both spatial (two‑venue) and triangular (multi‑leg) arbitrage strategies, and visualizes opportunities through interactive mapping. Features include bid/ask spread tracking, fee‑aware profitability analysis, trade simulation dialogs, and CSV‑based logging. The GUI architecture is fully modular, enabling expansion to additional exchanges and trading routes.  
> 
> **Business Impact**  
> The tool empowers quants, engineers, and financial analysts to detect arbitrage opportunities in real time, improving profitability and reducing manual analysis overhead. It supports paper‑trading and strategy prototyping, lowering entry barriers for quantitative learners while enhancing decision‑support capabilities for professional traders.  
> 
> **Consulting Relevance**  
> This project demonstrates how financial engineering and data visualization can be embedded into advisory services. It provides a replicable framework for consulting engagements in trading strategy development, risk advisory, and financial technology innovation — directly relevant to clients in banking, fintech, and quantitative research.  
> 
> **Compliance / ESG / Risk Management**  
> By focusing on simulation rather than live trading, the tool supports compliance with regulatory requirements around market testing and risk disclosure. ESG relevance is reflected in its educational value, fostering responsible trading practices and transparency in financial markets. From a risk management perspective, the GUI helps organizations evaluate arbitrage strategies safely, mitigating exposure before committing capital.


  #### Abstract:
  The purpose of a FOREX Arbitrage Seeker GUI is to act as a modular, real-time analytical tool that enables users—especially those 
with quantitative or technical backgrounds—to identify, visualize, and simulate arbitrage opportunities across cryptocurrency markets.  

  ##### 🎯 **Primary Aim**

  To provide a **hands-on decision-support system** that helps users:
  - Detect profitable arbitrage windows (spatial or triangular)
  - Analyze real-time spreads between market pairs
  - Simulate trade outcomes based on live pricing
  - Visualize trade paths interactively
  - Log and export arbitrage events for analysis or backtesting 
  
  ##### ⚙️ **Functional Goals**
  
  - **Market Monitoring:** Continuously track live prices (bid/ask) from Binance and compute spreads on user-selected pairs.
  - **Simulation:** Allow users to test hypothetical trades and view potential returns before acting.
  - **Triangular Arbitrage:** Identify and evaluate profitable circular conversion paths such as USDT → BTC → ETH → USDT.
  - **Interactive Visualization:** Present triangle loops visually using `QGraphicsScene`, making abstract relationships tangible.
  - **Analytics & Logging:** Log each opportunity, generate spread charts, and support export for offline analysis.
  - **User Control:** Offer manual refresh, auto-refresh toggling, and selective export of arbitrage findings.
   
  ##### 🧠 **Why It Matters**
  
  In fast-moving crypto markets, arbitrage opportunities are fleeting. This tool arms users with:
  - Clarity: Real-time insight into pricing anomalies
  - Speed: Auto-refreshing mechanics and instant simulation
  - Intuition: Graphical triangle mapping for quick interpretation
  - Reproducibility: Logged data that can feed back into research or trading models  
   
  (see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/ForexArbitrageSeeker/ArbitrageSeeker_GUI.md#6--references) 1 - 3 below). [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links)

  [![FOREX_Arbitrage_Seeker_Implementations](https://img.shields.io/badge/FOREX_Arbitrage_Seeker_Implementations%20(Forex_ccxt)-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/e2957fd3b70a8e1da30a3909e106253d4efc429c/ForexArbitrageSeeker/ArbitrageSeeker_GUI.md)

- ### 12. **Score Card Evaluator GUI (Quality Management) - Jul 2025**

> ## Executive Summary: Scorecard Evaluator GUI
> 
> **Business Problem**  
> Production environments require continuous monitoring of process stability to ensure quality, reduce defects, and comply with industry standards. Traditional statistical process control (SPC) methods can be complex to interpret and are often inaccessible to non‑specialists, limiting their effectiveness on the shop floor.  
> 
> **AI/ML Solution**  
> This project delivers a PyQt5‑based GUI that integrates five parallel SPC charting methods: X̄–S charts, iterative filtering, defect count (D‑Chart), moving average, and EWMA. The interface supports CSV input, real‑time chart rendering, anomaly detection, and synthetic data generation for testing. Automated recalculation of control limits and iterative outlier rejection ensure robust statistical evaluation.  
> 
> **Business Impact**  
> The Scorecard Evaluator GUI transforms complex SPC theory into actionable insights, enabling engineers and quality control professionals to detect instability early, reduce rework, and improve production efficiency. Real‑time visualization accelerates decision‑making, while automated reporting enhances transparency and accountability in manufacturing processes.  
> 
> **Consulting Relevance**  
> This project demonstrates how statistical quality control can be operationalized through intuitive tooling. It provides a replicable framework for consulting engagements in Six Sigma, lean manufacturing, and industrial risk advisory — helping clients strengthen process reliability and embed continuous improvement practices.  
> 
> **Compliance / ESG / Risk Management**  
> By ensuring measurement accuracy and process stability, the tool supports compliance with ISO, Six Sigma, and industry‑specific quality standards. ESG relevance is reflected in reduced waste and resource efficiency through early defect detection. From a risk management perspective, the GUI mitigates operational risks by identifying process drifts and anomalies before they escalate into costly failures.


  #### Abstract:
  The Score Card Evaluator is a practical, user-friendly GUI application designed to empower professionals in quality management and process control. 
Built with PyQt5 and powered by robust statistical logic, it serves as a powerful tool for engineers, inspectors, and analysts working in daily production environments.
Its primary goal is to make statistical process control (SPC) accessible and intuitive — enabling users to visualize, evaluate, and troubleshoot
quality metrics using industry-standard techniques like X̄–S charts, defect count monitoring, moving averages, and EWMA.
By offering a streamlined interface, built-in logging, and flexible input handling, the Score Card Evaluator simplifies the complexity of SPC 
and delivers reliable insights that help maintain product consistency, detect shifts early, and support data-driven decision making on the shop floor or in laboratory settings 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/ScoreCardEvaluator_GUI/ScoreCardEvaluator_GUI.md#7--references) 1 - 3 below). [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links)
  
  [![ScorecardEvaluator_GUI](https://img.shields.io/badge/Scorecard_Evaluator_GUI%20(Quality_Management)-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/093fcf8ebe65a32e0fae9d6ee01382fc14857e4f/ScoreCardEvaluator_GUI/ScoreCardEvaluator_GUI.md)

- ### 13. **DRW Crypto Market Prediction (Kaggle competition) - Aug 2025**

> ## Executive Summary: DRW Crypto Forecasting Pipeline
> 
> **Business Problem**  
> Cryptocurrency markets are highly volatile, with short‑term price movements driven by complex, high‑dimensional signals. Traders and researchers require robust forecasting pipelines to transform noisy market data into actionable insights. Without advanced modeling, institutions risk poor execution, missed opportunities, and exposure to rapid market swings.  
> 
> **AI/ML Solution**  
> This project delivers a modular, high‑frequency forecasting pipeline that integrates feature engineering, deep learning, and ensemble learning strategies. Proprietary and public trading features are processed through lag/rolling engines, RSI and Bollinger Band modules, PCA dimensionality reduction, and Boruta feature selection. Models include gradient boosting (XGBoost, LightGBM), sequential deep learning (LSTM, CNN‑LSTM), and stacking ensembles, optimized via Bayesian search frameworks like Optuna. Rolling window experimentation ensures robustness across multiple time horizons, while Kaggle‑aligned submission formatting supports reproducibility.  
> 
> **Business Impact**  
> The pipeline enables institutional‑grade forecasting of short‑term crypto price movements, improving trading strategies, risk management, and portfolio performance. It reduces noise in high‑frequency data, enhances predictive accuracy, and provides scalable infrastructure for quantitative research and competitive modeling.  
> 
> **Consulting Relevance**  
> This project demonstrates how advanced forecasting architectures can be applied in financial consulting. It provides a replicable framework for advisory services in algorithmic trading, quantitative research, and fintech innovation — directly relevant to engagements with hedge funds, exchanges, and digital asset managers.  
> 
> **Compliance / ESG / Risk Management**  
> Transparent forecasting pipelines support compliance with financial reporting and risk disclosure requirements. ESG relevance is reflected in responsible trading practices, as improved forecasting reduces speculative volatility and promotes market stability. From a risk management perspective, the pipeline strengthens resilience against sudden market shocks by providing early directional signals and robust scenario testing.


  #### Abstract:
  The purpose of a DRW Crypto Market Prediction project is to develop a model capable of predicting crypto market price movements using synthetized realistic production data. Accurate directional signals derived through quantitative methods
  can significantly enhance trading strategies and enable more precise market opportunity identification. The cryptocurrency market represents one of the most dynamic and rapidly evolving financial landscapes, offering a wealth of
  opportunities for those who can extract meaningful insights from its vast streams of data. However, market information in crypto has an inherently low signal-to-noise ratio making it exceptionally difficult to identify predictive patterns.
  Price movements are shaped by a complex interplay of liquidity, order flow dynamics, sentiment shifts, and structural inefficiencies, requiring sophisticated quantitative techniques to decode.

  At DRW, we have been at the forefront of financial innovation for over three decades, embracing cutting-edge technology and rigorous quantitative research to optimize trading strategies. Through Cumberland, our dedicated crypto trading arm,
  we were among the earliest institutional participants in the digital asset space, helping to shape market structure and improve efficiency. As one of the largest liquidity providers in crypto, we thrive on developing proprietary trading
  strategies that adapt to the ever-changing market environment.

  In this competition, we invite you to build a model capable of predicting short-term crypto future price movements using our production feature data alongside publicly available market volume statistics. The proprietary production features
  we provide are integral to our trading strategies, capturing subtle market signals that help us navigate and seize opportunities in real time. Moreover, these production features, combined with public data describing the broader market state,
  create a rich and challenging dataset for data mining and modeling. Your task is to integrate these diverse sources of information into a single directional signal that effectively predicts crypto future price movements. Within this project
  however we will use instead of the original data set it synthesized realistic equivalent.

  Through this challenge, we aim to replicate the real-world problems we tackle at DRW every day—leveraging advanced machine learning techniques to extract structure from noisy, high-dimensional market data. The most successful solutions will
  provide a learning model that efficiently incorporates both explicit patterns and implicit interactions between all data features to refine price movement predictions. We look forward to seeing how the Kaggle community approaches this problem
  and how different modeling techniques can enhance our understanding of market dynamics. If you're excited by complex, high-impact challenges beyond predictive modeling, DRW offers a diverse range of opportunities at the intersection of quantitative
  research, technology, and trading strategy development. In the following the author will present his own prediction model and delve into its algorithmic aspects (see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/DRW_Crypto_Competition/Crypto_TimeseriesForecast.md#6--references) 1 - 3 below). [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links)

   [![DRW_Crypto_Analysis_Pipeline](https://img.shields.io/badge/DRW_Crypto_Forecasting%20(Kaggle_Competition)-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/c7acc73cbc371947aeeaf79ebfac5f193829075b/DRW_Crypto_Competition/Crypto_TimeseriesForecast.md)

- ### 14. **Analytics Engineer Exercise (Gaming) - Sep 2025**

> ## Executive Summary: Analytics Engineer Exercise (Gaming)
> 
> **Business Problem**  
> Mobile gaming companies collect millions of user events daily, but without structured analytics, product managers and marketers struggle to understand retention patterns, campaign efficiency, and user engagement. This leads to wasted acquisition spend and missed opportunities for growth.  
> 
> **AI/ML Solution**  
> This project simulates the role of an Analytics Engineer by transforming raw telemetry data into actionable insights. Using SQLite and Python, the pipeline integrates event parsing, cohort filtering, retention calculation, and campaign performance evaluation. Visualizations include retention curves, CPI trend charts, and dashboard‑ready PDF exports, providing a comprehensive view of user behavior and marketing efficiency.  
> 
> **Business Impact**  
> The framework enables stakeholders to track retention rates (Day 1, Day 3, Day 14), evaluate campaign cost efficiency, and identify engagement drop‑offs. These insights support data‑driven decision‑making, optimize marketing spend, and improve long‑term user retention strategies. Lightweight deployment ensures portability and rapid iteration for growth teams.  
> 
> **Consulting Relevance**  
> This project demonstrates how analytics engineering can be operationalized in mobile gaming. It provides a replicable framework for consulting engagements in product analytics, campaign attribution, and growth optimization — directly relevant to clients in gaming, app development, and digital marketing.  
> 
> **Compliance / ESG / Risk Management**  
> Transparent retention and CPI reporting supports compliance with advertising disclosure and fair marketing practices. ESG relevance is reflected in efficient resource allocation, reducing unnecessary acquisition costs and promoting sustainable growth strategies. From a risk management perspective, the tool mitigates financial risk by identifying underperforming campaigns early and ensuring reproducible analytics for stakeholder review.


  #### Abstract: 
  Imagine you are working as an Analytics Engineer for a company offering a mobile app. This app collects millions of data points from users every day. 
  Your goal is to transform and organize this data so that not only you can derive valuable insights from it, but also product managers and marketers 
  can easily query it.  
  
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
  
  6. Query the view marketing and report the Costs per Install (CPI) on August
  6, 2022, for campaign “google_campaign1”?
  
  ● Please submit your documented code along with instructions on how to
  run the code and the answers to the questions above.  
  
  ● Please only spend 120 minutes on the exercise. We know that it is
  challenging to complete all tasks but please respect the time limit.
  
  Attachment
  - exercise.db (see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/AnalyticsEngineerExercise/AnalyticsEngineeringExercise.md#6--references) 1 - 3 below). [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links)

   [![Analytics Engineer Exercise (Gaming)](https://img.shields.io/badge/Analytics_Engineer_Exercise%20(Gaming)-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/aa8bf069002f6022cfbd49332b042ab3f17064c4/AnalyticsEngineerExercise/AnalyticsEngineeringExercise.md)

- ### 15. **Meta Multi-Asset Analysis Project - Oct 2025**

> ## Executive Summary: Meta Multi Asset Management GUI
> 
> **Business Problem**  
> Multi‑asset management firms face challenges in adapting to dynamic portfolio structures, integrating diverse risk models, and ensuring reproducible analytics. Traditional tools often lack flexibility, limiting collaboration and slowing down exploratory research in fast‑moving financial environments.  
> 
> **AI/ML Solution**  
> This project delivers a schema‑aware PyQt5 GUI that dynamically adapts to changing asset structures. It integrates synthetic data generation using stochastic models (GBM, OU, Heston, regime‑switching), YAML‑based scenario loading, and automatic CSV persistence. Risk pipelines compute volatility, VaR, drawdown, and PCA‑based factor risks, while physics‑inspired models (entropy, Hurst exponent, Kalman filtering, Langevin dynamics) provide deeper insights into portfolio dynamics. A plugin registry supports external model injection, ensuring extensibility.  
> 
> **Business Impact**  
> The GUI empowers analysts and portfolio engineers to simulate, visualize, and diagnose multi‑asset portfolios in real time. It improves decision‑making by combining financial risk metrics with physics‑based diagnostics, supports reproducible workflows through autosave and schema validation, and enhances collaboration with modular, plug‑and‑play pipelines.  
> 
> **Consulting Relevance**  
> This project demonstrates how advanced portfolio analytics can be operationalized in advisory contexts. It provides a replicable framework for consulting engagements in asset management, quantitative research, and risk advisory — directly relevant to clients seeking innovative approaches to portfolio simulation, diagnostics, and scenario analysis.  
> 
> **Compliance / ESG / Risk Management**  
> Transparent schema validation and reproducible workflows support compliance with financial reporting and audit requirements. ESG relevance is reflected in sustainable portfolio practices, as physics‑based diagnostics help identify systemic risks and promote responsible asset allocation. From a risk management perspective, the GUI strengthens resilience by enabling scenario testing, volatility clustering analysis, and early detection of portfolio instabilities.


  #### Abstract: 
  Meta and functional programming paradigms offer significant flexibility in designing robust and flexible GUIs via PyQt5 capable of automatically adapting their widget structure to continuously changing data set schemas. Especially in the course of multi asset management tasks such flexibilities promise higher efficiencies in simultaneously evaluating numerous portfolio structures comprised of diverse asset classes. Therefore, multi asset management may be regarded as an adequate playground for testing highly adaptable GUI designs and their UX performance (see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/PortfolioRiskAnalysisProject/MetaMultiAssetAnalysis.md#7--references) 1 - 3 below).

  ##### 🎯 Primary Aim  

  Design a data analysis (pythonic) project regarding portfolio, risk and volatility analysis used within the framework of a multi-asset management company. Generate a large synthetic data set and design a pyqt gui which uses modern physical and statistical methods and accomodates its widget structure to feature changes within a csv data set by means of functional and meta programming paradigms. [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links)

  [![Meta_Multi_Asset_Analysis](https://img.shields.io/badge/Meta_Multi_Asset%20(Portfolio_Analysis)-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/609e08a2523fe1024856136bbefa88b38624b112/PortfolioRiskAnalysisProject/MetaMultiAssetAnalysis.md)

- ### 16. **Quantum-Kalman Time Series Forecasting - Oct 2025**

> ## Executive Summary: Quantum-Kalman Forecasting of Time Series
> 
> **Business Problem**  
> Organizations and researchers require accurate time series forecasting to manage financial risk, optimize operations, and anticipate market dynamics. Traditional models often struggle with noisy data and lack intuitive interfaces, limiting accessibility for non‑specialists and slowing adoption in applied environments.  
> 
> **AI/ML Solution**  
> This project integrates a hybrid forecasting pipeline that combines ARIMA, Kalman filtering, Richardson‑extrapolation, asymptotic analysis, and quantum‑inspired denoising. A LangChain agent powered by a local LLM interprets natural language prompts, orchestrates forecasting tools, and delivers structured responses. Streamlit provides a conversational GUI for real‑time visualization, while automated result storage ensures reproducibility. The modular architecture supports synthetic asset generation, multi‑model forecasting, and extensibility for future tools.  
> 
> **Business Impact**  
> The solution enables intuitive, voice‑accessible forecasting that bridges advanced modeling with user‑friendly interaction. It improves predictive accuracy, reduces noise sensitivity, and accelerates research workflows. By making complex forecasting logic accessible through natural language, the system empowers analysts, educators, and developers to prototype and deploy forecasting solutions more efficiently.  
> 
> **Consulting Relevance**  
> This project demonstrates how hybrid modeling and conversational AI can be operationalized in advisory contexts. It provides a replicable framework for consulting engagements in financial forecasting, operational planning, and AI‑driven decision support — directly relevant to clients seeking explainable, modular, and reproducible forecasting systems.  
> 
> **Compliance / ESG / Risk Management**  
> Transparent workflows and reproducible outputs support compliance with audit and reporting standards in finance and operations. ESG relevance is reflected in accessible forecasting tools that democratize advanced analytics, fostering responsible adoption of AI. From a risk management perspective, the pipeline strengthens resilience against noisy or incomplete data by combining classical and quantum‑inspired denoising techniques.


  #### Abstract: 
  This project explores a novel hybrid framework for time series forecasting by integrating three powerful methodologies: classical Kalman filtering, 
Sidis-style mathematical extrapolation, and quantum-inspired noise mitigation. The central aim is to enhance predictive robustness and reduce 
volatility bands in complex temporal data. By leveraging Kalman filters for state estimation, applying Richardson extrapolation and anti-limit 
techniques to extend trends, and incorporating quantum error correction strategies to suppress noise, we seek to construct a unified pipeline capable 
of resilient and precise forecasting. This interdisciplinary synthesis promises new insights into volatility modeling, especially in chaotic or 
regime-switching systems, and opens the door to quantum-enhanced predictive analytics. The Pythonic noise-mitigation forecasting pipeline emerging from inquiries 
of this project will be packaged into a user-friendly streamlit app capable of interacting with user's prompts 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/KalmanSignalForecasting/QuantumKalmanSignalForecasting.md#7--references) 1 - 3 below). [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links) 

  [![Quantum_Kalman_Forecasting](https://img.shields.io/badge/Quantum_Kalman_Forecasting%20(TimeSeries_Analysis)-English-yellowblue?logoColor=blue&labelColor=yellow)]( https://github.com/NenadBalaneskovic/ExternalProjects/blob/2595ff1848b382dec0be605c3c2f6c87f97713ab/KalmanSignalForecasting/QuantumKalmanSignalForecasting.md)

- ### 17. **Linear Optimization Solver GUI - Nov 2025**

> ## Executive Summary: Linear Programming Optimization GUI
> 
> **Business Problem**  
> Organizations across manufacturing, logistics, and finance rely on linear programming to optimize resource allocation, scheduling, and cost efficiency. Traditional solver environments are often opaque, difficult to use, and lack interactive visualization, limiting accessibility for decision‑makers and students alike.  
> 
> **AI/ML Solution**  
> This project delivers a PyQt5‑based GUI that integrates multiple optimization methods — including Simplex, Dual Simplex, Dikin interior‑point, parametric LP, and integer‑constrained solvers. A symbolic parser (SymPy) converts algebraic input into structured matrices, while an auto‑detection module selects the most appropriate solving strategy. Real‑time visualization renders feasible regions and optimization paths, with solver traces and diagnostic exports ensuring transparency and reproducibility.  
> 
> **Business Impact**  
> The GUI empowers analysts, engineers, and educators to interact with complex optimization logic through an intuitive interface. It accelerates problem‑solving, improves accuracy in resource planning, and enhances decision‑support workflows. By combining solver orchestration with dynamic visualization, the tool reduces barriers to adoption and strengthens operational efficiency.  
> 
> **Consulting Relevance**  
> This project demonstrates how optimization theory can be operationalized into practical, user‑friendly tooling. It provides a replicable framework for consulting engagements in operations research, supply chain optimization, and financial modeling — directly relevant to clients seeking scalable, explainable, and interactive optimization solutions.  
> 
> **Compliance / ESG / Risk Management**  
> Transparent solver logic and reproducible workflows support compliance with audit and reporting standards in regulated industries. ESG relevance is reflected in efficient resource allocation, reducing waste and promoting sustainable operations. From a risk management perspective, the GUI strengthens resilience by enabling scenario testing, constraint sensitivity analysis, and early detection of infeasible solutions.


  #### Abstract:
  This project presents a unified computational framework for solving linear optimization problems through a multi-method graphical interface. By integrating classical simplex algorithms, dual formulations, parametric linear programming, integer-constrained solvers, and interior-point techniques such as the Dikin method, the system enables robust exploration of feasible regions and optimality paths across diverse problem structures. The central aim is to democratize access to advanced linear programming strategies while offering real-time visualization and interpretability. Leveraging simplex-based pivoting for vertex traversal, parametric solvers for dynamic constraint sensitivity, and integer programming for discrete decision modeling, the framework accommodates both continuous and combinatorial optimization scenarios. The inclusion of the Dikin method introduces a smooth, interior-point trajectory that complements boundary-based approaches, offering insights into curvature and convergence behavior within feasible polyhedra. This methodological synthesis is encapsulated in a Python-powered GUI that dynamically adapts to user-defined variable types, constraint structures, and solver preferences. The interface supports interactive input parsing, automatic method detection, and 2D geometric rendering of feasible regions, constraint boundaries, and optimization paths. Designed for both educational and applied contexts, the system fosters intuitive understanding of linear optimization mechanics while maintaining algorithmic rigor. The resulting application serves as a modular launchpad for future extensions into nonlinear programming, multi-objective optimization, and hybrid solver orchestration. By bridging algorithmic depth with visual clarity, this project opens new avenues for accessible, interpretable, and customizable optimization workflows (see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/LinearProgramming_GUI/LinearProgramming_GUI.md#8--references) 1 - 3 below). [[<<]](https://github.com/NenadBalaneskovic/ExternalProjects#-dataset-analysis-links)

  [![Linear_Optimization_Solver_GUI](https://img.shields.io/badge/Linear_Optimization_GUI%20(Operations_Research)-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/59d32634dafa9c9219cd162b1b7b8ffadd143238/LinearProgramming_GUI/LinearProgramming_GUI.md)

  
