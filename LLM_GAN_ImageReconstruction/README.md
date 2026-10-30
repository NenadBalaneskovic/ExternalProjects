# GAN vs OpenCV Chessboard Reconstruction

## 📌 Project Overview  

This project aims to compare traditional **OpenCV-based** methods for chessboard image reconstruction with
**Generative Adversarial Network (GAN)-driven** approaches. The goal is to evaluate the effectiveness of deep learning 
in reconstructing **secluded or obscured chessboard sections** more accurately than conventional techniques 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/LLM_GAN_ImageReconstruction/README.md#-references) 1 - 3 below).

## 🚀 Features  

- **OpenCV Reconstruction**: Uses contour detection and perspective transformation to restore missing parts.
- **GAN-Based Reconstruction**: Trained with a dataset of **chessboard images** to generate missing squares using adversarial learning.

## 📊 Methodology  

1. **Preprocessing**: Chessboard images are converted to grayscale and normalized.
2. **OpenCV Approach**:
   - Detects edges using `cv2.Canny()`.
   - Identifies missing squares through contour detection.
   - Attempts reconstruction using transformations.
3. **GAN Approach**:
   - Uses a **trained generator** to recreate missing squares.
   - Fine-tunes output through **loss function optimization**.
   - Adjusts resolution dynamically with `cv2.resize()`.

## 🔬 Comparison Goals  

- **Accuracy of square reconstruction** in obstructed images.
- **Performance on tilted chessboards** vs. frontal views (future implementation).
- **Processing speed and computational efficiency**.

## 📂 Repository Structure  

```plaintext
- dataset/
   - secluded/  # Chessboards with missing sections
   - complete/  # Fully intact chessboards
- models/
   - gan_chessboard_model.h5  # Trained GAN model
- scripts/
   - train_gan.py  # GAN training function
   - reconstruct_chessboard.py  # Reconstruction script
```
   
## 📷 Visualizations  

### 1. Standard Edge Detection via OpenCV
![Standard Edge Detection via OpenCV](https://github.com/NenadBalaneskovic/ExternalProjects/blob/9658c9a261c518adf596dc7539602e029b391bc3/LLM_GAN_ImageReconstruction/ReconstructedStandardResults.PNG)  
### 2. GAN-induced Reconstruction of Chessboard Images
![GAN-induced Reconstruction of Chessboard Images](https://github.com/NenadBalaneskovic/ExternalProjects/blob/372401f6476fbaf9224bcf0964badcb5220d197e/LLM_GAN_ImageReconstruction/SecludedReconstructed.png)  
### 3. Google Colab terminal - GAN model training session
![Google Colab terminal](https://github.com/NenadBalaneskovic/ExternalProjects/blob/1dae994977816bd5ae1d81dc7112939a555540a1/LLM_GAN_ImageReconstruction/GoogleColab_Terminal.PNG)  

## 🚀 Results  

✔ GAN trainings have been performed via Google Colabs free access to a T4 GPU for  

1. 5000 epochs, batch size of 64 and 200 chessboard images (duration 2h),  
2. 10000 epochs, batch size of 32 and 200 chessboard images (duration 5 - 6 hours).  

✔ After 10000 epochs for batch sizes 32 the GAN model was capable of draving an (8x8) rectangular boundary around the 
originally secluded (8x7) portion of the chessboard and reconstructing its rectangles
(the last 8th row is the missing corrected chessboard portion).  
   
## 🛠 How to Run (in Google Colab)

### 1️⃣ Open Google Colab  

1️⃣ Go to Google Colab  
2️⃣ Click "New Notebook" to create a fresh environment  

### 2️⃣ Enable GPU for Faster Training  

1️⃣ Click "Runtime" → "Change runtime type"  
2️⃣ Select "GPU" from the dropdown  
3️⃣ Click "Save"

✔ This ensures your GAN training runs on GPU, making it much faster.  

### 3️⃣ Upload Your Training Script  

1️⃣ Inside Google Colab, go to "Files" (left sidebar)  
2️⃣ Click "Upload" and add:

-- the GAN training script (train_gan.py)

-- the chessboard dataset (dataset/secluded/ & dataset/complete/)  

### 4️⃣ Install Dependencies

```bash
!pip install tensorflow keras numpy matplotlib opencv-python
```  

### 2️⃣ Run Reconstruction  

1️⃣. Run a GAN Training (specify number of stochastic gradient descent epochs and batch size)  

2️⃣  Load our script and start training:  
```python
!python train_gan.py
```

### 3️⃣  Download the h5 model file ("gan_chessboard_model.h5"):
1️⃣. Once training is complete, save the trained GAN model:  
```python
from tensorflow.keras.models import save_model

generator.save("/content/gan_chessboard_model.h5")  # Save model in Colab
```  

2️⃣  Then download it manually by clicking the file in the Colab sidebar. 

## 📈 Future Improvements  
✔ **Future improvements may involve the following aspects**:

1. Enhanced GAN training with more diverse chessboard perspectives.

2. Incorporation of object detection models for robust missing square identification.

3. Optimized processing speed using GPU acceleration.

4. Tilted & Obscured Image Handling: Perspective correction improves the accuracy of chessboard restoration from angled views.

## 📚 References
1. **Generative Adversarial Networks** – https://realpython.com/generative-adversarial-networks/, https://www.tensorflow.org/tutorials/generative/dcgan 
2. [![Jupyter Notebook | English](https://img.shields.io/badge/Jupyter%20Notebook-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/ea3c345a0d55882a5cad502f77d41a6e7e50402f/LLM_GAN_ImageReconstruction/LLM_OpenCV_ImageReconstruction.ipynb)
3. [![GAN Report | English](https://img.shields.io/badge/LLM_GAN%20Report-English-yellowblue?logoColor=blue&labelColor=red)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/12b49cfdd5b5d457264786c899ccbc6db839d094/LLM_GAN_ImageReconstruction/LLM_GAN_ImageReconstruction.pdf) 
4. **pygan Documentation** – https://pypi.org/project/pygan/
5. **GAN with pytorch** – https://medium.com/the-research-nest/how-to-program-a-simple-gan-559ad707e201
6. **GAN in Python** - https://www.codemotion.com/magazine/ai-ml/deep-learning/how-to-build-a-gan-in-python/
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
---
