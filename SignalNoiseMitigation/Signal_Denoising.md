# A 🚀 Adaptive Denoising Framework

## 🎯 1. Objective  
Noise contamination is a fundamental challenge in **signal processing**, affecting the accuracy and reliability of measurements across various domains, 
from biomedical signals to communications and industrial data analysis. This project aims to **experiment with different noise mitigation techniques**, 
particularly **non-deep-learning approaches**, and optimize them to identify the **most flexible noise suppression strategy** 
(see [References](https://github.com/NenadBalaneskovic/ExternalProjects/blob/main/SignalNoiseMitigation/Signal_Denoising.md#-references) 1 - 3 below).

## 2. 🔎 Motivation  
While deep learning-based denoising has gained popularity, traditional and hybrid techniques remain essential in scenarios where:
✔ **Computational efficiency** is required.  
✔ **Real-time processing** is needed without extensive training datasets.  
✔ **Robust adaptability** across various noise conditions is beneficial.  

By **evaluating and refining classical noise suppression methods**, this project will establish an **ensemble-based framework** capable of 
selecting the **most appropriate denoising technique** based on real-time signal characteristics.

## 3. ⚙️ Research Approach  

### 🧪 3.1 Comparative Methodology  
The study will evaluate an **ensemble of noise reduction techniques**, including:
✔ **Median-filtered variance estimation** – Adaptive noise variance assessment.  
✔ **Correlation-based noise mitigation** – Signal coherence preservation.  
✔ **Beta-sigma adaptive resampling** – Fine-tuned smoothing strategies.  
✔ **Multi-stage hybrid filtering** – Dynamic fusion of different denoising approaches.  

Each method will be **benchmarked** against various signal types, with performance assessed using metrics like **Root Mean Square Error (RMSE)** and **signal preservation fidelity**.

### ⚡ 3.2 Optimization Strategy  
The project will refine each technique by:  

✅ **Enhancing dynamic kernel selection** to improve adaptability.  
✅ **Optimizing hybrid fusion approaches** for balancing noise suppression and signal integrity.  
✅ **Developing real-time variance estimation techniques** to improve robustness in non-stationary environments.  

## 🎯 4. Expected Outcome  
By systematically testing and optimizing these **noise mitigation strategies**, the project will:
✔ Identify the **most flexible denoising approach** from an ensemble of functionality choices.  
✔ Develop an **adaptive noise suppression framework** suitable for real-time applications.  
✔ Contribute to signal processing methodologies independent of deep-learning constraints.  

## 🏁 5. Conclusion  
This project provides a **structured exploration of non-deep-learning denoising techniques**, leading to an **adaptive and optimized noise reduction framework**. 
The goal is to enhance **real-time signal processing capabilities** while maintaining computational efficiency and accuracy.

---

# B 📖 Theoretical Introduction to Advanced Signal Denoising

## 🏗️ 1. Introduction  
Noise contamination is a critical issue in **signal processing**, leading to distortions and reduced accuracy in measured data. The need for
**adaptive and hybrid noise reduction methods** has driven research into robust techniques. This project explores various **multi-modal denoising approaches**,
integrating **variance estimation, correlation-based filtering, and adaptive resampling** for enhanced noise suppression.

## 🔬 2. Noise Variance Estimation  

### 🔹 2.1 Median-Filtered Variance Estimation  
Noise variance estimation is essential for effective denoising. One method involves applying a **median filter** to smooth signal variations while estimating 
noise levels dynamically. The variance \( \sigma^2 \) is computed as:



$$
\[
\sigma^2 = \text{median}(|x_i - \text{median}(x)|)
\]
$$



where $$\( x_i \)$$ represents local signal samples.

**Distribution-Specific Adjustments**:

| **Noise Type**  | **Variance Adjustment Formula** |
|-----------------|--------------------------------|
| **Gaussian Noise**  | $$\( \sigma^2_{\text{adj}} = 0.8 \cdot \sigma^2 \)$$ |
| **Uniform Noise**   | $$\( \sigma^2_{\text{adj}} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2 \)$$ |
| **Laplace Noise**   | $$\( \sigma^2_{\text{adj}} = 2 \cdot \sigma^2 \)$$ |

where $$\( \mu \)$$ is the signal mean and $$\( N \)$$ is the number of samples.

### 🔹 2.2 Correlation-Based Noise Variance Estimation  
Another approach uses **autocorrelation properties** to estimate noise variance dynamically:


$$
\[
\sigma^2_{\text{corr}} = \frac{1}{M} \sum_{k=1}^{M} \text{ACF}(k)
\]
$$



where $$\( \text{ACF}(k) \)$$ represents the autocorrelation function at lag $$\( k \)$$, and $$\( M \)$$ is the number of lags used.

## ⚙️ 3. Adaptive Noise Variance Estimation  

### 🔄 3.1 Dynamic Noise Variance Estimation  
To enhance robustness in **non-stationary noise environments**, a **real-time complexity analysis** adapts variance estimation:


$$
\[
\sigma^2_{\text{dyn}} = \alpha \cdot \sigma^2_{\text{median}} + (1-\alpha) \cdot \sigma^2_{\text{corr}}
\]
$$



where $$\( \alpha \)$$ is an adaptive weighting factor dependent on **local signal complexity**.

### 🎭 3.2 Hybrid Variance Estimation Framework  
A hybrid approach combining **median-filtered** and **correlation-based variance estimation** offers greater adaptability:


$$
\[
\sigma^2_{\text{hybrid}} = \frac{1}{2} \left( \sigma^2_{\text{median}} + \sigma^2_{\text{corr}} \right)
\]
$$



This framework optimally adjusts for diverse **multi-modal noise distributions**.

## 🚀 4. Advanced Hybrid Signal Denoising  

### 📊 4.1 Multi-Stage Adaptive Filtering  
By integrating multiple filtering stages, signal integrity is preserved while enhancing noise suppression. A **dynamic kernel adjustment** is employed:


$$
\[
K_{\text{opt}} = K_{\text{min}} + (K_{\text{max}} - K_{\text{min}}) \left( 1 - \frac{\sigma^2_{\text{local}}}{\sigma^2_{\text{max}}} \right)
\]
$$



where $$\( K_{\text{opt}} \)$$ is the optimal kernel size, dynamically tuned based on local **noise variance $$\( \sigma^2_{\text{local}} \)$$**.

### 🔗 4.2 Hybrid Statistical Noise Reduction  
Combining **correlation-based denoising** and **adaptive variance estimation**, this technique enhances **adaptability**:


$$
\[
x_{\text{filtered}} = \beta \cdot x_{\text{median}} + (1-\beta) \cdot x_{\text{corr}}
\]
$$



where $$\( \beta \)$$ adapts to signal **complexity levels**.

## 🔄 5. Real-Time Adaptive Noise Suppression  

### ⚙️ 5.1 Dynamic Fusion of Denoising Techniques  
An **adaptive weighting mechanism** selects optimal noise suppression strategies dynamically:


$$
\[
x_{\text{fusion}} = \sum_{i=1}^{N} \omega_i \cdot x_{\text{denoised}, i}
\]
$$



where $$\( \omega_i \)$$ is the fusion weight and $$\( x_{\text{denoised}, i} \)$$ represents denoised signals from various methods.

### 🔀 5.2 Multi-Stage Hybrid Filtering  
This method integrates **variance estimation, correlation-based denoising, and adaptive resampling** into a **multi-step framework**:


$$
\[
x_{\text{final}} = \lambda_1 \cdot x_{\text{median}} + \lambda_2 \cdot x_{\text{corr}} + \lambda_3 \cdot x_{\text{adaptive}}
\]
$$



where $$\( \lambda_i \)$$ are **adaptive fusion parameters**.

### 🎯 5.3 Optimized Signal Reconstruction  
Final reconstruction includes **error correction mechanisms**:


$$
\[
x_{\text{corrected}} = x_{\text{final}} - \gamma \cdot \text{Error}(x_{\text{final}})
\]
$$



where $$\( \gamma \)$$ scales the correction factor based on residual noise levels.

## 🏁 6. Conclusion  
This **adaptive signal denoising framework** integrates **variance estimation, multi-stage filtering, and fusion-based noise suppression** 
to achieve **robust and efficient noise reduction**. The incorporation of **real-time adaptability** ensures that signals retain essential properties while effectively mitigating noise distortions.

---
# C Implementation Details of the Denoising Framework  

## 1. Introduction  
This document describes the implementation details of the **ensemble-based denoising framework**, outlining the individual functions, their integration, and the rationale behind their design.

The project employs **multiple noise mitigation techniques**, with an emphasis on **adaptive and hybrid approaches** that do not rely on deep learning.
The goal is to refine and optimize **non-deep-learning denoising strategies**, selecting the most **flexible and effective noise suppression method** based on real-time signal properties.

## 2. Functions and Their Implementation  

### **2.1 Hybrid Multi-Pass Adaptive Median Filtering**  

#### **Purpose:**  
Designed to **smooth signals while preserving edges**, this technique applies an **adaptive median filter** over multiple passes.

#### **Implementation Overview:**  
- Computes **local complexity** by evaluating **short-term variations** around each sample.
- Dynamically selects a **median filter kernel size** based on local signal behavior.
- Applies **multi-pass processing** to iteratively refine noise suppression.
- Implements **hybrid smoothing**, where the new signal is blended with the previous signal using an **adjustable weighting factor** \( \alpha \).

#### **Mathematical Formulation:**  
Given a **signal $$\( x \)$$** of length $$\( N \)$$, the local complexity at position $$\( i \)$$ is defined as:


$$
\[
C_i = \left| x_i - \frac{1}{L} \sum_{k=i-5}^{i+5} x_k \right|
\]
$$



where $$\( L \)$$ is the local window size.

The kernel size is then dynamically adjusted:


$$
\[
K_i = K_{\min} + (K_{\max} - K_{\min}) \left( 1 - \frac{C_i}{\max(C)} \right)
\]
$$



Ensuring **odd-sized kernels** prevents artifacts in filtering:


$$
\[
K_i = K_i + 1 \quad \text{(if \( K_i \) is even)}
\]
$$



Each pass refines the signal $$\( x \)$$, blending the filtered version:


$$
\[
x_{\text{filtered}} = \alpha x_{\text{median}} + (1 - \alpha) x_{\text{original}}
\]
$$



---

### **2.2 Enhanced Correlation-Based Noise Reduction**  

#### **Purpose:**  
Leverages **autocorrelation properties** to estimate and mitigate noise dynamically.

#### **Implementation Overview:**  
- Computes the **autocorrelation function (ACF)** to analyze periodic dependencies in the signal.
- Estimates noise levels by averaging the **second half of the ACF spectrum**.
- Applies an **adaptive correction factor**, ensuring **optimal noise reduction** without distorting the primary waveform.

#### **Mathematical Formulation:**  


$$
\[
\text{ACF}(k) = \sum_{i=1}^{N-k} x_i \cdot x_{i+k}
\]
$$



Noise estimation:


$$
\[
\sigma_{\text{noise}} = \frac{1}{M} \sum_{k=M/2}^{M} \text{ACF}(k)
\]
$$



Corrected signal:


$$
\[
x_{\text{corrected}} = x - (\sigma_{\text{noise}} \cdot \text{Correction Factor})
\]
$$



---

### **2.3 Adaptive βσ-Resampling**  

#### **Purpose:**  
This method applies **βσ-scaling** to mitigate noise variations and resample signals dynamically.

#### **Implementation Overview:**  
- Computes **local noise levels** using **standard deviation-based analysis**.
- Applies **adaptive β-scaling** to suppress noise dynamically.
- Fine-tunes resampling parameters to ensure a balance between **noise removal** and **signal preservation**.

#### **Mathematical Formulation:**  

Noise level estimation:


$$
\[
\sigma_{\text{local}, i} = \sqrt{\frac{1}{L} \sum_{k=i-10}^{i+10} (x_k - \mu)^2}
\]
$$



Adaptive β-scaling:


$$
\[
\beta_i = \beta_0 \left( 1 - \exp\left(- \frac{\sigma_{\text{local}, i}}{\max(\sigma_{\text{local}})} \right) \right) + \text{Offset}
\]
$$



Resampled signal:


$$
\[
x_{\text{resampled}, i} = (1 - \beta_i) x_i + \beta_i \cdot \text{Local Mean}
\]
$$



---

### **2.4 Hybrid Correlated Beta-Sigma Denoiser**  

#### **Purpose:**  
Combines **correlation-based denoising** and **βσ-resampling** to achieve an **adaptive hybrid denoiser**.

#### **Implementation Overview:**  
- Applies **correlation-based filtering** for periodic noise removal.
- Implements **adaptive βσ-resampling** to smooth residual noise.
- Integrates both methods into a **single hybrid strategy**, dynamically blending them.

#### **Mathematical Formulation:**  


$$
\[
x_{\text{hybrid}} = \alpha \cdot x_{\text{corr}} + (1-\alpha) \cdot x_{\text{resampled}}
\]
$$



where $$\( \alpha \)$$ controls the weighting.

---

### **2.5 Flexible Dynamic Denoiser**  

#### **Purpose:**  
Designed to **adaptively select** the best noise mitigation strategy **based on real-time signal characteristics**.

#### **Implementation Overview:**  
- Evaluates **local signal complexity and noise intensity**.
- Dynamically assigns **fusion weights** for each technique.
- Integrates all previously defined methods into a **flexible, ensemble-based framework**.

#### **Mathematical Formulation:**  

Local noise estimation:


$$
\[
\sigma_{\text{noise}} = \frac{1}{N} \sum_{i=1}^{N} \left( x_i - \mu \right)^2
\]
$$



Fusion weighting:


$$
\[
\alpha_{\text{fusion}} = \min(1.0, \max(0.3, (1 - \sigma_{\text{noise}}) \cdot (1 - \text{Complexity Factor})))
\]
$$



Final adaptive signal:


$$
\[
x_{\text{flexible}} = \frac{1}{2} \left( \alpha_{\text{fusion}} \cdot x_{\text{median}} + (1-\alpha_{\text{fusion}}) \cdot x_{\text{corr}} + 0.5 x_{\text{resampled}} + 0.5 x_{\text{hybrid}} \right)
\]
$$



---

## 3. Integration Strategy  

The **Flexible Dynamic Denoiser** serves as the **central selection mechanism**, dynamically blending **multiple denoising techniques** based on **signal complexity and noise intensity**.

### **Overall Workflow:**
1️⃣ **Signal Preprocessing:** Estimate noise variance using **Median-Filtered** or **Correlation-Based techniques**.  
2️⃣ **Denoising Application:** Apply **Hybrid Adaptive Filtering**, **Beta-Sigma Resampling**, or **Correlation-Based Noise Reduction**.  
3️⃣ **Dynamic Optimization:** The **Flexible Denoiser** selects and blends techniques in **real-time** to optimize noise mitigation.  
4️⃣ **Visualization & Evaluation:** Compute RMSE and compare methods for **benchmarking performance**.  

### **Pythonic workflow implementation**  

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from statsmodels.tsa.stattools import acf

# 📌 1️⃣ Hybrid Multi-Pass Adaptive Median Filtering
def hybrid_multi_pass_adaptive_median_filter(signal_data, min_kernel=3, max_kernel=9, passes=3, alpha=0.5):
    """Apply multi-pass adaptive median filtering with kernel tuning based on signal complexity and hybrid smoothing."""
    N = len(signal_data)
    filtered_signal = np.copy(signal_data)

    for _ in range(passes):
        temp_signal = np.zeros_like(filtered_signal)

        for i in range(N):
            local_complexity = np.abs(filtered_signal[i] - np.mean(filtered_signal[max(0, i-5):min(N, i+5)]))
            kernel_size = int(min_kernel + (max_kernel - min_kernel) * (1 - local_complexity / np.max(filtered_signal)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            start_idx = max(0, i - kernel_size // 2)
            end_idx = min(N, i + kernel_size // 2 + 1)
            temp_signal[i] = np.median(filtered_signal[start_idx:end_idx])

        filtered_signal = alpha * temp_signal + (1 - alpha) * filtered_signal

    return filtered_signal

# 📌 2️⃣ Enhanced Correlation-Based Noise Reduction
def correlation_based_denoising(signal_data, smoothing_factor=0.8):
    """Estimate noise via autocorrelation with adaptive correction."""
    auto_corr = acf(signal_data, fft=True)
    noise_est = np.mean(auto_corr[len(auto_corr)//2:] * (1 - smoothing_factor))  
    corrected_signal = signal_data - (noise_est * smoothing_factor)

    return corrected_signal

# 📌 3️⃣ Adaptive βσ-Resampling
def beta_sigma_resampling(signal_data, base_beta=0.1, sigma_factor=0.5, offset_factor=0.01, noise_scale=0.02):
    """Apply βσ-resampling with adaptive noise mitigation and fine-tuned smoothing."""
    N = len(signal_data)
    noise_levels = np.array([
        np.std(signal_data[max(0, i-10):min(N, i+10)]) if len(signal_data[max(0, i-10):min(N, i+10)]) > 1 else offset_factor
        for i in range(N)
    ])
    max_noise = np.max(noise_levels) if np.max(noise_levels) > 0 else 1.0
    adaptive_beta = base_beta * (1 - np.exp(-noise_levels / max_noise)) + offset_factor
    resampled_signal = np.zeros_like(signal_data)

    for i in range(N):
        sigma_dynamic = int(sigma_factor * (1 + noise_levels[i]))
        lower_bound = max(0, i - sigma_dynamic)
        upper_bound = min(N - 1, i + sigma_dynamic)
        subset = signal_data[lower_bound:upper_bound]
        weighted_mean = np.mean(subset) if len(subset) > 1 else signal_data[i]
        resampled_signal[i] = weighted_mean * (1 - adaptive_beta[i]) + signal_data[i] * adaptive_beta[i]
        resampled_signal[i] += np.random.normal(scale=noise_scale)

    return resampled_signal

# 📌 4️⃣ Hybrid Correlated Beta-Sigma Denoiser
def correlated_beta_sigma_denoiser(signal_data, alpha=0.6):
    """Hybrid method combining correlation-based denoising and adaptive βσ-resampling."""
    correlation_denoised = correlation_based_denoising(signal_data)
    beta_sigma_denoised = beta_sigma_resampling(signal_data)
    hybrid_signal = alpha * correlation_denoised + (1 - alpha) * beta_sigma_denoised

    return hybrid_signal

# 📌 5️⃣ Flexible Denoiser (Real-Time Adaptive Mode Selection)
def flexible_denoiser(signal_data):
    """Selects and adjusts denoising techniques dynamically based on real-time signal properties."""
    N = len(signal_data)
    
    # ✅ Compute noise levels for real-time fusion strategy
    local_noise = np.array([
        np.std(signal_data[max(0, i-10):min(N, i+10)]) if len(signal_data[max(0, i-10):min(N, i+10)]) > 1 else 0.01
        for i in range(N)
    ])
    
    # ✅ Adjust fusion weight dynamically based on local signal complexity and noise intensity
    noise_intensity = np.mean(local_noise)
    complexity_factor = np.mean(np.abs(np.diff(signal_data)))  # Evaluate signal fluctuations
    fusion_alpha = min(1.0, max(0.3, (1 - noise_intensity) * (1 - complexity_factor)))  

    # ✅ Apply methods dynamically
    median_filtered = hybrid_multi_pass_adaptive_median_filter(signal_data)
    correlation_filtered = correlation_based_denoising(signal_data)
    beta_sigma_filtered = beta_sigma_resampling(signal_data)
    hybrid_filtered = correlated_beta_sigma_denoiser(signal_data)

    # ✅ Combine methods dynamically based on real-time signal fluctuations
    flexible_signal = (
        fusion_alpha * median_filtered +
        (1 - fusion_alpha) * correlation_filtered +
        0.5 * beta_sigma_filtered +
        0.5 * hybrid_filtered
    ) / 2.0

    return flexible_signal

# 📌 Main Function for Denoising & Visualization
def compare_denoising_methods(signal_data, t):
    """Compare all denoising methods and visualize results."""
    hybrid_median_filtered = hybrid_multi_pass_adaptive_median_filter(signal_data)
    correlation_denoised = correlation_based_denoising(signal_data)
    beta_sigma_denoised = beta_sigma_resampling(signal_data)
    hybrid_denoised = correlated_beta_sigma_denoiser(signal_data)
    flexible_filtered = flexible_denoiser(signal_data)

    def compute_rmse(original, denoised):
        return np.sqrt(np.mean((original - denoised) ** 2))

    stats = {
        "Hybrid Multi-Pass Median Filtering RMSE": compute_rmse(signal_data, hybrid_median_filtered),
        "Enhanced Correlation-Based RMSE": compute_rmse(signal_data, correlation_denoised),
        "Adaptive Beta-Sigma Resampling RMSE": compute_rmse(signal_data, beta_sigma_denoised),
        "Hybrid Correlated Beta-Sigma Denoiser RMSE": compute_rmse(signal_data, hybrid_denoised),
        "Flexible Dynamic Denoiser RMSE": compute_rmse(signal_data, flexible_filtered)
    }

    plt.figure(figsize=(12, 6))
    plt.plot(t, signal_data, label="Noisy Signal", color="black", alpha=0.6)
    plt.plot(t, hybrid_median_filtered, label="Hybrid Multi-Pass Median Filtered", linestyle="dashed")
    plt.plot(t, correlation_denoised, label="Enhanced Correlation-Based", linestyle="dashed")
    plt.plot(t, beta_sigma_denoised, label="Adaptive Beta-Sigma Resampling", linestyle="dashed")
    plt.plot(t, hybrid_denoised, label="Hybrid Correlated Beta-Sigma Denoiser", linestyle="dashed")
    plt.plot(t, flexible_filtered, label="Flexible Dynamic Denoiser", linestyle="dashed", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("📉 Comparison of Advanced Noise Mitigation Methods")
    plt.legend()
    plt.grid()
    plt.show()

    print("\n📊 Evaluation Statistics:")
    for method, rmse in stats.items():
        print(f"{method}: {rmse:.4f}")

# 🔬 Example Usage
t, noisy_signal = generate_noisy_signal(freq=5, duration=2, sampling_rate=1000, noise_std=0.2)
compare_denoising_methods(noisy_signal, t)
```

---

## 4. Conclusion  

This framework provides a **multi-modal, adaptable noise suppression strategy**, integrating:
✔ **Variance estimation techniques**  
✔ **Correlation-based filtering**  
✔ **Adaptive resampling and hybrid denoising**  
✔ **Dynamic fusion-based optimization**  

The **flexible selection mechanism** ensures that the most effective method is applied in real-time, enhancing **signal fidelity and robustness** while maintaining **computational efficiency**.

---
# D Generate Noisy Signals: Function Overview  

## 1. Introduction  
In signal processing, testing denoising algorithms requires synthetic noisy signals that simulate real-world conditions. 
The `generate_noisy_signals()` function provides a **flexible framework** for generating various signal types with **controllable noise levels**.  

This function enables **benchmarking of denoising methods** by providing a diverse range of signals affected by adjustable noise characteristics.

---

## 2. Function Purpose  

### **Objective**  
✔ Generate synthetic signals with different waveforms.  
✔ Introduce **controllable Gaussian noise** to simulate real-world distortion.  
✔ Provide a **customizable signal generation** framework for testing and optimization.  

### **Applications**  
✅ Benchmarking noise reduction techniques.  
✅ Simulating real-world measurement noise.  
✅ Validating adaptive filtering methods.  

---

## 3. Implementation Details  

### **3.1 Function Parameters**  

| **Parameter**   | **Description** |
|----------------|----------------|
| `signal_type`  | Type of signal to generate (`sine`, `square`, `sawtooth`, `gaussian`). |
| `freq`         | Frequency of the generated signal (Hz). |
| `duration`     | Total length of the signal (seconds). |
| `sampling_rate`| Number of samples per second. |
| `noise_std`    | Standard deviation of Gaussian noise added to the signal. |

### **3.2 Signal Generation Process**  

#### **Step 1: Time Vector Creation**  
The function first **constructs the time vector** with evenly spaced samples:


$$
\[
t = \text{linspace}(0, \text{duration}, \text{sampling_rate} \times \text{duration})
\]
$$



where `linspace` ensures consistent time intervals.

#### **Step 2: Clean Signal Generation**  
Based on the selected `signal_type`, the function generates the clean waveform:
- **Sine wave** → $$\( \sin(2\pi f t) \)$$  
- **Square wave** → Alternating high/low states.
- **Sawtooth wave** → Linearly increasing ramp signal.
- **Gaussian noise** → Random normally distributed values.

#### **Step 3: Noise Addition**  
To simulate realistic conditions, **Gaussian noise** is applied:


$$
\[
\text{noisy signal} = \text{clean signal} + \mathcal{N}(0, \text{noise_std})
\]
$$



where $$\( \mathcal{N}(0, \text{noise_std}) \)$$ represents a **zero-mean normal distribution**.

#### **Step 4: Return Values**  
The function returns:
- **Time vector `t`** (sample points).
- **Noisy signal array `signal_noisy`** for denoising experiments.

---

## 4. Error Handling  

The function includes a safeguard to **prevent invalid input types**:


$$
\[
\text{if signal type is invalid} \Rightarrow \text{raise ValueError}
\]
$$



This ensures that users provide a **supported signal type**.

---

## 5. Conclusion  

The `generate_noisy_signals()` function is a **powerful tool** for testing denoising algorithms, offering:  

✔ Multiple signal types.  
✔ Adjustable noise levels.  
✔ Customizable frequency and duration.  

This modular design **supports benchmarking and research applications**, making it an integral component of noise mitigation studies.  

### **Pythonic implementation:**  
```python
import numpy as np
import scipy.signal as signal

def generate_noisy_signals(signal_type="sine", freq=5, duration=2, sampling_rate=1000, noise_std=0.2):
    """Generate synthetic noisy signals for testing denoising methods."""
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Generate the clean signal
    if signal_type == "sine":
        clean_signal = np.sin(2 * np.pi * freq * t)
    elif signal_type == "square":
        clean_signal = signal.square(2 * np.pi * freq * t)
    elif signal_type == "sawtooth":
        clean_signal = signal.sawtooth(2 * np.pi * freq * t)
    elif signal_type == "gaussian":
        clean_signal = np.random.normal(0, 1, size=len(t))
    else:
        raise ValueError("Unsupported signal type. Choose from: 'sine', 'square', 'sawtooth', 'gaussian'.")

    # Add Gaussian noise
    noisy_signal = clean_signal + np.random.normal(0, noise_std, size=len(t))

    return t, noisy_signal
```

---
# E Signal Processing Experiment Results I: Sine Wave Signal

## 1. Introduction
This document presents the results of a signal processing experiment using various noise mitigation techniques. The primary focus is 
on evaluating the **efficacy of different denoising strategies** through statistical analysis and visual representation.

## 2. Signal Characteristics
The test signal consists of:
- **Sampling Rate**: 1000 Hz
- **Signal Duration**: 2 seconds
- **Signal Type**: Sine wave with added Gaussian noise (standard deviation = 0.2)

## 3. Noise Reduction Techniques Evaluated
The following noise reduction methods were applied:
1️⃣ **Hybrid Multi-Pass Median Filtering**  
2️⃣ **Enhanced Correlation-Based Noise Reduction**  
3️⃣ **Adaptive Beta-Sigma Resampling**  
4️⃣ **Hybrid Correlated Beta-Sigma Denoiser**  
5️⃣ **Flexible Dynamic Denoiser**

## 4. Evaluation Metrics
The performance of each method was assessed using the **Root Mean Square Error (RMSE)**:

| **Method**                                       | **RMSE** |
|--------------------------------------------------|----------|
| Hybrid Multi-Pass Median Filtering              | 0.1625   |
| Enhanced Correlation-Based Noise Reduction      | 0.1037   |
| Adaptive Beta-Sigma Resampling                  | **0.0197** |
| Hybrid Correlated Beta-Sigma Denoiser           | 0.0629   |
| Flexible Dynamic Denoiser                       | 0.0610   |   

### **Key Observations**:
✔ **Adaptive Beta-Sigma Resampling** achieved the lowest RMSE (0.0197), suggesting it is highly effective at noise suppression while maintaining signal integrity.  
✔ **Enhanced Correlation-Based Denoising** also performed well (0.1037 RMSE), showing solid noise reduction with an analytical approach.  
✔ **Hybrid Multi-Pass Median Filtering** had a higher RMSE (0.1625), which may indicate that it struggled with preserving signal accuracy while removing noise.  
✔ **Hybrid Correlated Beta-Sigma Denoiser** and **Flexible Dynamic Denoiser** delivered similar performance (0.0629 and 0.0610 RMSE, respectively), striking a balance between different techniques.

## 5. Visual Representation
Below is a **plot illustrating the comparative results** of the different denoising techniques applied to the noisy signal:

![Noisy Sine Signal](https://github.com/NenadBalaneskovic/ExternalProjects/blob/c76c76aab44c215e883d596dcc9dcfb51295fb44/SignalNoiseMitigation/res1.PNG)  

Looking at the results, the graph presents a detailed comparison of different noise mitigation techniques applied to the noisy signal.  
The evaluation statistics indicate how well each method reduces noise based on their Root Mean Square Error (RMSE) values. 

The graph visually illustrates how each method affects the signal shape, with smoother curves representing stronger denoising performance. 
Overall, the results suggest that **Adaptive Beta-Sigma Resampling** might be the best candidate for high-fidelity noise removal while maintaining essential signal details.

## 6. Conclusion
The results indicate that **Adaptive Beta-Sigma Resampling** achieved the **lowest RMSE**, suggesting it provides the most effective noise suppression 
while maintaining signal fidelity. However, **Hybrid Correlated Beta-Sigma Denoiser** and **Flexible Dynamic Denoiser** also delivered strong performance, 
balancing **noise removal and signal preservation** effectively.

Further refinements could involve **optimizing fusion weights** in the **Flexible Dynamic Denoiser** to enhance adaptability across signal types.

---
# F Signal Processing Experiment Results II: Square Wave Signal

## 1. Introduction  
This document presents the experimental results of **noise reduction techniques** applied to a **square wave signal** contaminated with noise. The primary objective 
is to **evaluate and compare denoising methods** based on their effectiveness in preserving the sharp transitions of the square wave while reducing noise.

---

## 2. Signal Characteristics  
The test signal consists of:  
- **Sampling Rate**: 1000 Hz  
- **Signal Duration**: 2 seconds  
- **Signal Type**: Square Wave with added Gaussian noise  
- **Noise Standard Deviation**: 0.2  

### **2.1 Challenges of Square Wave Denoising**  
Square waves have **sharp transitions** between high and low states, making noise mitigation particularly challenging. Traditional smoothing filters tend to 
**distort edges**, so **adaptive filtering methods** must be used to preserve the square wave's key features.

---

## 3. Applied Denoising Methods  
The following **five noise suppression techniques** were tested:  
1️⃣ **Hybrid Multi-Pass Median Filtering** → Edge-preserving smoothing.  
2️⃣ **Enhanced Correlation-Based Noise Reduction** → Autocorrelation-based filtering.  
3️⃣ **Adaptive Beta-Sigma Resampling** → Dynamic noise suppression using local variance analysis.  
4️⃣ **Hybrid Correlated Beta-Sigma Denoiser** → Fusion of correlation-based and beta-sigma filtering.  
5️⃣ **Flexible Dynamic Denoiser** → Real-time adaptive method selection.

---

## 4. Evaluation Metrics  
Performance was assessed using **Root Mean Square Error (RMSE)**, which quantifies the difference between the original and denoised signals:

| **Denoising Method**                            | **RMSE Score** |
|-------------------------------------------------|---------------|
| Hybrid Multi-Pass Median Filtering              | 0.1678        |
| Enhanced Correlation-Based Noise Reduction      | 0.0786        |
| Adaptive Beta-Sigma Resampling                  | 0.0828        |
| Hybrid Correlated Beta-Sigma Denoiser           | **0.0569**    |
| Flexible Dynamic Denoiser                       | 0.0624        |

### **Key Observations**  
✔ **Hybrid Correlated Beta-Sigma Denoiser** performed best, achieving the lowest RMSE (**0.0569**) and effectively balancing **noise suppression with signal preservation**.  
✔ **Flexible Dynamic Denoiser** demonstrated strong adaptability, adjusting denoising techniques **based on local signal complexity**.  
✔ **Median Filtering struggled with sharp transitions**, yielding the highest RMSE (**0.1678**), indicating possible edge distortion.  
✔ **Correlation-Based Noise Reduction and Beta-Sigma Resampling** delivered **moderate noise suppression**, maintaining a fair balance between **smoothness and structural integrity**.  

---

## 5. Visual Representation  
The following plot illustrates the **comparative performance of different denoising techniques** applied to the square wave signal:

![Noisy Square Signal](https://github.com/NenadBalaneskovic/ExternalProjects/blob/7e583cf2d27a2d0e3cd5c9aa52035511d1c96647/SignalNoiseMitigation/res2.PNG)  

The square wave structure makes noise mitigation more challenging due to its sharp transitions. The Hybrid Correlated Beta-Sigma Denoiser appears to be the most effective in 
preserving the waveform while reducing unwanted variations.

---

## 6. Conclusion  
The **Hybrid Correlated Beta-Sigma Denoiser** is identified as the **most effective method** for square wave signals, preserving sharp edges while mitigating unwanted noise.
The **Flexible Dynamic Denoiser** also showed promising results, making it a viable alternative for **adaptive real-time applications**.

Future research could explore **fusion-based optimization strategies** to enhance **edge preservation** while maximizing noise suppression.

---
# G Signal Processing Experiment Results III: Sawtooth Wave Signal

## 1. Introduction  
This document presents the experimental results of **noise reduction techniques** applied to a **sawtooth wave signal** contaminated with noise. 
The goal is to evaluate different denoising methods based on their ability to suppress noise while maintaining the **linearly increasing ramp structure** of the sawtooth wave.

---

## 2. Signal Characteristics  
The test signal consists of:  
- **Sampling Rate**: 1000 Hz  
- **Signal Duration**: 2 seconds  
- **Signal Type**: Sawtooth Wave with added Gaussian noise  
- **Noise Standard Deviation**: 0.2  

### **2.1 Challenges of Sawtooth Wave Denoising**  
Unlike square waves, sawtooth waves have **gradual linear transitions**, making them susceptible to noise distortion. The challenge in denoising is to **remove random noise** 
without **blurring the edges** or **affecting the ramp slope**.

---

## 3. Applied Denoising Methods  
The following **five noise suppression techniques** were tested:  
1️⃣ **Hybrid Multi-Pass Median Filtering** → Edge-preserving smoothing.  
2️⃣ **Enhanced Correlation-Based Noise Reduction** → Autocorrelation-based filtering.  
3️⃣ **Adaptive Beta-Sigma Resampling** → Dynamic noise suppression using local variance analysis.  
4️⃣ **Hybrid Correlated Beta-Sigma Denoiser** → Fusion of correlation-based and beta-sigma filtering.  
5️⃣ **Flexible Dynamic Denoiser** → Real-time adaptive method selection.

---

## 4. Evaluation Metrics  
Performance was assessed using **Root Mean Square Error (RMSE)**, which quantifies the difference between the original and denoised signals:

| **Denoising Method**                            | **RMSE Score** |
|-------------------------------------------------|---------------|
| Hybrid Multi-Pass Median Filtering              | 0.1648        |
| Enhanced Correlation-Based Noise Reduction      | 0.0533        |
| Adaptive Beta-Sigma Resampling                  | 0.0508        |
| Hybrid Correlated Beta-Sigma Denoiser           | **0.0371**    |
| Flexible Dynamic Denoiser                       | 0.0534        |

### **Key Observations**  
✔ **Hybrid Correlated Beta-Sigma Denoiser** was the most effective method for **sawtooth wave signals**, achieving the lowest RMSE (**0.0371**).  
✔ **Adaptive Beta-Sigma Resampling** and **Correlation-Based Denoising** performed similarly, preserving ramp features while reducing noise.  
✔ **Median Filtering introduced some edge distortion**, yielding the highest RMSE (**0.1648**).  
✔ **Flexible Dynamic Denoiser** adapted well to local signal variations, maintaining a stable performance.

---

## 5. Visual Representation  
The following plot illustrates the **comparative performance of different denoising techniques** applied to the sawtooth wave signal:

![Noisy Sawtooth Signal](https://github.com/NenadBalaneskovic/ExternalProjects/blob/8a9ac174ff282954baf8cd5f4807a3faf8643854/SignalNoiseMitigation/res3.PNG)  

The results suggest that Hybrid Correlated Beta-Sigma Denoiser is the most effective in preserving the waveform while reducing unwanted 
variations. The Adaptive Beta-Sigma Resampling method also performed well, balancing noise suppression and signal fidelity.

---

## 6. Conclusion  
The results suggest that **Hybrid Correlated Beta-Sigma Denoiser** is the **most effective method** for preserving the **linear ramp structure** of the sawtooth wave while minimizing noise interference.  

Further refinements could involve **adaptive fusion enhancements** to fine-tune edge preservation and dynamic signal complexity adjustments.

---
# H Signal Processing Experiment Results IV: Gaussian Wave Signal

## 1. Introduction  
This document presents the experimental results of **noise reduction techniques** applied to a **Gaussian signal** contaminated with noise. The primary goal is to evaluate different 
denoising methods based on their ability to suppress noise while maintaining the statistical properties of the Gaussian distribution.

---

## 2. Signal Characteristics  
The test signal consists of:  
- **Sampling Rate**: 1000 Hz  
- **Signal Duration**: 2 seconds  
- **Signal Type**: Gaussian signal with added noise  
- **Noise Standard Deviation**: 0.2  

### **2.1 Challenges of Gaussian Signal Denoising**  
Gaussian signals are inherently stochastic, meaning they have no distinct edges or periodic features. The challenge in denoising is to **reduce unwanted noise fluctuations** 
while preserving the underlying statistical distribution.

---

## 3. Applied Denoising Methods  
The following **five noise suppression techniques** were tested:  
1️⃣ **Hybrid Multi-Pass Median Filtering** → Adaptive smoothing for random fluctuations.  
2️⃣ **Enhanced Correlation-Based Noise Reduction** → Uses autocorrelation for statistical correction.  
3️⃣ **Adaptive Beta-Sigma Resampling** → Dynamic noise suppression based on local variance analysis.  
4️⃣ **Hybrid Correlated Beta-Sigma Denoiser** → Fusion of correlation-based and beta-sigma filtering.  
5️⃣ **Flexible Dynamic Denoiser** → Real-time adaptive method selection.

---

## 4. Evaluation Metrics  
Performance was assessed using **Root Mean Square Error (RMSE)**, measuring deviations from the original signal:

| **Denoising Method**                            | **RMSE Score** |
|-------------------------------------------------|---------------|
| Hybrid Multi-Pass Median Filtering              | 0.1723        |
| Enhanced Correlation-Based Noise Reduction      | 0.0517        |
| Adaptive Beta-Sigma Resampling                  | 0.0482        |
| Hybrid Correlated Beta-Sigma Denoiser           | **0.0351**    |
| Flexible Dynamic Denoiser                       | **0.0294**    |

### **Key Observations**  
✔ **Flexible Dynamic Denoiser** achieved the lowest RMSE (**0.0294**), demonstrating strong adaptability in handling Gaussian noise fluctuations.  
✔ **Hybrid Correlated Beta-Sigma Denoiser** followed closely (**0.0351 RMSE**), providing effective suppression while maintaining statistical integrity.  
✔ **Adaptive Beta-Sigma Resampling** and **Enhanced Correlation-Based Denoising** delivered **balanced noise reduction** without excessive smoothing.  
✔ **Hybrid Multi-Pass Median Filtering** struggled with maintaining signal accuracy, yielding the highest RMSE (**0.1723**), indicating **significant distortion** in stochastic signals.

---

## 5. Visual Representation  
The following plot illustrates the **comparative performance of different denoising techniques** applied to the Gaussian signal:

![Noisy Gaussian Signal](https://github.com/NenadBalaneskovic/ExternalProjects/blob/2601597a60bea77e9345eea149df42d9ca779354/SignalNoiseMitigation/res4.PNG)  

This result suggests that Flexible Dynamic Denoiser is the most effective in preserving the waveform while reducing unwanted variations. 
The Hybrid Correlated Beta-Sigma Denoiser also performed well, offering a strong balance between noise suppression and signal preservation.

---

## 6. Conclusion  
The **Flexible Dynamic Denoiser** proved to be the **most effective method** for Gaussian noise suppression, adapting dynamically to signal fluctuations while maintaining fidelity.
The **Hybrid Correlated Beta-Sigma Denoiser** also performed well, offering a structured approach to **adaptive noise filtering**.

Future enhancements may explore **ensemble-based fusion strategies** to refine real-time adaptability and **optimize signal preservation**.

---
# I Interpretation  

We offer a structured overview of the results, highlighting each method's performance based on RMSE values and overall effectiveness.

| **Method**                                       | **Result 1 (RMSE)** | **Result 2 (RMSE)** | **Result 3 (RMSE)** | **Result 4 (RMSE)** | **Key Observations** |
|--------------------------------------------------|---------------------|---------------------|---------------------|---------------------|----------------------|
| **Hybrid Multi-Pass Median Filtering**          | 0.1625              | 0.1678              | 0.1648              | 0.1723              | Struggled with noise reduction while preserving accuracy. |
| **Enhanced Correlation-Based Denoising**        | 0.1037              | 0.0786              | 0.0533              | 0.0517              | Performed reasonably well but not the best at maintaining signal fidelity. |
| **Adaptive Beta-Sigma Resampling**              | **0.0197**          | 0.0828              | 0.0508              | 0.0482              | Best performer in Result 1 but varied effectiveness in other cases. |
| **Hybrid Correlated Beta-Sigma Denoiser**       | 0.0629              | **0.0569**          | **0.0371**          | 0.0351              | Strong hybrid approach, consistently effective. |
| **Flexible Dynamic Denoiser**                   | 0.0610              | 0.0624              | 0.0534              | **0.0294**          | Most adaptable technique, performed best in Result 4. |

### 📌 **Summary Interpretation:**
- **Adaptive Beta-Sigma Resampling** was highly effective in **Result 1**, but performance fluctuated across different signal types.
- **Hybrid Correlated Beta-Sigma Denoiser** demonstrated strong consistency, especially excelling in **Result 3**.
- **Flexible Dynamic Denoiser** proved to be the **best performer in Result 4**, showing adaptability across scenarios.
- **Hybrid Multi-Pass Median Filtering** had the highest RMSE in all cases, indicating it struggled compared to other methods.

Based on the observed RMSE values and overall effectiveness across the different noisy signal cases, here's the **proposed ranking** of the denoising functions:

### **🏆 Function Ranking (Best to Least Effective)**
1️⃣ **Flexible Dynamic Denoiser** → Performed best in Result 4 and maintained strong adaptability across different signals.  
2️⃣ **Hybrid Correlated Beta-Sigma Denoiser** → Consistently strong performance, particularly excelling in Result 3.  
3️⃣ **Adaptive Beta-Sigma Resampling** → Outstanding in Result 1 but showed fluctuations across cases.  
4️⃣ **Enhanced Correlation-Based Denoising** → Delivered moderate effectiveness, balancing noise removal and signal fidelity.  
5️⃣ **Hybrid Multi-Pass Median Filtering** → Consistently had the highest RMSE, indicating struggles with maintaining accuracy.  

### 📌 **Summary Insight:**
- **Flexible Dynamic Denoiser** seems the **most adaptable**, making it an excellent all-purpose choice.  
- **Hybrid Correlated Beta-Sigma Denoiser** shows **strong hybrid performance**, especially for structured signals.  
- **Adaptive Beta-Sigma Resampling** might be **ideal for specific conditions**, but its effectiveness varies.  
- **Enhanced Correlation-Based Denoising** provides **moderate results**, but not the best option overall.  
- **Hybrid Multi-Pass Median Filtering** may need adjustments or modifications to improve reliability.  
---
# J 🔎 Future refinements and improvements  

Below is a **structured overview** of possible refinements for each denoising technique based on the observed performance results:

| **Method**                                      | **Observed Limitations**                              | **Suggested Refinements** |
|-------------------------------------------------|-----------------------------------------------------|---------------------------|
| **⚙️ Flexible Dynamic Denoiser**                   | Slight fluctuations in certain signal cases.       | Enhance adaptive weighting strategy to further fine-tune adjustments based on local signal complexity. |
| **🔗 Hybrid Correlated Beta-Sigma Denoiser**        | Occasional residual noise in high-frequency signals. | Improve correlation weighting to emphasize local signal coherence for better accuracy. |
| **🔄 Adaptive Beta-Sigma Resampling**              | Performance varied across different scenarios.    | Implement dynamic beta-scaling adjustments based on signal type to increase stability across cases. |
| **📊 Enhanced Correlation-Based Denoising**        | Moderate noise suppression, but not the best at preserving sharp transitions. | Improve autocorrelation estimation by incorporating multi-window averaging techniques to adapt better to abrupt changes. |
| **🔹 Hybrid Multi-Pass Median Filtering**          | Consistently had the highest RMSE, struggling with noise removal and signal integrity. | Introduce edge-preserving smoothing techniques to mitigate excessive blurring and refine the kernel adaptation process. |

### 📌 **Summary of Refinements:**
✅ **Flexible Dynamic Denoiser:** Fine-tune **adaptive fusion weights** for **greater stability.**  
✅ **Hybrid Correlated Beta-Sigma Denoiser:** Adjust **correlation weighting** to **better capture signal coherence.**  
✅ **Adaptive Beta-Sigma Resampling:** Optimize **beta scaling** to ensure **consistent performance.**  
✅ **Enhanced Correlation-Based Denoising:** Improve **autocorrelation methods** to **preserve sharp transitions.**  
✅ **Hybrid Multi-Pass Median Filtering:** Enhance **edge-preserving techniques** to **prevent signal distortion.**  

This refinement strategy should improve overall denoising effectiveness while ensuring signal fidelity.  

# 📚 References
1. **Scipy Signal** –  https://docs.scipy.org/doc/scipy/reference/signal.html, https://scipy.org/, https://docs.scipy.org/doc/scipy/tutorial/signal.html
2. [![Jupyter Notebook | English](https://img.shields.io/badge/Jupyter%20Notebook-English-yellowblue?logoColor=blue&labelColor=yellow)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/ec17bd68fcaadcf322445315762ab7a2884050b6/SignalNoiseMitigation/Signal_Denoising.ipynb)
3. [![Signal Analysis Report | English](https://img.shields.io/badge/Signal%20Analysis%20Report-English-yellowblue?logoColor=blue&labelColor=red)](https://github.com/NenadBalaneskovic/ExternalProjects/blob/26be04b6bceb4ea17e676b5b9625c9156c635ad7/SignalNoiseMitigation/SignalDenoisingAnalysis.pdf) 
4. **Statsmodel package documentation** – https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html, https://www.statsmodels.org/stable/index.html
5. **NumPy Documentation** – https://numpy.org/
6. **Matplotlib Documentation** - https://matplotlib.org/
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
31. Thomas Haslwanter: „__Hands-on Signal Analysis with Python: An Introduction__“, Springer (2021).
32. Jose Unpingco: „__Python for Signal Processing__“, Springer (2023).

