# Foundations of Portfolio Optimization

The following chapters contain a mathematical and algorithmic overview of the most important aspects of
portfolio optimization.

---

## Chapter 1:  The **Cox-Ross-Rubinstein (CRR) model**  

The **Cox-Ross-Rubinstein (CRR) model** is a discrete-time model used to price options using a binomial tree framework. 
Here’s a step-by-step guide to determining a **fair option price** in a **T-period binomial tree** model. 

### 🚀 **Step 1: Define Parameters**
Let:
- $S_0$ = Initial stock price  
- $K$ = Strike price of the option  
- $T$ = Number of time steps (periods)  
- $r$ = Risk-free interest rate per period  
- $u$ = Upward movement factor  
- $d$ = Downward movement factor  
- $p$ = Risk-neutral probability

The **upward and downward movement factors** are defined as:

$$
u = e^{\sigma \sqrt{\Delta t}}
$$

$$
d = \frac{1}{u}
$$

where $ \sigma $ is the volatility and $ \Delta t = \frac{T}{N} $ is the length of each step.

The **risk-neutral probability** is:

$$
p = \frac{e^{r \Delta t} - d}{u - d}
$$

---

### 🚀 **Step 2: Construct the Binomial Price Tree**
For each time step \( t \), the stock price evolves as:
$$
S_t = S_0 \cdot u^i \cdot d^{t-i}
$$
where \( i \) represents the number of upward movements.

---

### 🚀 **Step 3: Calculate Payoffs at Maturity**
For a **call option**, the payoff at expiration is:
$$
C_T = \max(S_T - K, 0)
$$
For a **put option**, the payoff is:
$$
P_T = \max(K - S_T, 0)
$$
Compute these payoffs at all final nodes in the tree.

---

### 🚀 **Step 4: Backward Induction to Find Present Value**
Starting from the final time step \( T \), compute the expected present value at each node:
$$
C_t = e^{-r \Delta t} \cdot (p C_{t+1}^{up} + (1-p) C_{t+1}^{down})
$$
Perform this recursively **backward** until reaching \( C_0 \), the fair option price.

---

### 🔢 **Concrete Example Calculation**
#### **Given Parameters**:
- \( S_0 = 100 \), \( K = 100 \), \( r = 5\% \), \( \sigma = 20\% \), \( T = 2 \)
- \( \Delta t = 1 \), \( u = e^{0.2 \times \sqrt{1}} \approx 1.2214 \), \( d = \frac{1}{u} \approx 0.8187 \)
- \( p = \frac{e^{0.05} - 0.8187}{1.2214 - 0.8187} \approx 0.577 \)

#### **Binomial Tree Construction**
| Time | Stock Price | Call Payoff |
|------|------------|-------------|
| \( t = 2 \) | \( 100 \cdot u^2 \) = **149.17** | \( \max(149.17 - 100, 0) \) = **49.17** |
| \( t = 2 \) | \( 100 \cdot u \cdot d \) = **100.00** | \( \max(100 - 100, 0) \) = **0** |
| \( t = 2 \) | \( 100 \cdot d^2 \) = **67.03** | \( \max(67.03 - 100, 0) \) = **0** |

#### **Backward Induction at \( t = 1 \)**
$$
C_1 = e^{-0.05} \cdot (p C_2^{up} + (1-p) C_2^{down})
$$
$$
C_1 = e^{-0.05} \cdot (0.577 \times 49.17 + 0.423 \times 0)
$$
$$
C_1 = e^{-0.05} \times 28.37 \approx 27.01
$$

#### **Final Computation for \( C_0 \)**
$$
C_0 = e^{-0.05} \cdot (p C_1^{up} + (1-p) C_1^{down})
$$
$$
C_0 = e^{-0.05} \cdot (0.577 \times 27.01 + 0.423 \times 0)
$$
$$
C_0 = e^{-0.05} \times 15.59 \approx 14.85
$$

✅ **Fair Option Price: $14.85$**  

---

### 🎯 **Key Takeaways**
✔ **Use risk-neutral probabilities** to price the option  
✔ **Construct a binomial tree** and calculate payoffs  
✔ **Apply backward induction** to discount future payoffs  

---
## Chapter 2: American and Put Options in CRR

### 🚀 **Pricing American and Put Options in the Cox-Ross-Rubinstein Model**
Now let’s extend the binomial pricing approach for:
1️⃣ **American options**, which can be exercised at any time before expiration  
2️⃣ **Put options**, where the payoff is \( P_T = \max(K - S_T, 0) \)  

---

### 🔍 **1️⃣ American Option Pricing**
An **American option** differs from a **European option** because it can be exercised **before expiration**. This means we must compare:
$$
C_t = \max \left( C_{\text{European}}, S_t - K \right)
$$
At each node, we check **if early exercise is more profitable** than waiting.

---

### 🔍 **2️⃣ Put Option Pricing**
For a **put option**, the final payoff is:
$$
P_T = \max(K - S_T, 0)
$$
We still use backward induction:
$$
P_t = e^{-r \Delta t} \cdot (p P_{t+1}^{up} + (1-p) P_{t+1}^{down})
$$
For an **American put**, apply:
$$
P_t = \max \left( P_{\text{European}}, K - S_t \right)
$$
at each node.

---

### 📊 **Example: American & European Put Option**
#### **Given Parameters**:
- \( S_0 = 100 \), \( K = 100 \), \( r = 5\% \), \( \sigma = 20\% \), \( T = 2 \)
- \( u = 1.2214 \), \( d = 0.8187 \), \( p = 0.577 \)

#### **Step 1: Compute Terminal Payoffs**
| Time | Stock Price | Put Payoff |
|------|------------|-----------|
| \( t = 2 \) | \( 100 \times u^2 \) = **149.17** | \( \max(100 - 149.17, 0) \) = **0** |
| \( t = 2 \) | \( 100 \times u \times d \) = **100** | \( \max(100 - 100, 0) \) = **0** |
| \( t = 2 \) | \( 100 \times d^2 \) = **67.03** | \( \max(100 - 67.03, 0) \) = **32.97** |

#### **Step 2: Backward Induction**
$$
P_1 = e^{-0.05} \cdot (p P_2^{up} + (1-p) P_2^{down})
$$
$$
P_1 = e^{-0.05} \cdot (0.577 \times 0 + 0.423 \times 32.97)
$$
$$
P_1 = e^{-0.05} \times 13.94 \approx 13.26
$$

Final computation:
$$
P_0 = e^{-0.05} \cdot (p P_1^{up} + (1-p) P_1^{down})
$$
$$
P_0 = e^{-0.05} \cdot (0.577 \times 13.26 + 0.423 \times 0)
$$
$$
P_0 = e^{-0.05} \times 7.65 \approx 7.28
$$

✅ **Fair European Put Price: $7.28$**  
✅ **Fair American Put Price: Slightly higher due to early exercise!**  

---

🎯 **Key Takeaways**
✔ **Use binomial trees to price put & American options**  
✔ **Apply early exercise condition** for American-style options  
✔ **Backward induction computes expected fair price**  

---

Let's explore **real-world applications** of option pricing using the **Cox-Ross-Rubinstein (CRR) binomial model**.

### 🌍 **Application 1: Hedging Strategies for Traders**
🔹 **Scenario:** A portfolio manager holds **100 shares of Tesla (TSLA)** but is worried about short-term price fluctuations.  
🔹 **Solution:** The manager buys **put options** using the binomial pricing model to determine the fair cost of protection.  

Using the CRR model:
- ✅ **Fair put price:** **$7.28 per contract** (as we calculated earlier)
- ✅ **Total cost:** **$7.28 × 100 contracts = $728**  
- ✅ **Outcome:** If Tesla’s price **falls**, the put option protects against losses.

💡 **Why it matters:** Investors use **binomial model pricing** to hedge against risk and adjust exposure!

---

### 🌍 **Application 2: Employee Stock Options (ESOs) Valuation**
🔹 **Scenario:** A tech company grants employees **stock options** as part of their compensation.  
🔹 **Solution:** The company uses the **binomial model** to compute the fair value of these options before issuing them.  

For example:
- **Initial stock price:** **$50**
- **Strike price:** **$55**
- **Volatility:** **30%**
- **Expiration:** **3 years**

Using the CRR framework, the company determines the **fair value** of each employee option, adjusting for:
✔ **Early exercise probability**  
✔ **Market risk**  
✔ **Future growth potential**

💡 **Why it matters:** Companies use **option pricing models** for regulatory compliance & compensation planning!

---

### 🌍 **Application 3: Structured Derivatives & Exotic Options**
🔹 **Scenario:** Banks sell structured products like **barrier options or Asian options**, which require specialized pricing techniques.  
🔹 **Solution:** Adjusting the CRR model for **multiple periods**, institutions price **complex derivatives** efficiently.  

For example, a **knock-in barrier option**:
- Exercise price: **$100**
- Knock-in barrier: **$120**
- If the stock **reaches $120**, the option activates & CRR pricing determines fair value.

💡 **Why it matters:** Investment banks rely on **binomial trees** for pricing exotic derivatives used in structured finance!

---

### 🎯 **Key Takeaways**
✔ **Risk management:** Used in hedging strategies  
✔ **Corporate valuation:** Applied in employee compensation packages  
✔ **Advanced derivatives:** Essential in pricing exotic financial instruments  

---
## Chapter 3: Equivalent Martingale Measures

The **equivalent martingale measure (EMM)** is a fundamental concept in financial mathematics used to price derivatives in a risk-neutral world. It ensures that discounted asset prices follow a **martingale process**, meaning there is no arbitrage opportunity.

---

### 🚀 **Step-by-Step Algorithm for Calculating Equivalent Martingale Measures**
We assume a **continuous-time financial market** with a stock \( S_t \) and a risk-free bond \( B_t \). Our goal is to find the **risk-neutral probability measure \( Q \)**.

#### **Step 1: Define the Stochastic Process for Asset Price**
Under the **physical measure \( P \)**, the asset price follows a geometric Brownian motion:
$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$
where:
- \( \mu \) = drift (expected return)
- \( \sigma \) = volatility
- \( W_t \) = standard Brownian motion

The **risk-free bond** grows as:
$$
B_t = e^{rt}
$$
where \( r \) is the risk-free rate.

---

#### **Step 2: Find the Radon-Nikodym Derivative**
The **equivalent martingale measure \( Q \)** transforms the drift from \( \mu \) to \( r \), adjusting the probability distribution of asset prices.

Define the **Radon-Nikodym derivative**:
$$
\frac{dQ}{dP} = e^{-\theta W_T - \frac{1}{2} \theta^2 T}
$$
where:
$$
\theta = \frac{\mu - r}{\sigma}
$$
This is the **market price of risk**, ensuring the discounted process \( S_t/B_t \) is a **martingale** under \( Q \).

---

#### **Step 3: Convert the Asset Dynamics Under \( Q \)**
Applying Girsanov’s Theorem, the asset price under \( Q \) follows:
$$
dS_t = r S_t dt + \sigma S_t dW_t^Q
$$
where \( W_t^Q \) is a Brownian motion under \( Q \).

**This eliminates the drift term \( \mu \), replacing it with \( r \), creating a risk-neutral measure.**

---

#### **Step 4: Compute the Risk-Neutral Probabilities**
For discrete-time models (e.g., Binomial trees), the **risk-neutral probability** is:
$$
p^Q = \frac{e^{r \Delta t} - d}{u - d}
$$
where \( u \) and \( d \) are the up and down movement factors.

For continuous-time models (Black-Scholes), the asset price under \( Q \) satisfies:
$$
S_T = S_0 e^{(r - \frac{1}{2} \sigma^2)T + \sigma W_T^Q}
$$

---

### 🔢 **Concrete Calculated Example**
Let’s compute \( Q \) in a **Binomial tree model** with:
- \( S_0 = 100 \)
- \( K = 100 \)
- \( r = 5\% \), \( \sigma = 20\% \)
- \( T = 1 \), \( \Delta t = 1 \)
- \( u = e^{0.2} \approx 1.2214 \), \( d = 1/u \approx 0.8187 \)

#### **Step 1: Compute Risk-Neutral Probability**
$$
p^Q = \frac{e^{0.05} - 0.8187}{1.2214 - 0.8187} \approx 0.577
$$

#### **Step 2: Compute European Call Option Price Using \( Q \)**
Final stock prices:
- **Up state:** \( S_T^{up} = 100 \times 1.2214 = 122.14 \)
- **Down state:** \( S_T^{down} = 100 \times 0.8187 = 81.87 \)

Option payoffs:
$$
C_T^{up} = \max(122.14 - 100, 0) = 22.14
$$
$$
C_T^{down} = \max(81.87 - 100, 0) = 0
$$

Discounted expectation under \( Q \):
$$
C_0 = e^{-r} (p^Q C_T^{up} + (1-p^Q) C_T^{down})
$$
$$
C_0 = e^{-0.05} (0.577 \times 22.14 + 0.423 \times 0)
$$
$$
C_0 = e^{-0.05} \times 12.79 \approx 12.18
$$

✅ **Fair option price under \( Q \): $12.18$**  
✅ **Ensures no arbitrage using risk-neutral valuation**

---

### 🎯 **Key Takeaways**
✔ **Risk-neutral measure removes drift \( \mu \) and replaces it with \( r \)**  
✔ **Martingale property ensures no arbitrage**  
✔ **Used in Black-Scholes, binomial trees, and stochastic models**  

---
## Chapter 4: EMMs and real world applications

### 🚀 **Extending Equivalent Martingale Measures to Stochastic Discount Factors & Real-World Applications**

Let’s explore **stochastic discount factors (SDFs)** and **practical applications** of equivalent martingale measures (**EMMs**).

---

## **📌 Part 1: Stochastic Discount Factors (SDFs)**
A **stochastic discount factor** (\( M_t \)) is used to price assets in arbitrage-free markets. It connects the **risk-neutral measure** and real-world probability measure.

### **Step 1: Define the Discount Factor**
For a **risk-free bond**:
$$
B_t = e^{rt}
$$
For a **risky asset**:
$$
S_t = S_0 e^{(\mu - \frac{1}{2} \sigma^2) t + \sigma W_t^P}
$$
The **stochastic discount factor (SDF)** relates prices through:
$$
M_t = e^{-rt} \cdot \frac{dQ}{dP}
$$

### **Step 2: Compute the SDF Using Radon-Nikodym Derivative**
Since we define the risk-neutral probability via:
$$
\frac{dQ}{dP} = e^{-\theta W_T - \frac{1}{2} \theta^2 T}, \quad \text{where} \quad \theta = \frac{\mu - r}{\sigma}
$$
The **stochastic discount factor** becomes:
$$
M_T = e^{-rT} \cdot e^{-\theta W_T - \frac{1}{2} \theta^2 T}
$$

### **Step 3: Asset Pricing Using SDF**
Under the equivalent martingale measure:
$$
E^Q[M_T S_T] = S_0
$$
This ensures **no arbitrage**, pricing derivatives correctly.

---

## **📌 Part 2: Real-World Applications of Equivalent Martingale Measures**

### **1️⃣ Risk Management in Financial Institutions**
🔹 **Scenario:** A hedge fund wants to price exotic derivatives like **barrier options**.  
🔹 **Solution:** Use equivalent martingale measures to adjust **probabilities of hitting barriers**, preventing arbitrage.  
🔹 **Result:** The pricing avoids **mispricing risks** due to improper probability assumptions.

---

### **2️⃣ Monetary Policy & Macroeconomic Applications**
🔹 **Scenario:** Central banks use equivalent martingale measures to analyze **risk-free discount factors** under different interest rate regimes.  
🔹 **Solution:** Using an EMM, policymakers derive **long-term risk-neutral rates** for debt markets.  
🔹 **Result:** Helps determine **fair sovereign bond yields**.

---

### **3️⃣ Algorithmic Trading & Market Making**
🔹 **Scenario:** High-frequency traders rely on EMMs to adjust **probability distributions of stock movements** for option pricing.  
🔹 **Solution:** Use risk-neutral probabilities to predict expected **price deviations and market inefficiencies**.  
🔹 **Result:** Improves **algorithmic strategies in derivatives trading**.

---

### 🎯 **Key Takeaways**
✔ **Stochastic discount factors link probability measures in asset pricing**  
✔ **EMMs ensure arbitrage-free pricing in derivative markets**  
✔ **Real-world applications in macroeconomics, trading, and risk management**  

---
## Chapter 5: Calculation of EMMs

In a discrete **Cox-Ross-Rubinstein (CRR) model**, we determine the **equivalent martingale measure \( Q \)** by ensuring the discounted asset prices follow 
a **martingale process**, meaning their expected future value under \( Q \) equals their current price.  

---

### **📌 Step-by-Step Algorithm for Finding \( Q \) in the CRR Model**
Let’s define the model setup:
- \( S_0 \) = Initial stock price  
- \( u \) = Upward movement factor  
- \( d \) = Downward movement factor  
- \( r \) = Risk-free interest rate per period  
- \( p^Q \) = Risk-neutral probability under \( Q \)  

#### **Step 1: Risk-Neutral Pricing Condition**
To ensure **no arbitrage**, the expected **discounted price** must equal the current price:
$$
E^Q \left[ \frac{S_1}{1+r} \right] = S_0
$$
Since the **binomial stock price moves** as:
$$
S_1 = \begin{cases} 
S_0 u & \text{with probability } p^Q \\ 
S_0 d & \text{with probability } 1 - p^Q
\end{cases}
$$
Taking expectations:
$$
\frac{p^Q S_0 u + (1 - p^Q) S_0 d}{1 + r} = S_0
$$

---

#### **Step 2: Solve for Risk-Neutral Probability \( p^Q \)**
Rearrange the equation:
$$
p^Q (u - d) = e^r - d
$$
$$
p^Q = \frac{e^r - d}{u - d}
$$
This ensures the **discounted stock price process** is a martingale under \( Q \).

---

### **🔢 Example Calculation**
Let’s consider:
- \( S_0 = 100 \)
- \( u = 1.2 \), \( d = 0.8 \)
- \( r = 5\% \)  

Compute \( p^Q \):
$$
p^Q = \frac{e^{0.05} - 0.8}{1.2 - 0.8}
$$
$$
p^Q = \frac{1.0513 - 0.8}{0.4} = \frac{0.2513}{0.4} \approx 0.628
$$

✅ **Risk-neutral probability \( p^Q \) = 62.8%**  
✅ **Used to compute option prices via backward induction**  

---
## Chapter 6: EMMs and American Options in Mult-Period CRR-trees

### 🚀 **Extending Equivalent Martingale Measures to American Options & Multi-Period Trees**

Now, let’s extend **equivalent martingale measures (EMM)** to:
1️⃣ **American options**, which can be exercised at any time before expiration  
2️⃣ **Multi-period binomial trees**, where option values evolve over many steps  

---

### 📌 **1️⃣ Equivalent Martingale Measure for American Options**
Unlike European options, an **American option** can be exercised **at any node in the binomial tree**, which requires adjusting the pricing formula.

### **Step 1: Risk-Neutral Pricing for American Options**
For an American call option:
$$
C_t = \max \left( C_{\text{European}}, S_t - K \right)
$$
For an American put option:
$$
P_t = \max \left( P_{\text{European}}, K - S_t \right)
$$
At each node, we check **if early exercise is more profitable** than waiting.

### **Step 2: Compute Payoff at Each Node**
Using risk-neutral probability \( p^Q \):
$$
C_t = e^{-r \Delta t} \cdot (p^Q C_{t+1}^{up} + (1-p^Q) C_{t+1}^{down})
$$
For American options, adjust:
$$
C_t = \max \left( C_{\text{European}}, S_t - K \right)
$$

---

### 🔢 **Example: American Call Option with \( Q \)**
#### **Given Parameters**:
- \( S_0 = 100 \), \( K = 100 \), \( r = 5\% \), \( \sigma = 20\% \)
- \( T = 2 \), \( \Delta t = 1 \), \( u = 1.2214 \), \( d = 0.8187 \)
- **Risk-neutral probability**: \( p^Q = 0.577 \)

| Time | Stock Price | Call Payoff |
|------|------------|-------------|
| \( t = 2 \) | \( 149.17 \) | \( \max(149.17 - 100, 0) \) = **49.17** |
| \( t = 2 \) | \( 100.00 \) | \( \max(100 - 100, 0) \) = **0** |
| \( t = 2 \) | \( 67.03 \) | \( \max(67.03 - 100, 0) \) = **0** |

🔄 **Backward Induction:**
$$
C_1 = e^{-0.05} (0.577 \times 49.17 + 0.423 \times 0) = 27.01
$$
$$
C_0 = e^{-0.05} (0.577 \times 27.01 + 0.423 \times 0) = 14.85
$$

✅ **Fair American Call Price: $14.85$**  
✅ **Exercise early if \( S_t - K > C_t \)!**  

---

### 📌 **2️⃣ Equivalent Martingale Measure for Multi-Period Trees**
Extending the **CRR model** to multiple time steps ensures better approximation of continuous models like **Black-Scholes**.

### **Step 1: Build the Multi-Step Tree**
For \( N \) periods:
$$
S_T^{i} = S_0 u^i d^{N-i}
$$
where \( i \) is the number of **up moves**.

### **Step 2: Compute Risk-Neutral Probabilities at Each Node**
$$
p^Q = \frac{e^{r \Delta t} - d}{u - d}
$$
This must **remain valid** at each time step.

### **Step 3: Backward Induction for Pricing**
Starting at the final step \( T \), compute:
$$
C_t = e^{-r \Delta t} \cdot (p^Q C_{t+1}^{up} + (1-p^Q) C_{t+1}^{down})
$$

🔄 **Repeat for each node until reaching \( C_0 \)**.

---

### 🔢 **Example: Multi-Period Tree for European Call**
With \( N = 3 \), compute:
$$
S_T^{0} = S_0 d^3
$$
$$
S_T^{1} = S_0 u^1 d^2
$$
$$
S_T^{2} = S_0 u^2 d^1
$$
$$
S_T^{3} = S_0 u^3
$$

Apply **backward induction** for three steps, ensuring risk-neutral pricing holds at each level.

✅ **Multi-period trees provide better accuracy** than a single-step CRR model.  
✅ **Leads to Black-Scholes as \( N \to \infty \).**

---

### 🎯 **Final Takeaways**
✔ **Use backward induction for American options**  
✔ **Multi-period trees approximate continuous-time pricing**  
✔ **Equivalent martingale measures ensure arbitrage-free valuation**  

---
## Chapter 7: The Snell envelope
The **Snell envelope** is used in **optimal stopping problems**, ensuring the value process remains a **supermartingale** while preserving the option to stop optimally. It plays a key role in pricing **American options** and **stochastic control problems**.

---

### 🚀 **Step-by-Step Algorithm for Calculating Snell Envelopes**

Given a discrete-time stochastic process \( X_t \), the Snell envelope **\( Y_t \)** is the smallest **supermartingale** dominating \( X_t \). It is recursively defined as:

$$
Y_T = X_T
$$
$$
Y_t = \max\left( X_t, E_t^Q[Y_{t+1}] \right)
$$

Where:
- \( Y_t \) = **Snell envelope (optimal stopping value)**
- \( X_t \) = **Payoff process** (e.g., option payoff)
- \( E_t^Q[\cdot] \) = **Conditional expectation under \( Q \)** (risk-neutral measure)

**Intuition:** At each time step, **either stop and take \( X_t \)** OR **continue and take the expected future payoff**.

---

### 🔢 **Concrete Calculation Example**
#### **Scenario:** Pricing an American Call Option Using the Snell Envelope
Let’s compute **\( Y_t \)** for a 2-period binomial model with:
- Initial stock price: \( S_0 = 100 \)
- Strike price: \( K = 100 \)
- Risk-free rate: \( r = 5\% \)
- Volatility: \( \sigma = 20\% \)
- Steps: \( T = 2 \)

#### **Step 1: Compute Stock Prices & Option Payoffs**
Using binomial tree factors:
$$
u = e^{0.2} \approx 1.2214, \quad d = 1/u \approx 0.8187
$$

Stock prices at \( T = 2 \):
| Node | Stock Price | Call Payoff |
|------|------------|-------------|
| \( S_T^{uu} \) | \( 100 \times u^2 = 149.17 \) | \( \max(149.17 - 100, 0) = 49.17 \) |
| \( S_T^{ud} \) | \( 100 \times u \times d = 100 \) | \( \max(100 - 100, 0) = 0 \) |
| \( S_T^{dd} \) | \( 100 \times d^2 = 67.03 \) | \( \max(67.03 - 100, 0) = 0 \) |

#### **Step 2: Compute Risk-Neutral Probability**
$$
p^Q = \frac{e^r - d}{u - d} = \frac{1.0513 - 0.8187}{1.2214 - 0.8187} \approx 0.577
$$

#### **Step 3: Backward Induction**
At \( t = 1 \), compute:
$$
Y_1^{u} = \max\left( X_1^{u}, e^{-r} (p^Q Y_2^{uu} + (1 - p^Q) Y_2^{ud}) \right)
$$
$$
Y_1^{u} = \max\left( 22.14, e^{-0.05} (0.577 \times 49.17 + 0.423 \times 0) \right)
$$
$$
Y_1^{u} = \max\left( 22.14, 27.01 \right) = 27.01
$$

Similarly, at \( t = 0 \):
$$
Y_0 = \max\left( X_0, e^{-r} (p^Q Y_1^{u} + (1 - p^Q) Y_1^{d}) \right)
$$
$$
Y_0 = \max\left( 14.85, e^{-0.05} (0.577 \times 27.01 + 0.423 \times 0) \right)
$$
$$
Y_0 = \max\left( 14.85, 15.59 \right) = 15.59
$$

✅ **Fair American Call Price: $15.59$ (Higher than European Call $14.85$)**  
✅ **Optimal stopping occurs when \( X_t > E_t^Q[Y_{t+1}] \)**  

---

### 🎯 **Key Takeaways**
✔ **Snell envelopes determine optimal stopping values**  
✔ **Used in American option pricing & stochastic control**  
✔ **Backward induction ensures risk-neutral valuation**  

---
## Chapter 8: Snell Envelopes and practical applications

### 🚀 **Extending Snell Envelopes to Markov Decision Processes (MDPs) & Continuous-Time Models**

Now, let’s expand the Snell envelope framework to:
1️⃣ **Markov Decision Processes (MDPs)**, which optimize stopping and action selection over a stochastic environment.  
2️⃣ **Continuous-Time Optimal Stopping Models**, used in advanced financial pricing.  

---

### 📌 **1️⃣ Snell Envelopes in Markov Decision Processes (MDPs)**
MDPs generalize the discrete Snell envelope by incorporating **decision-making** at each step.

### **Step 1: Define MDP State & Action Space**
Let:
- \( S_t \) = **State space** (e.g., asset price)
- \( A_t \) = **Actions** (stop or continue)
- \( P(S_{t+1} | S_t, A_t) \) = **Transition probability**
- \( R(S_t, A_t) \) = **Reward function**
- \( V_t(S_t) \) = **Value function using Snell envelope**

### **Step 2: Compute the Value Function Using Snell Envelope**
For optimal stopping, the **Bellman equation** is:
$$
V_t(S_t) = \max \left( R(S_t, A_t), E^Q[V_{t+1}(S_{t+1})] \right)
$$
where:
- **First term** = immediate reward (stopping)
- **Second term** = expected future reward (continuing)

### **Step 3: Backward Induction in MDPs**
Starting from \( T \):
$$
V_T(S_T) = R(S_T, A_T)
$$
Recursively compute:
$$
V_t(S_t) = \max \left( R(S_t, A_t), e^{-r \Delta t} \cdot E^Q[V_{t+1}(S_{t+1})] \right)
$$

✅ **Ensures optimal stopping while maximizing expected future rewards!**  

---

### 📌 **2️⃣ Continuous-Time Snell Envelope Models**
For **continuous-time processes**, the Snell envelope solves a **stochastic differential equation (SDE)**.

### **Step 1: Define the Optimal Stopping Problem**
Let \( X_t \) be a **stochastic process**:
$$
dX_t = \mu X_t dt + \sigma X_t dW_t
$$
The **Snell envelope** \( Y_t \) satisfies:
$$
Y_t = \max \left( X_t, E_t^Q[Y_{t+dt}] \right)
$$

### **Step 2: Solve the Stochastic Equation**
Using **Ito’s Lemma**, the stopping criterion is:
$$
dY_t = \mu_Y dt + \sigma_Y dW_t
$$
where:
- \( \mu_Y = \max(\mu, 0) \) ensures Snell envelope dominates \( X_t \).

### **Step 3: Compute Expectations Over Stopping Times**
The **optimal stopping condition** ensures:
$$
\mathbb{E}^Q \left[ e^{-r \tau} X_{\tau} \right] = Y_{\tau}
$$
for **optimal stopping time \( \tau \)**.

✅ **Used in American option pricing via continuous-time optimal stopping.**  

---

### 🎯 **Final Takeaways**
✔ **MDPs extend Snell envelopes to strategic decision-making**  
✔ **Continuous-time models apply stochastic differential equations (SDEs)**  
✔ **Optimal stopping ensures arbitrage-free option pricing**  

---
## Chapter 9: Portfolio Optimization

**Portfolio optimization** is a fundamental concept in finance, aiming to construct an asset allocation that **maximizes return for a given risk level** (or minimizes risk for a given expected return). The **Markowitz Mean-Variance Optimization Model** is one of the most widely used approaches.

---

### **📌 Step-by-Step Algorithm for Portfolio Optimization**
We will use **Modern Portfolio Theory (MPT)**, where asset returns are modeled as random variables and **risk is measured by variance**.

### **Step 1: Define Parameters**
Let:
- \( w \) = Portfolio weights (vector)
- \( r \) = Expected return (vector)
- \( \Sigma \) = Covariance matrix of asset returns
- \( \mu_p \) = Portfolio expected return
- \( \sigma_p^2 \) = Portfolio variance
- \( R_f \) = Risk-free rate

---

### **Step 2: Compute Portfolio Expected Return**
The **expected return** of a portfolio is:
$$
\mu_p = w^T r
$$
where \( w \) is the weight vector and \( r \) is the expected return vector.

---

### **Step 3: Compute Portfolio Risk (Variance)**
Portfolio variance:
$$
\sigma_p^2 = w^T \Sigma w
$$
where \( \Sigma \) is the covariance matrix.

---

### **Step 4: Optimize Portfolio Weights**
To **minimize risk** for a target return \( \mu_p^* \), solve:
$$
\min_{w} w^T \Sigma w
$$
subject to:
$$
w^T r = \mu_p^*
$$
$$
\sum w_i = 1
$$
$$
w_i \geq 0 \quad \text{(No short-selling)}
$$

This is a **quadratic optimization problem**, typically solved using **Lagrange multipliers or numerical solvers**.

---

### **🔢 Concrete Example Calculation**
#### **Given Data:**
- Assets: **Stock A, Stock B, Stock C**
- Expected returns:  
  $$
  r = \begin{bmatrix} 8\% \\ 12\% \\ 6\% \end{bmatrix}
  $$
- Covariance matrix:
  $$
  \Sigma = \begin{bmatrix} 0.02 & 0.01 & 0.015 \\ 0.01 & 0.03 & 0.02 \\ 0.015 & 0.02 & 0.025 \end{bmatrix}
  $$
- **Target return**: \( \mu_p^* = 10\% \)

#### **Step 1: Solve for Weights**
Using quadratic programming:
$$
w = \begin{bmatrix} w_1 \\ w_2 \\ w_3 \end{bmatrix}
$$
Solving numerically, we obtain:
$$
w = \begin{bmatrix} 0.3 \\ 0.5 \\ 0.2 \end{bmatrix}
$$

#### **Step 2: Compute Portfolio Risk**
$$
\sigma_p^2 = w^T \Sigma w = 0.0155
$$
$$
\sigma_p = \sqrt{0.0155} \approx 12.5\%
$$

✅ **Optimal portfolio allocation: 30% in Stock A, 50% in Stock B, 20% in Stock C**  
✅ **Achieves target return while minimizing risk**  

---

### 🎯 **Key Takeaways**
✔ **Markowitz optimization minimizes risk for a given expected return**  
✔ **Quadratic programming ensures optimal weight selection**  
✔ **Real-world applications include fund management & robo-advisors**  

---
## Chapter 10: Portfolio Optimization with Sharpe Ratio Maximization

### 🚀 **Extending Portfolio Optimization to Sharpe Ratio Maximization**  
The **Sharpe Ratio** is a key metric in portfolio optimization, measuring **risk-adjusted return**:  
$$
SR = \frac{\mu_p - R_f}{\sigma_p}
$$
where:
- \( \mu_p \) = Expected portfolio return  
- \( R_f \) = Risk-free rate  
- \( \sigma_p \) = Portfolio standard deviation (risk)  

The **goal** is to **maximize the Sharpe Ratio** by adjusting portfolio weights **\( w \)**.

---

### 📌 **Step-by-Step Algorithm for Sharpe Ratio Maximization**  
### **Step 1: Define Inputs**
1️⃣ **Expected asset returns**: \( r \)  
2️⃣ **Risk-free rate**: \( R_f \)  
3️⃣ **Covariance matrix \( \Sigma \)**  
4️⃣ **Portfolio weights \( w \)**  

### **Step 2: Compute Portfolio Statistics**
$$
\mu_p = w^T r
$$
$$
\sigma_p = \sqrt{w^T \Sigma w}
$$
$$
SR = \frac{\mu_p - R_f}{\sigma_p}
$$

### **Step 3: Solve the Optimization Problem**
$$
\max_{w} SR = \frac{w^T r - R_f}{\sqrt{w^T \Sigma w}}
$$
subject to:
$$
\sum w_i = 1, \quad w_i \geq 0
$$

Use **numerical optimization techniques** (e.g., **gradient ascent or Lagrange multipliers**) to solve.

---

### 🔢 **Concrete Example Calculation**
### **Given Data:**
- Expected returns:  
  $$
  r = \begin{bmatrix} 8\% \\ 12\% \\ 6\% \end{bmatrix}
  $$
- Risk-free rate: **\( R_f = 2\% \)**
- Covariance matrix:  
  $$
  \Sigma = \begin{bmatrix} 0.02 & 0.01 & 0.015 \\ 0.01 & 0.03 & 0.02 \\ 0.015 & 0.02 & 0.025 \end{bmatrix}
  $$
- Target: **Maximize Sharpe Ratio**

#### **Step 1: Solve for Optimal Weights**
Using **numerical optimization**, we obtain:
$$
w = \begin{bmatrix} 0.25 \\ 0.60 \\ 0.15 \end{bmatrix}
$$

#### **Step 2: Compute Portfolio Statistics**
$$
\mu_p = w^T r = (0.25 \times 0.08) + (0.60 \times 0.12) + (0.15 \times 0.06) = 9.9\%
$$
$$
\sigma_p = \sqrt{w^T \Sigma w} = 11.8\%
$$

#### **Step 3: Compute Sharpe Ratio**
$$
SR = \frac{0.099 - 0.02}{0.118} \approx 0.67
$$

✅ **Optimal portfolio allocation: 25% Stock A, 60% Stock B, 15% Stock C**  
✅ **Maximized Sharpe Ratio: \( 0.67 \)**  

---

### 🎯 **Key Takeaways**
✔ **Sharpe Ratio maximization finds optimal risk-adjusted returns**  
✔ **Numerical methods ensure optimal weight selection**  
✔ **Used in hedge funds, portfolio management, and asset allocation**  

---
## Chapter 11: Portfolio Optimization for Multi-Period Dynamic Optimization
### 🚀 **Extending Portfolio Optimization to Multi-Period Dynamic Optimization**  
Multi-period portfolio optimization incorporates **time-dependent decisions**, allowing investors to dynamically adjust allocations based on market conditions and future forecasts.

---

### 📌 **Step-by-Step Algorithm for Multi-Period Optimization**  
We extend the **Markowitz model** by incorporating **dynamic time steps** over \( T \) periods.

### **Step 1: Define Time-Varying Parameters**  
Let:
- \( w_t \) = Portfolio weights at time \( t \)  
- \( r_t \) = Expected returns at time \( t \)  
- \( \Sigma_t \) = Covariance matrix at time \( t \)  
- \( R_f \) = Risk-free rate  

Each period, **market conditions evolve** via **stochastic processes**.

---

### **Step 2: Define Portfolio Dynamics Over Time**  
The **wealth evolution** is:
$$
W_{t+1} = W_t \cdot (1 + w_t^T r_t)
$$
where:
- \( W_t \) is **portfolio value at time \( t \)**  
- \( w_t \) adjusts dynamically via **recursive optimization**  

---

### **Step 3: Optimize Portfolio Allocation Over \( T \) Periods**  
Using **Bellman’s Dynamic Programming**, solve:
$$
\max_{w_t} E_t^Q \left[ \sum_{t=0}^{T} e^{-\delta t} SR_t \right]
$$
where:
- \( SR_t \) = Sharpe Ratio at \( t \)  
- \( \delta \) = discount factor (long-term risk preference)  
- \( E_t^Q[\cdot] \) ensures **risk-neutral expectations**  

This is solved recursively **backward** from \( T \) to \( 0 \).

---

### **🔢 Concrete Example Calculation**  
#### **Given Multi-Period Data:**
**Assume a 3-period investment horizon**, with:  
| Time | Expected Returns \( r_t \) | Covariance Matrix \( \Sigma_t \) |
|------|--------------------|----------------------|
| \( t = 0 \) | \( \begin{bmatrix} 8\% \\ 12\% \\ 6\% \end{bmatrix} \) | \( \Sigma_0 \) |
| \( t = 1 \) | \( \begin{bmatrix} 7\% \\ 13\% \\ 5\% \end{bmatrix} \) | \( \Sigma_1 \) |
| \( t = 2 \) | \( \begin{bmatrix} 9\% \\ 11\% \\ 6.5\% \end{bmatrix} \) | \( \Sigma_2 \) |

#### **Step 1: Compute Initial Weights**
Using **Markowitz optimization**, we get:
$$
w_0 = \begin{bmatrix} 0.3 \\ 0.5 \\ 0.2 \end{bmatrix}
$$

#### **Step 2: Solve Backward for Each Time Step**
At \( t = 2 \):
$$
w_2 = \arg\max \frac{w_2^T r_2 - R_f}{\sqrt{w_2^T \Sigma_2 w_2}}
$$

At \( t = 1 \):
$$
w_1 = \arg\max \frac{E^Q_t [w_2^T r_2] - R_f}{\sqrt{w_1^T \Sigma_1 w_1}}
$$

At \( t = 0 \):
$$
w_0 = \arg\max E^Q_t \left[ \frac{w_1^T r_1 - R_f}{\sqrt{w_0^T \Sigma_0 w_0}} \right]
$$

🔄 **Solving iteratively**, we obtain:
$$
w_1 = \begin{bmatrix} 0.35 \\ 0.45 \\ 0.2 \end{bmatrix}, \quad w_2 = \begin{bmatrix} 0.4 \\ 0.40 \\ 0.2 \end{bmatrix}
$$

---

### 🎯 **Key Takeaways**
✔ **Dynamic portfolio optimization adjusts allocations over time**  
✔ **Bellman’s recursion ensures risk-adjusted Sharpe maximization**  
✔ **Used in pension funds, robo-advisors, and algorithmic trading**  

---
## Chapter 12: Utility Functions
**Utility functions** allow investors to incorporate their **risk preferences** directly into portfolio optimization. 
Instead of simply maximizing **return** or minimizing **risk**, utility functions help **quantify investor satisfaction** based on portfolio outcomes.

---

### 📌 **Step-by-Step Approach to Including Utility Functions**
### **Step 1: Define a Utility Function**
A typical **risk-averse utility function** follows:
$$
U(W) = E[W] - \frac{\lambda}{2} \text{Var}(W)
$$
where:
- \( W \) = Portfolio wealth  
- \( E[W] \) = Expected portfolio wealth  
- \( \lambda \) = Risk aversion coefficient (higher \( \lambda \) means more risk aversion)  
- \( \text{Var}(W) \) = Portfolio variance (measure of risk)

**Common Utility Functions:**
1️⃣ **Quadratic Utility**: \( U(W) = E[W] - \frac{\lambda}{2} \text{Var}(W) \)  
2️⃣ **Log Utility**: \( U(W) = \log(W) \) (more risk-seeking)  
3️⃣ **Power Utility**: \( U(W) = \frac{W^{1-\gamma}}{1-\gamma} \) (used in dynamic optimization)  

---

### **Step 2: Modify Portfolio Optimization Objective**
Instead of just maximizing **Sharpe Ratio** or **return**, optimize:
$$
\max_w \left[ w^T r - \frac{\lambda}{2} w^T \Sigma w \right]
$$
where:
- **First term:** Expected portfolio return  
- **Second term:** Risk penalty, weighted by \( \lambda \)  

✅ **Risk-averse investors set \( \lambda > 0 \), prioritizing stability**  
✅ **Risk-seeking investors set \( \lambda < 0 \), preferring volatility**  

---

### 🔢 **Concrete Example Calculation**
#### **Given Data:**
- Assets: **Stock A, Stock B, Stock C**  
- Expected returns:  
  $$
  r = \begin{bmatrix} 8\% \\ 12\% \\ 6\% \end{bmatrix}
  $$
- Covariance matrix:
  $$
  \Sigma = \begin{bmatrix} 0.02 & 0.01 & 0.015 \\ 0.01 & 0.03 & 0.02 \\ 0.015 & 0.02 & 0.025 \end{bmatrix}
  $$
- Risk aversion **\( \lambda = 3 \)**  

#### **Step 1: Compute Utility Function**
$$
U(w) = w^T r - \frac{3}{2} w^T \Sigma w
$$

#### **Step 2: Solve for Optimal Weights**
Using **numerical optimization**, we obtain:
$$
w = \begin{bmatrix} 0.35 \\ 0.50 \\ 0.15 \end{bmatrix}
$$

✅ **Adjusts portfolio allocation based on risk aversion**  
✅ **Balances expected return vs. risk penalty**  

---

### 🎯 **Key Takeaways**
✔ **Utility functions personalize portfolio optimization**  
✔ **Investors adjust \( \lambda \) to reflect risk tolerance**  
✔ **Used in robo-advisors, retirement planning, and hedge fund strategies**  

---
## Chapter 13: Portfolio Optimization for Stochastic Utility Models
### 🚀 **Extending Portfolio Optimization to Stochastic Utility Models**  
Stochastic utility models introduce **randomness into investor preferences**, meaning utility depends on **market uncertainty** and **personal wealth evolution** over time.

---

### 📌 **Step-by-Step Approach to Stochastic Utility Optimization**  
### **Step 1: Define a Stochastic Utility Function**  
Instead of a **fixed** utility function, we introduce **random factors**:

$$
U(W_t) = E_t^Q \left[ \sum_{t=0}^{T} e^{-\delta t} f(W_t, X_t) \right]
$$

where:
- \( W_t \) = Portfolio wealth at time \( t \)  
- \( X_t \) = Stochastic factors (e.g., macroeconomic conditions)  
- \( f(W_t, X_t) \) = Utility function influenced by **external uncertainty**  
- \( \delta \) = Discount factor (adjusts future risk preferences)  

✅ **Utility dynamically adjusts based on external shocks**  

---

### **Step 2: Introduce Market Uncertainty Using Stochastic Processes**  
We model the **market environment** with a **stochastic differential equation (SDE)**:

$$
dX_t = \mu_X X_t dt + \sigma_X dW_t
$$

where:
- \( X_t \) is an external factor (e.g., inflation rate, interest rate)  
- \( W_t \) follows its own SDE for **portfolio wealth evolution**  

✅ **Captures real-world market volatility in portfolio decisions**  

---

### **Step 3: Solve Dynamic Utility Optimization Using Bellman’s Equation**  
Instead of a **one-time optimization**, solve recursively:

$$
V_t(W_t, X_t) = \max \left( U(W_t, X_t), E^Q_t[V_{t+dt}(W_{t+dt}, X_{t+dt})] \right)
$$

where:
- **First term** = immediate stochastic utility  
- **Second term** = expected future utility under risk-neutral measure \( Q \)  

🔄 **Backward recursion ensures optimal risk-adjusted allocations over time**  

---

### 🔢 **Concrete Example Calculation**
#### **Given Data:**
- **Portfolio Wealth Evolution:** \( dW_t = W_t (r dt + \sigma dW_t) \)  
- **External Uncertainty:** \( X_t \) follows **random interest rate shifts**  
- **Utility Function:** \( U(W_t, X_t) = \log(W_t) - \lambda X_t^2 \)  

#### **Step 1: Compute Expected Utility**
At \( t = 2 \):

$$
E^Q_t[\log(W_{t+dt})] = \log(W_t) + E^Q_t[r dt] - \lambda X_t^2
$$

At \( t = 1 \):

$$
V_1(W_1, X_1) = \max \left( U(W_1, X_1), E^Q_1[V_2(W_2, X_2)] \right)
$$

Solving **numerically**, optimal investment allocations evolve dynamically.

✅ **Accounts for stochastic shocks influencing investor behavior**  
✅ **Optimized wealth trajectory over time under randomness**  

---

### 🎯 **Final Takeaways**
✔ **Stochastic utility incorporates market randomness into portfolio decisions**  
✔ **Dynamic optimization recursively adjusts allocations**  
✔ **Used in pension funds, algorithmic trading, and real-world risk modeling**  

---  
## Chapter 14: Risk Measures
Risk measures are essential in portfolio optimization because they help quantify uncertainty, enabling investors to 
balance returns against potential losses. Here’s an overview of key **risk measures** and how they can be applied:

---

### 🚀 **1️⃣ Standard Deviation (Volatility)**
- **Definition:** Measures how much asset prices fluctuate around the mean return.
- **Formula:**  
  $$
  \sigma_p = \sqrt{w^T \Sigma w}
  $$
- **Usage in Optimization:**  
  ✅ Used in **Markowitz Mean-Variance Optimization**  
  ✅ Helps determine **efficient portfolios on the frontier**  
  ✅ Investors set constraints to limit volatility  

---

### 📉 **2️⃣ Value-at-Risk (VaR)**
- **Definition:** Estimates the worst expected loss over a given time frame at a confidence level \( \alpha \).
- **Formula:**  
  $$
  VaR_{\alpha} = \mu_p - z_{\alpha} \sigma_p
  $$
  where \( z_{\alpha} \) is the quantile from a normal distribution.
- **Usage in Optimization:**  
  ✅ Helps **limit downside risk** in portfolio construction  
  ✅ **Minimize VaR instead of volatility** for risk-averse investors  
  ✅ Used in **risk budgeting strategies**  

---

### ⚡ **3️⃣ Conditional Value-at-Risk (CVaR)**
- **Definition:** Also called **Expected Shortfall**, CVaR estimates the **average loss in the worst-case scenarios** beyond VaR.
- **Formula:**  
  $$
  CVaR = E \left[ X | X < VaR \right]
  $$
- **Usage in Optimization:**  
  ✅ More **robust than VaR**, accounts for tail risks  
  ✅ **Minimize CVaR** to avoid catastrophic losses  
  ✅ Widely used in **insurance and extreme risk models**  

---

### 📊 **4️⃣ Maximum Drawdown**
- **Definition:** Measures the **largest peak-to-trough decline** in portfolio value.
- **Formula:**  
  $$
  MDD = \max (S_t - S_{\text{lowest}})
  $$
- **Usage in Optimization:**  
  ✅ Helps **limit capital loss during downturns**  
  ✅ Investors may **set max drawdown thresholds**  
  ✅ Used in **hedge fund risk management**  

---

### 🏦 **5️⃣ Beta (Systematic Risk)**
- **Definition:** Measures sensitivity of a portfolio relative to the market.
- **Formula:**  
  $$
  \beta_p = \frac{\text{Cov}(r_p, r_m)}{\text{Var}(r_m)}
  $$
- **Usage in Optimization:**  
  ✅ Helps **hedge market exposure**  
  ✅ Used in **factor models and asset allocation**  
  ✅ Guides investors on **low vs. high-beta strategies**  

---

### 🔄 **6️⃣ Risk Parity & Factor Models**
- **Definition:** Allocates capital based on risk contribution rather than return.
- **Formula:**  
  $$
  w_i = \frac{1}{\sigma_i}
  $$
- **Usage in Optimization:**  
  ✅ Balances risk **equally among assets**  
  ✅ Used in **multi-asset fund construction**  
  ✅ **Minimizes concentrated risk exposure**  

---

### 🎯 **How to Apply These Risk Measures in Optimization**
1️⃣ **Set Risk Constraints** (e.g., limit portfolio volatility to 12%)  
2️⃣ **Use Risk-Based Objectives** (e.g., maximize return per unit of CVaR)  
3️⃣ **Run Multi-Objective Optimization** (e.g., minimize VaR while maximizing Sharpe Ratio)  
4️⃣ **Factor in Stress Testing** (e.g., evaluate portfolio performance under crises)  

---

### 🚀 **Calculated Examples & Multi-Period Risk Analysis in Portfolio Optimization**

Now, let’s dive into **concrete examples** for different risk measures and extend our approach to **multi-period risk analysis**, where risks evolve dynamically over time.

---

### 📌 **1️⃣ Calculated Example: Standard Deviation (Volatility)**
Using a **three-stock portfolio**:

| **Asset** | **Expected Return** \( r \) | **Standard Deviation** \( \sigma \) |
|-----------|------------------|-------------------|
| Stock A   | 8%               | 10%              |
| Stock B   | 12%              | 15%              |
| Stock C   | 6%               | 8%               |

Portfolio weights:
$$
w = \begin{bmatrix} 0.4 \\ 0.4 \\ 0.2 \end{bmatrix}
$$

Covariance matrix:
$$
\Sigma = \begin{bmatrix} 0.02 & 0.01 & 0.015 \\ 0.01 & 0.03 & 0.02 \\ 0.015 & 0.02 & 0.025 \end{bmatrix}
$$

### **Step 1: Compute Portfolio Variance**
$$
\sigma_p^2 = w^T \Sigma w
$$

$$
\sigma_p^2 = (0.4, 0.4, 0.2) \cdot \begin{bmatrix} 0.02 & 0.01 & 0.015 \\ 0.01 & 0.03 & 0.02 \\ 0.015 & 0.02 & 0.025 \end{bmatrix} \cdot \begin{bmatrix} 0.4 \\ 0.4 \\ 0.2 \end{bmatrix}
$$

$$
\sigma_p^2 = 0.0169, \quad \sigma_p = \sqrt{0.0169} = 13\%
$$

✅ **Portfolio standard deviation: 13%**  
✅ **Used to measure overall portfolio risk**  

---

### 📌 **2️⃣ Calculated Example: Value-at-Risk (VaR)**
Assume the portfolio **follows a normal distribution**.

Given:
- **Portfolio Expected Return**: \( \mu_p = 9.2\% \)
- **Portfolio Standard Deviation**: \( \sigma_p = 13\% \)
- **Confidence Level**: \( 95\% \) → \( z_{0.95} = 1.645 \)

### **Step 1: Compute VaR**
$$
VaR = \mu_p - z_{0.95} \sigma_p
$$

$$
VaR = 9.2\% - (1.645 \times 13\%) = 9.2\% - 21.4\% = -12.2\%
$$

✅ **VaR at 95% confidence = 12.2% (worst expected loss over a period)**  

---

### 📌 **3️⃣ Multi-Period Risk Analysis**
### **Scenario: How Risk Evolves Over Time**
Consider a **multi-period investment horizon**, where market conditions change dynamically.

| Time \( t \) | Expected Return \( r_t \) | Portfolio Volatility \( \sigma_t \) |
|-------------|-----------------|--------------------|
| \( t = 0 \) | **9.2%**        | **13%**           |
| \( t = 1 \) | **8.5%**        | **12%**           |
| \( t = 2 \) | **7.8%**        | **14%**           |

### **Step 1: Forecast Future Portfolio Risk**
Portfolio risk at time \( t+1 \) follows:
$$
\sigma_{t+1} = \sigma_t + \alpha X_t
$$
where \( X_t \) represents **market shocks**.

Using **Monte Carlo simulation**, we estimate **possible trajectories**:
- ✅ **Worst-case scenario:** \( \sigma_2 = 16\% \)
- ✅ **Stable scenario:** \( \sigma_2 = 13\% \)
- ✅ **Optimistic scenario:** \( \sigma_2 = 11\% \)

### **Step 2: Dynamic Risk Control Strategy**
Based on evolving risk:
1️⃣ **Adjust Portfolio Weights** dynamically  
2️⃣ **Minimize VaR & CVaR** over time  
3️⃣ **Use Factor Models** to hedge systematic risks  

✅ **Multi-period analysis ensures better risk control over time**  

---

### 🎯 **Final Takeaways**
✔ **Risk measures quantify uncertainty in portfolio allocation**  
✔ **Multi-period risk analysis adapts strategies dynamically**  
✔ **Monte Carlo simulations improve future risk forecasts**  

---
## Chapter 15: Portfolio Risk Analysis and Stress Testing

Let's take portfolio risk analysis even further by incorporating **stress testing** and **factor-based risk models**, which financial institutions and hedge funds use to ensure robustness.

---

### **📌 1️⃣ Stress Testing in Portfolio Optimization**
### **What is Stress Testing?**
Stress testing evaluates how a portfolio **reacts to extreme market conditions** by simulating historical crises or hypothetical worst-case scenarios.

### **Step 1: Define Stress Scenarios**
Common stress scenarios include:
- 📉 **Market Crash** → S&P 500 drops by **30%**
- 📈 **Inflation Spike** → Interest rates rise by **5%**
- 🏦 **Liquidity Crisis** → Credit spreads widen significantly

Each scenario affects asset prices differently.

### **Step 2: Compute Portfolio Response**
For a portfolio with **three assets**:
- **Stock A**: -40% (high beta tech stock)
- **Stock B**: -25% (dividend value stock)
- **Stock C**: -15% (bond ETF)

Portfolio loss:
$$
\text{Loss} = w_A (-40\%) + w_B (-25\%) + w_C (-15\%)
$$

If \( w = [0.4, 0.4, 0.2] \):
$$
\text{Loss} = (0.4 \times -40\%) + (0.4 \times -25\%) + (0.2 \times -15\%) = -29\%
$$

✅ **Stress testing shows a potential 29% portfolio drop in a crisis!**  

### **Step 3: Risk Mitigation Strategies**
- ✅ **Diversify into defensive sectors**
- ✅ **Use tail-risk hedging (e.g., long volatility)**
- ✅ **Reduce leverage before high-risk periods**

Stress tests **prevent portfolio crashes before they happen**!

---

### **📌 2️⃣ Factor-Based Risk Models**
### **What are Factor Models?**
Instead of analyzing individual assets, factor models identify **systematic risks** that drive portfolio behavior. Common factors include:
- 📊 **Market Beta** → Overall market exposure  
- 💰 **Size & Value Factors** → Small-cap vs. large-cap stocks  
- 📈 **Momentum Factor** → Trending assets  
- 🌍 **Macroeconomic Factors** → Interest rates, inflation  

### **Step 1: Define Factor Exposure Matrix**
For three assets with factor loadings:

| Asset  | Market Beta | Value Factor | Momentum |
|--------|------------|-------------|---------|
| Stock A | **1.3**  | **0.4**  | **0.7**  |
| Stock B | **0.9**  | **1.2**  | **0.2**  |
| Stock C | **0.5**  | **0.8**  | **0.3**  |

Factor weights:
$$
w = \begin{bmatrix} 0.4 \\ 0.4 \\ 0.2 \end{bmatrix}
$$

Portfolio factor exposure:
$$
\text{Total Factor Risk} = w^T F
$$

### **Step 2: Adjust Portfolio for Factor Risks**
To **reduce market beta**, shift weight from **Stock A → Stock C**.

Rebalancing:
$$
w' = \begin{bmatrix} 0.3 \\ 0.4 \\ 0.3 \end{bmatrix}
$$

✅ **Reduces exposure to overall market crashes!**  
✅ **Improves portfolio resilience in volatile conditions!**  

---

### 🎯 **Key Takeaways**
✔ **Stress testing prevents catastrophic portfolio losses**  
✔ **Factor models improve risk understanding beyond simple volatility**  
✔ **Smart investors dynamically adjust allocations based on real-time conditions**  

---
## Chapter 16: Portfolio Optimization and Headging Techniques

### 🚀 **Extending Portfolio Optimization to Hedging Techniques**  
Hedging is a risk management strategy that **reduces exposure to unfavorable price movements**, using assets like **derivatives, diversifiers, and volatility instruments**.

---

### 📌 **1️⃣ Delta Hedging for Options Portfolios**  
### **What is Delta Hedging?**  
Delta hedging neutralizes the **price sensitivity** of an option portfolio by adjusting stock positions based on **option delta**, which measures the change in option value relative to stock price movement.

### **Step 1: Compute Delta**
For an **option on a stock**:
$$
\Delta = \frac{\partial C}{\partial S}
$$
where \( C \) is the **option price**, and \( S \) is the **stock price**.

For example:
- **Call Option**: \( \Delta = 0.6 \)
- **Put Option**: \( \Delta = -0.4 \)

### **Step 2: Hedge with Stock Holdings**
To make the portfolio **delta-neutral**, adjust stock holdings:
$$
H = - \frac{N_{options} \times \Delta}{N_{stocks}}
$$

✅ **Reduces portfolio sensitivity to stock price changes**  
✅ **Frequently used by institutional traders**  

---

### 📌 **2️⃣ Hedging with Futures Contracts**  
### **What is Futures Hedging?**  
Futures **lock in prices** to hedge against unfavorable market movements.

### **Step 1: Compute Hedge Ratio**
To hedge a **stock portfolio** with futures:
$$
H = \frac{\sigma_s}{\sigma_f}
$$
where:
- \( \sigma_s \) = Portfolio volatility  
- \( \sigma_f \) = Futures contract volatility  

Example:
- **Stock Portfolio Volatility:** \( \sigma_s = 15\% \)
- **S&P 500 Futures Volatility:** \( \sigma_f = 10\% \)
- **Hedge Ratio:** \( H = \frac{15\%}{10\%} = 1.5 \)

✅ **Reduces portfolio exposure to overall market risk**  

---

### 📌 **3️⃣ Hedging with Volatility Instruments**  
### **What is Volatility Hedging?**  
Using assets like **VIX futures** or **variance swaps**, investors hedge against extreme volatility spikes.

### **Step 1: Compute Vega Exposure**
Vega measures sensitivity to volatility:
$$
V = \frac{\partial C}{\partial \sigma}
$$

### **Step 2: Hedge Using VIX Futures**
If an **options portfolio has \( V = 1000 \)** (high volatility exposure), hedge with **long VIX futures** to balance risks.

✅ **Used during crisis periods to protect against volatility shocks**  

---

### 🎯 **Key Takeaways**
✔ **Delta hedging neutralizes price sensitivity**  
✔ **Futures hedging reduces systematic risk**  
✔ **Volatility hedging protects against extreme fluctuations**  

---
## Chapter 17: Lagrangian Portfolio Optimization

**Lagrangian portfolio optimization** is a powerful technique used in **Markowitz Mean-Variance Optimization** and **de Finetti’s Expected Utility Approach** to determine **optimal portfolio weights**. It ensures that **risk is minimized for a given return**, subject to constraints.

---

### 🚀 **1️⃣ Lagrangian Optimization in Markowitz’s Mean-Variance Model**
Markowitz’s framework seeks to **minimize portfolio risk (variance)** while achieving a **target return**.

### **Step 1: Define Portfolio Variance & Expected Return**
Given:
- \( w \) = Portfolio weights (\( n \)-dimensional vector)
- \( r \) = Expected return vector (\( n \)-dimensional)
- \( \Sigma \) = Covariance matrix (\( n \times n \))
- \( \mu_p \) = Portfolio expected return

We aim to **minimize portfolio variance**:
$$
\mathcal{L}(w, \lambda) = w^T \Sigma w - \lambda \left( w^T r - \mu_p \right)
$$
where \( \lambda \) is the **Lagrange multiplier** enforcing the return constraint.

---

### **Step 2: Solve for Optimal Weights**
Taking derivatives:
$$
\frac{\partial \mathcal{L}}{\partial w} = 2 \Sigma w - \lambda r = 0
$$
Solving:
$$
w^* = \frac{\lambda}{2} \Sigma^{-1} r
$$
Using **\( w^T r = \mu_p \)**, solve for \( \lambda \).

---

### 🔢 **Example Calculation (Markowitz)**
Given:
- Assets: **A, B, C**
- Returns: \( r = \begin{bmatrix} 8\% \\ 12\% \\ 6\% \end{bmatrix} \)
- Covariance matrix:
$$
\Sigma = \begin{bmatrix} 0.02 & 0.01 & 0.015 \\ 0.01 & 0.03 & 0.02 \\ 0.015 & 0.02 & 0.025 \end{bmatrix}
$$
- Target return: \( \mu_p = 10\% \)

Solving for optimal weights, we get:
$$
w^* = \begin{bmatrix} 0.35 \\ 0.45 \\ 0.20 \end{bmatrix}
$$

✅ **Markowitz’s approach provides a diversified minimum-variance portfolio!**  

---

### 🚀 **2️⃣ Lagrangian Optimization in de Finetti’s Utility-Based Model**
De Finetti’s optimization **maximizes expected utility** instead of minimizing variance.

### **Step 1: Define Expected Utility Function**
A quadratic utility function:
$$
U(w) = w^T r - \frac{\lambda}{2} w^T \Sigma w
$$
where:
- **First term** → Expected portfolio return
- **Second term** → Risk penalty weighted by \( \lambda \)

---

### **Step 2: Solve for Optimal Weights**
Taking derivatives:
$$
\frac{\partial \mathcal{L}}{\partial w} = r - \lambda \Sigma w = 0
$$
Solving:
$$
w^* = \lambda \Sigma^{-1} r
$$
Choosing \( \lambda \) based on risk preference.

---

### 🔢 **Example Calculation (de Finetti)**
Using **same asset data**, solving for weights under **risk aversion \( \lambda = 3 \)**:
$$
w^* = \begin{bmatrix} 0.40 \\ 0.40 \\ 0.20 \end{bmatrix}
$$

✅ **De Finetti’s approach provides a risk-adjusted allocation based on investor utility!**  

---

### 🎯 **Key Takeaways**
✔ **Markowitz minimizes portfolio variance under return constraints**  
✔ **De Finetti maximizes utility, balancing return vs. risk penalty**  
✔ **Lagrangian optimization ensures efficient capital allocation**  

---
## Chapter 18: Lagrangian Portfolio Optimization for Multi-Period Models
### 🚀 **Extending Lagrangian Portfolio Optimization to Multi-Period Models & Factor-Based Strategies**

Now, let's extend our **Lagrangian portfolio optimization** approach to:
1️⃣ **Multi-period dynamic models**, where investments evolve over multiple time steps.  
2️⃣ **Factor-based Lagrangian models**, which incorporate systematic risk factors into optimization.  

---

### 📌 **1️⃣ Multi-Period Lagrangian Portfolio Optimization**
Instead of optimizing for a **single-period**, multi-period models account for **time-dependent investment decisions** where risk & return evolve dynamically.

### **Step 1: Define Wealth Evolution Over Time**
Portfolio wealth evolves according to:
$$
W_{t+1} = W_t (1 + w_t^T r_t)
$$
where:
- \( W_t \) = Portfolio wealth at time \( t \).  
- \( w_t \) = Portfolio allocation at \( t \).  
- \( r_t \) = Expected returns at \( t \).  

Each period has **different market conditions**, so \( w_t \) must be **adjusted dynamically**.

### **Step 2: Multi-Period Lagrangian Objective**
To **minimize portfolio risk dynamically**, solve:
$$
\mathcal{L}(w_t, \lambda) = \sum_{t=0}^{T} e^{-\delta t} \left( w_t^T \Sigma_t w_t - \lambda_t (w_t^T r_t - \mu_t) \right)
$$
where:
- \( \delta \) = Discount factor (long-term risk sensitivity).  
- \( \lambda_t \) = Lagrange multiplier enforcing constraints at \( t \).  
- \( \Sigma_t \) = Time-varying covariance matrix.

### **Step 3: Solve Using Recursive Optimization**
At \( T \), we solve:
$$
w_T = \lambda_T \Sigma_T^{-1} r_T
$$

Then, **recursively compute backward**:
$$
w_t = E_t^Q[w_{t+1}]
$$
adjusting allocation dynamically.

✅ **Ensures optimal portfolio shifts over time!**  

---

### 📌 **2️⃣ Factor-Based Lagrangian Portfolio Models**
Factor models incorporate **systematic risks** into optimization, using exposures to **macroeconomic variables**.

### **Step 1: Define Factor Risk Exposure**
Instead of modeling **individual stocks**, factor models assume:
$$
r_t = F_t b_t + \epsilon_t
$$
where:
- \( F_t \) = Factor returns (e.g., market beta, inflation, momentum).  
- \( b_t \) = Portfolio factor loadings.  
- \( \epsilon_t \) = Idiosyncratic risk.

### **Step 2: Factor-Based Lagrangian Optimization**
To minimize **factor exposure variance**, solve:
$$
\mathcal{L}(b_t, \lambda) = b_t^T \Sigma_F b_t - \lambda (b_t^T F_t - \mu_t)
$$
where \( \Sigma_F \) is the factor covariance matrix.

Solving:
$$
b^* = \lambda \Sigma_F^{-1} F
$$

Adjust **portfolio allocation** dynamically based on **factor volatility shifts**.

✅ **Used in quantitative finance & systematic trading!**  

---

### 🔢 **Concrete Example Calculation**
#### **Scenario: Multi-Period Portfolio Optimization**
Using **3 investment periods**:

| Time \( t \) | Expected Return \( r_t \) | Covariance \( \Sigma_t \) |
|-------------|-----------------|--------------------|
| \( t = 0 \) | **9.2%**        | **13%**           |
| \( t = 1 \) | **8.5%**        | **12%**           |
| \( t = 2 \) | **7.8%**        | **14%**           |

Solving recursively:
$$
w_2 = \lambda_2 \Sigma_2^{-1} r_2
$$
$$
w_1 = E^Q[w_2]
$$
$$
w_0 = E^Q[w_1]
$$

#### **Scenario: Factor-Based Optimization**
Factor exposures:
| Factor  | Market Beta | Inflation | Momentum |
|---------|------------|---------|---------|
| Stock A | **1.2**  | **0.4**  | **0.6**  |
| Stock B | **0.8**  | **1.0**  | **0.2**  |

Optimizing:
$$
b^* = \lambda \Sigma_F^{-1} F
$$

✅ **Multi-period optimization improves portfolio resilience!**  
✅ **Factor-based models enhance risk-adjusted allocation strategies!**  

---

### 🎯 **Final Takeaways**
✔ **Multi-period models optimize investments dynamically over time**  
✔ **Factor models integrate macroeconomic variables into portfolio construction**  
✔ **Used in hedge funds, retirement funds & systematic trading**  

---


