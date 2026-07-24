# RESEARCH INTERNSHIP PROJECT REPORT

---

### **Project Title:** Surrogate Modeling in Statistical Learning: Theoretical Foundations, Algorithm Engineering, and Real-World Benchmark Applications of Gaussian Process Regression  
**Internship Period:** May 25, 2026 – July 5, 2026  
**Student Name:** Prateek Raj (Roll No.: BT23CS021)  
**Supervisor / Guide:** Prof. Neelesh Shankar Upadhye  
**Department of Mathematics, Indian Institute of Technology Madras**  
**Repository:** [https://github.com/Prateek312413/IITM_Internship](https://github.com/Prateek312413/IITM_Internship)  

---

## 📌 EXECUTIVE SUMMARY

During the 6-week research internship from **May 25, 2026, to July 5, 2026**, under the supervision of **Prof. Neelesh Shankar Upadhye**, I conducted an in-depth investigation into non-parametric Bayesian machine learning, specifically focusing on **Gaussian Process Regression (GPR)**. 

The primary objectives of this research project were:
1. **Mathematical Foundations & Theoretical Study:** Deep exploration of non-parametric surrogate modeling, weight-space vs. function-space views, inductive bias, likelihood formulations, spectral properties, and kernel design based on *Gaussian Processes for Machine Learning (GPML)* by Rasmussen & Williams (Chapters 1, 2, and 4).
2. **Algorithm Engineering (From-Scratch Implementation):** Building robust numerical implementations of GPR and baseline algorithms from first principles using **raw NumPy and SciPy**, incorporating **Cholesky matrix decomposition ($LL^T = K + \sigma_n^2 I$)** for numerical stability and matrix solves.
3. **Benchmarking & Validation:** Comparing custom implementations against industry-standard **Scikit-Learn** implementations across 6 diverse real-world engineering and environmental datasets (12 Jupyter Notebooks in total).
4. **Research Communication:** Creating formal LaTeX Beamer presentation slide decks for academic presentation and documentation.

---

## 1. THEORETICAL FOUNDATIONS & MATHEMATICAL FORMULATIONS

### 1.1 Supervised Learning & Inductive Bias
In supervised learning, given a dataset $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ with $\mathbf{x}_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$, the goal is to infer the underlying function $y = f(\mathbf{x}) + \epsilon$. 

* **The Under-determination Problem:** Training data alone is insufficient to uniquely identify a function; infinitely many curves pass through any finite set of points.
* **Restriction Bias (Parametric View):** Constrains the solution space to a fixed, finite-dimensional parameter family $f(\mathbf{x}) = \mathbf{x}^T \mathbf{w}$. This introduces rigid structural assumptions leading to under-fitting on non-linear data.
* **Preference Bias (Bayesian View):** Places a probability distribution over an infinite class of functions, preferring smoother functions while remaining capable of fitting complex structures.

### 1.2 Weight-Space Bayesian Linear Regression
Placing a zero-mean Gaussian prior on the weight vector $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_p)$ and assuming i.i.d. Gaussian measurement noise $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$, the likelihood for design matrix $X \in \mathbb{R}^{n \times d}$ and target vector $\mathbf{y} \in \mathbb{R}^n$ is:
$$p(\mathbf{y} \mid X, \mathbf{w}) = \mathcal{N}(X\mathbf{w}, \sigma_n^2 I)$$

By conjugacy and completing the square in the exponent of $p(\mathbf{w} \mid X, \mathbf{y}) \propto p(\mathbf{y} \mid X, \mathbf{w}) p(\mathbf{w})$, the posterior distribution over weights is derived as:
$$p(\mathbf{w} \mid X, \mathbf{y}) = \mathcal{N}(\boldsymbol{\mu}_{\text{post}}, \boldsymbol{\Sigma}_{\text{post}})$$
$$\boldsymbol{\Sigma}_{\text{post}} = \left(\boldsymbol{\Sigma}_p^{-1} + \frac{1}{\sigma_n^2} X^T X\right)^{-1}, \qquad \boldsymbol{\mu}_{\text{post}} = \frac{1}{\sigma_n^2} \boldsymbol{\Sigma}_{\text{post}} X^T \mathbf{y}$$

The predictive distribution for a new input $\mathbf{x}_*$ integrates over all posterior weight configurations:
$$p(y_* \mid \mathbf{x}_*, \mathcal{D}) = \mathcal{N}\left(\frac{1}{\sigma_n^2} \mathbf{x}_*^T \boldsymbol{\Sigma}_{\text{post}} X^T \mathbf{y}, \; \mathbf{x}_*^T \boldsymbol{\Sigma}_{\text{post}} \mathbf{x}_* + \sigma_n^2\right)$$

### 1.3 Feature Projections & The Dimension Bottleneck
To model non-linear relationships, inputs are mapped into an $N$-dimensional feature space $\boldsymbol{\phi}(\mathbf{x}) \in \mathbb{R}^N$. However, evaluating $\boldsymbol{\Sigma}_{\text{post}}$ requires inverting an $N \times N$ matrix $\left(\boldsymbol{\Sigma}_p^{-1} + \frac{1}{\sigma_n^2} \Phi^T \Phi\right)^{-1}$, which becomes computationally intractable as $N \to \infty$.

**Resolution via Woodbury Matrix Identity:**
Applying $(B + U C V)^{-1} = B^{-1} - B^{-1} U (C^{-1} + V B^{-1} U)^{-1} V B^{-1}$, the posterior covariance is rewritten as:
$$\boldsymbol{\Sigma}_{\text{post}} = \boldsymbol{\Sigma}_p - \boldsymbol{\Sigma}_p \Phi^T \left(\Phi \boldsymbol{\Sigma}_p \Phi^T + \sigma_n^2 I\right)^{-1} \Phi \boldsymbol{\Sigma}_p$$

This reduces the inversion from feature dimension **$N \times N$** to data dimension **$n \times n$**, yielding the **Kernel Trick** $k(\mathbf{x}, \mathbf{x}') = \boldsymbol{\phi}(\mathbf{x})^T \boldsymbol{\Sigma}_p \boldsymbol{\phi}(\mathbf{x}')$ and transitioning directly to the **Function-Space View**.

### 1.4 Covariance Functions & Path Properties (GPML Chapter 4)
A Gaussian Process $f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$ is defined by its mean function $m(\mathbf{x})$ and positive semidefinite (PSD) covariance function $k(\mathbf{x}, \mathbf{x}')$:
* **Bochner's Theorem:** A stationary kernel $k(\boldsymbol{\tau})$ is PSD iff it is the Fourier transform of a positive spectral density $S(\mathbf{s}) \ge 0$.
* **Differentiability:** A GP is $m$-times mean-square differentiable iff the mixed partial derivative $\frac{\partial^{2m} k(\mathbf{x}, \mathbf{x}')}{\partial x_i^m \partial {x'_i}^m}$ exists and is finite at $\mathbf{x} = \mathbf{x}'$.
* **Kernel Families Investigated:**
  * *Squared Exponential (RBF):* $k_{\text{SE}}(r) = \sigma_f^2 \exp\left(-\frac{r^2}{2\ell^2}\right)$ (Infinitely differentiable, ultra-smooth).
  * *Matérn Class:* $k_{\text{Matérn}}(r) = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}r}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}r}{\ell}\right)$ ($\nu=1/2$ continuous OU process, $\nu=3/2$ once differentiable, $\nu=5/2$ twice differentiable).
  * *Rational Quadratic (RQ):* $k_{\text{RQ}}(r) = \sigma_f^2 \left(1 + \frac{r^2}{2\alpha \ell^2}\right)^{-\alpha}$ (Infinite scale-mixture of RBF kernels).

---

## 2. ALGORITHM ENGINEERING & IMPLEMENTATION DETAILS

To ensure absolute computational rigor, two parallel codebase frameworks were engineered across **12 Jupyter Notebooks**:

1. **Custom Framework (From Scratch in NumPy/SciPy):**
   * **Cholesky Factorization:** Computed $L = \text{cholesky}(K(X,X) + \sigma_n^2 I)$ for stable inversion.
   * **Linear Solves:** Evaluated predictive weights $\boldsymbol{\alpha} = L^{-T} (L^{-1} \mathbf{y})$ and variance reduction via forward/backward substitution (`scipy.linalg.solve_triangular`).
   * **Baseline Regressors From Scratch:**
     - `LinearRegressionFromScratch`: Normal equations with regularization $\mathbf{w} = (X^T X + \epsilon I)^{-1} X^T \mathbf{y}$.
     - `PolynomialRegressionFromScratch`: Degree-2 basis feature expansion.
     - `KNNRegressorFromScratch`: Distance-weighted k-nearest neighbors search.
     - `DecisionTreeRegressorFromScratch`: Recursive binary splits using variance reduction.
2. **Scikit-Learn Benchmarking Framework:**
   * Utilized `sklearn.gaussian_process.GaussianProcessRegressor` with L-BFGS-B hyperparameter optimization.
   * Benchmarked baseline models (`LinearRegression`, `DecisionTreeRegressor`, `KNeighborsRegressor`).

---

## 3. BENCHMARK APPLICATIONS & EXPERIMENTAL RESULTS

The models were evaluated across 6 diverse real-world predictive domains. Custom NumPy implementations achieved **mathematically identical results** to Scikit-Learn.

### 3.1 NASA Battery State-of-Health (SOH) Capacity Degradation
* **Dataset:** NASA Lithium-Ion Battery Aging Dataset (B0005).
* **Experimental Setup:** 168 total discharge cycles (117 training cycles, 51 test cycles).
* **Task:** Predict capacity degradation (Ah) over operational cycles.
* **Results:**
  - **GPR (Matérn 5/2):** **RMSE = 0.010081 | Accuracy = 98.872%** (Top Performing)
  - **GPR (Linear):** RMSE = 0.016611 | Accuracy = 98.300%
  - **Linear Regression:** RMSE = 0.016612 | Accuracy = 98.300%
  - **Decision Tree:** RMSE = 0.042997 | Accuracy = 94.671%
  - **KNN Regressor:** RMSE = 0.045477 | Accuracy = 94.310%

### 3.2 California Housing Value Prediction
* **Dataset:** StatLib California Housing Census Dataset (20,640 total samples, 8 features).
* **Experimental Setup:** 80/20 train/test split (**16,512 training samples**, **4,128 test samples**).
* **Task:** Predict median house values across block groups.
* **Results:**
  - **GPR (Rational Quadratic):** **RMSE = 0.552962 | Accuracy = 78.907%** (Top Performing)
  - **GPR (Matérn 5/2):** RMSE = 0.554475 | Accuracy = 78.916%
  - **GPR (RBF):** RMSE = 0.561079 | Accuracy = 78.333%
  - **K-Nearest Neighbors:** RMSE = 0.652351 | Accuracy = 76.262%
  - **Decision Tree:** RMSE = 0.733951 | Accuracy = 67.702%

### 3.3 Delhi Air Quality Index (AQI) Forecasting
* **Dataset:** Central Pollution Control Board (CPCB) India air quality database.
* **Experimental Setup:** Cleaned and sampled to **16,000 hourly entries** (71 features), 90/10 train/test split (**14,400 training samples**, **1,600 test samples**).
* **Task:** Forecast multi-pollutant Air Quality Index.
* **Results:**
  - **GPR (Matérn 3/2):** **RMSE = 37.366086 | Accuracy = 88.055%** (Top Performing)
  - **GPR (Matérn 5/2):** RMSE = 37.905587 | Accuracy = 87.811%
  - **GPR (Rational Quadratic):** RMSE = 39.392439 | Accuracy = 87.340%
  - **K-Nearest Neighbors:** RMSE = 42.944977 | Accuracy = 87.034%
  - **Linear Regression:** RMSE = 57.084957 | Accuracy = 80.752%

### 3.4 Residential Building Energy Efficiency (ENB2012)
* **Dataset:** UCI Energy Efficiency Dataset (768 building shapes, 8 architectural features).
* **Experimental Setup:** 60/40 train/test split (**460 training samples**, **308 test samples**).
* **Task:** Predict Heating Load ($Y_1$).
* **Results:**
  - **Decision Tree (Baseline):** **RMSE = 1.037341 | Accuracy = 97.113%**
  - **GPR (RBF + RQ Hybrid):** **RMSE = 1.310292 | Accuracy = 94.982%** (Top GP Model)
  - **GPR (Matérn 5/2):** RMSE = 1.524127 | Accuracy = 94.297%
  - **Linear Regression:** RMSE = 2.837534 | Accuracy = 90.218%

### 3.5 Gas Turbine NOx Emissions Prediction
* **Dataset:** UCI Gas Turbine CO and NOx Emissions Dataset.
* **Experimental Setup:** Cleaned dataset of **7,151 samples** (10 operational features), 80/20 train/test split (**5,720 training samples**, **1,431 test samples**).
* **Task:** Predict NOx emission concentration ($mg/m^3$).
* **Results:**
  - **GPR (Matérn):** **RMSE = 2.889904 | Accuracy = 97.094%** (Top Performing)
  - **GPR (Rational Quadratic):** RMSE = 2.890578 | Accuracy = 96.997%
  - **K-Nearest Neighbors:** RMSE = 3.241098 | Accuracy = 96.787%
  - **Decision Tree:** RMSE = 5.249555 | Accuracy = 94.331%
  - **Linear Regression:** RMSE = 6.154854 | Accuracy = 93.163%

### 3.6 SARCOS Anthropomorphic Robot Arm Dynamics
* **Dataset:** GPML SARCOS Robot Arm Dataset (44,484 full train / 4,449 full test, 21 joint state features).
* **Experimental Setup:** Subsampled **5,000 training samples** and **1,000 test samples** for GPR evaluation.
* **Task:** Model inverse kinematics (predict joint 1 torque $\tau_1$).
* **Results:**
  - **GPR (Rational Quadratic):** **SMSE = 0.039985 | $R^2 = 0.960015$ | RMSE = 4.085159** (Top Performing)
  - **KNN Regressor:** SMSE = 0.075142 | $R^2 = 0.924858$ | RMSE = 5.600157
  - **Ridge Regression:** SMSE = 0.078939 | $R^2 = 0.921061$ | RMSE = 5.739895
  - **GPR (Matérn):** SMSE = 0.154116 | $R^2 = 0.845884$ | RMSE = 8.020153

---

## 4. RESEARCH PRESENTATION & DOCUMENTATION DECKS

To communicate findings academically, two clean LaTeX Beamer presentation decks were created, compiled, and published:

1. **`GPML_Chapters_1_and_2_Presentation.pdf` (13 Slides):**
   * Covers Supervised Learning Setup, Under-determination, Inductive Bias (Restriction vs. Preference), Gaussian Likelihood, Weight-Space Posterior Derivation, Predictive Distributions, and **The Dimension Bottleneck ($N \times N$) & Woodbury Identity Solution ($n \times n$)**.
2. **`GPML_Chapter_4_Presentation.pdf` (13 Slides):**
   * Covers Covariance Functions as Similarity Metrics, Stationarity, Isotropy, Positive Semidefiniteness (Gram Matrices), Spectral Density (Bochner's Theorem), MS Continuity & Differentiability, Rice's Upcrossing Formula, and Kernel Families (SE, Matérn, RQ).

---

## 5. SUMMARY OF KEY CONTRIBUTIONS & CONCLUSION

During this internship period (May 25 – July 5, 2026), I successfully delivered:
1. **Mathematical Mastery:** Rigorous understanding of non-parametric Bayesian learning, completing the square for Gaussian posteriors, Woodbury matrix transformations, and kernel differentiability properties.
2. **Clean Codebase:** 12 fully-documented Jupyter Notebooks providing side-by-side comparison of from-scratch NumPy/SciPy implementations against Scikit-Learn.
3. **Empirical Benchmarks:** Demonstrated that Gaussian Process Regression consistently outperforms traditional linear, polynomial, tree-based, and neighbor-based regressors on complex continuous tasks (achieving up to **98.87% accuracy** on battery SOH and **$R^2 = 0.9600$** on SARCOS dynamics).
4. **Open-Source Repository:** Published all code, datasets, and presentation slide decks at [GitHub: Prateek312413/IITM_Internship](https://github.com/Prateek312413/IITM_Internship).
