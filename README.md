# Gaussian Process Regression & Machine Learning for Battery SOH and Environmental Forecasting

Welcome to the repository containing the machine learning models and statistical frameworks developed during the **IITM Research Internship**. This repository hosts **12 Jupyter Notebooks** representing 6 distinct predictive tasks, each implemented in two comparable settings: **from-scratch (NumPy/SciPy)** and **Scikit-Learn**. 

To maintain a clean and lightweight repository, all raw datasets and auxiliary build scripts are ignored by Git, leaving only the core notebook implementations, configuration files, and this comprehensive guide.

---

## 🌟 Key Highlights

* **GPR from Scratch vs. Scikit-Learn:** Direct comparison of manual Gaussian Process Regression (raw NumPy using Cholesky factorizations for numerical stability) and standard Scikit-Learn implementations, yielding mathematically identical metrics.
* **Baselines from Scratch:** Custom NumPy implementations of Linear Regression, Polynomial Regression (Deg 2), K-Nearest Neighbors (KNN), and Decision Trees compared against their Scikit-Learn equivalents.
* **Advanced Kernels:** Implementations include Radial Basis Function (RBF), Matérn (3/2 & 5/2), Rational Quadratic (RQ), Automatic Relevance Determination (ARD), and Composite/Hybrid kernels.

---

## 📂 Repository Structure

The repository contains exactly the following files:
* **`*.ipynb`**: The 12 Jupyter Notebooks.
* **`requirements.txt`**: List of dependencies to reproduce the results.
* **`.gitignore`**: Git rules ensuring only code and documentation are tracked.
* **`README.md`**: This guide.

---

## 📊 Projects and Datasets Detail

### 1. Battery State-of-Health (SOH) Prediction
* **Notebooks:** [Battery_SOH_GPR_Scratch.ipynb](Battery_SOH_GPR_Scratch.ipynb) | [Battery_SOH_GPR_Sklearn.ipynb](Battery_SOH_GPR_Sklearn.ipynb)
* **Dataset:** NASA Li-ion Battery Aging Dataset (B0005). The dataset tracks battery capacity degradation over charge/discharge cycles at room temperature.
* **Size:** 168 discharge cycles (117 training cycles, 51 test cycles).
* **Features:** Voltage, current, temperature, and elapsed time under discharge.
* **Target Variable:** State of Health (SOH) capacity (Ah).
* **Summary Table:**
  | Model | RMSE | Accuracy (%) |
  | :--- | :---: | :---: |
  | **GPR (Matern 5/2)** | **0.010081** | **98.872%** |
  | **GPR (Linear)** | 0.016611 | 98.300% |
  | **Linear Regression** | 0.016612 | 98.300% |
  | **GPR (Matern 3/2)** | 0.022418 | 97.566% |
  | **GPR (RBF+Linear)** | 0.031424 | 96.224% |
  | **GPR (RBF)** | 0.037014 | 95.603% |
  | **Decision Tree** | 0.042997 | 94.671% |
  | **K-Nearest Neighbors** | 0.045477 | 94.310% |
  | **GPR (RQ)** | 0.087043 | 89.751% |
  | **Polynomial Regression (Deg 2)** | 0.089741 | 88.862% |

---

### 2. California Housing Value Prediction
* **Notebooks:** [California_Housing_GPR_Scratch.ipynb](California_Housing_GPR_Scratch.ipynb) | [California_Housing_GPR_Sklearn.ipynb](California_Housing_GPR_Sklearn.ipynb)
* **Dataset:** StatLib California Housing dataset (derived from the 1990 US Census). 
* **Size:** 20,640 total samples. 80/20 train/test split: **16,512 training samples** and **4,128 test samples**.
* **Features:** Median income, housing median age, average rooms, average bedrooms, population, average occupancy, latitude, longitude.
* **Target Variable:** Median house value (in $100k blocks).
* **Summary Table:**
  | Model | RMSE | Accuracy (%) |
  | :--- | :---: | :---: |
  | **GPR (RQ)** | **0.552962** | **78.907%** |
  | **GPR (Matern)** | 0.554475 | 78.916% |
  | **GPR (RBF)** | 0.561079 | 78.333% |
  | **GPR (ARD)** | 0.562854 | 78.980% |
  | **K-Nearest Neighbors** | 0.652351 | 76.262% |
  | **Decision Tree (Baseline)** | 0.733951 (Scratch) / 0.734604 (Sklearn) | 67.702% (Scratch) / 67.318% (Sklearn) |
  | **Linear Regression** | 0.745104 | 68.040% |
  | **Polynomial Regression (Deg 2)** | 0.754783 | 68.835% |

---

### 3. Delhi Air Quality Index (AQI) Forecasting
* **Notebooks:** [Delhi_AQI_GPR_Scratch.ipynb](Delhi_AQI_GPR_Scratch.ipynb) | [Delhi_AQI_GPR_Sklearn.ipynb](Delhi_AQI_GPR_Sklearn.ipynb)
* **Dataset:** Central Pollution Control Board (CPCB) India hourly and daily AQI tracking database.
* **Size:** 16,000 hourly samples (71 features) sampled after cleaning. 90/10 train/test split: **14,400 training samples** and **1,600 test samples**.
* **Features:** Air pollutants values (PM2.5, PM10, NO, NO2, NOx, CO, SO2, O3, Benzene, Toluene, Xylene).
* **Target Variable:** Air Quality Index (AQI).
* **Summary Table:**
  | Model | RMSE | Accuracy (%) |
  | :--- | :---: | :---: |
  | **GPR (Matern 3/2)** | **37.366086** | **88.055%** |
  | **GPR (Matern 5/2)** | 37.905587 | 87.811% |
  | **GPR (Rational Quadratic)** | 39.392439 | 87.340% |
  | **GPR (RBF)** | 40.772827 | 86.835% |
  | **K-Nearest Neighbors** | 42.944977 | 87.034% |
  | **Decision Tree (Baseline)** | 52.127295 (Scratch) / 51.944764 (Sklearn) | 83.139% (Scratch) / 83.117% (Sklearn) |
  | **Polynomial Regression (Deg 2)** | 54.570040 | 81.709% |
  | **Linear Regression** | 57.084957 | 80.752% |

---

### 4. Residential Building Energy Efficiency
* **Notebooks:** [Energy_Prediction_GPR_Scratch.ipynb](Energy_Prediction_GPR_Scratch.ipynb) | [Energy_Prediction_GPR_Sklearn.ipynb](Energy_Prediction_GPR_Sklearn.ipynb)
* **Dataset:** UCI Energy Efficiency Dataset (ENB2012). Simulates various residential building shapes and properties.
* **Size:** 768 total samples (8 features). 60/40 train/test split: **460 training samples** and **308 test samples**.
* **Features:** Relative compactness, surface area, wall area, roof area, overall height, orientation, glazing area, glazing area distribution.
* **Target Variable:** Heating Load (Y₁).
* **Summary Table:**
  | Model | RMSE | Accuracy (%) |
  | :--- | :---: | :---: |
  | **Decision Tree** | **1.037341** | **97.113%** |
  | **GPR (RBF+RQ Hybrid)** | 1.310292 | 94.982% |
  | **GPR (Matern 5/2)** | 1.524127 | 94.297% |
  | **GPR (RQ)** | 1.541599 | 94.302% |
  | **GPR (RBF)** | 1.542573 | 93.907% |
  | **GPR (Matern 3/2)** | 1.619858 | 94.008% |
  | **Polynomial Regression (Deg 2)** | 2.420561 | 89.062% |
  | **K-Nearest Neighbors** | 2.495632 (Scratch) / 2.491383 (Sklearn) | 92.442% (Scratch) / 92.450% (Sklearn) |
  | **GPR (Linear)** | 2.836373 | 90.223% |
  | **Linear Regression** | 2.837534 | 90.218% |

---

### 5. Gas Turbine NOx Emissions Prediction
* **Notebooks:** [NOx_Emissions_GPR_Scratch.ipynb](NOx_Emissions_GPR_Scratch.ipynb) | [NOx_Emissions_GPR_Sklearn.ipynb](NOx_Emissions_GPR_Sklearn.ipynb)
* **Dataset:** UCI Gas Turbine CO and NOx Emissions Dataset.
* **Size:** 7,151 cleaned samples (10 features). 80/20 train/test split: **5,720 training samples** and **1,431 test samples**.
* **Features:** Ambient temperature, pressure, humidity, air filter difference pressure, gas turbine exhaust pressure, compressor discharge pressure, energy yield, carbon monoxide.
* **Target Variable:** NOx emissions (mg/m³).
* **Summary Table:**
  | Model | RMSE | Accuracy (%) |
  | :--- | :---: | :---: |
  | **GPR (Matern)** | **2.889904** | **97.094%** |
  | **GPR (RQ)** | 2.890578 | 96.997% |
  | **GPR (RBF)** | 3.094508 | 96.844% |
  | **K-Nearest Neighbors** | 3.241098 | 96.787% |
  | **GPR (ARD)** | 3.331866 | 96.524% |
  | **Decision Tree** | 5.249555 (Scratch) / 5.357995 (Sklearn) | 94.331% (Scratch) / 94.051% (Sklearn) |
  | **Polynomial Regression (Deg 2)** | 5.747332 | 93.886% |
  | **Linear Regression** | 6.154854 | 93.163% |

---

### 6. SARCOS Robot Arm Dynamics (Inverse Kinematics)
* **Notebooks:** [Sarcos_Robot_GPR_Scratch.ipynb](Sarcos_Robot_GPR_Scratch.ipynb) | [Sarcos_Robot_GPR_Sklearn.ipynb](Sarcos_Robot_GPR_Sklearn.ipynb)
* **Dataset:** GPML SARCOS Robot Arm data. Models inverse kinematics for an anthropomorphic robot arm.
* **Size:** Full dataset of 44,484 training and 4,449 test samples (21 features). Subsampled for GPR to **5,000 training samples** and **1,000 test samples**.
* **Features:** 21 variables representing joint positions (7), velocities (7), and accelerations (7).
* **Target Variable:** Torque on the first joint.
* **Summary Table:**
  | Model | SMSE | R² | RMSE | MAE |
  | :--- | :---: | :---: | :---: | :---: |
  | **GPR (Rational Quadratic)** | **0.039985** | **0.960015** | **4.085159** | **2.360496** |
  | **K-Nearest Neighbors** | 0.075142 | 0.924858 | 5.600157 | 3.514242 |
  | **Ridge Regression** | 0.078939 | 0.921061 | 5.739895 | 3.979843 |
  | **GPR (Matern)** | 0.154116 | 0.845884 | 8.020153 | 4.152606 |
  | **GPR (SE/RBF)** | 0.179732 | 0.820268 | 8.661074 | 4.431183 |

## 📚 Gaussian Process (GPML) Presentation Slide Decks

This repository includes two comprehensive Beamer presentation slide decks prepared during the **IIT Madras Research Internship** under the guidance of **Prof. Neelesh Shankar Upadhye** (Department of Mathematics):

### 1. 📄 [GPML_Chapters_1_and_2_Presentation.pdf](GPML_Chapters_1_and_2_Presentation.pdf)
* **Title:** Surrogate Modeling in Statistical Learning: A Deep Dive into Gaussian Processes (Chapters 1 & 2)
* **Author:** Prateek Raj (BT23CS021)
* **Key Topics Covered (13 Slides):**
  * **Supervised Learning Setup:** Continuous regression, discrete classification, and generalisation bounds.
  * **Inductive Bias & Ambiguity:** Under-determination problem and why every ML algorithm requires inductive bias.
  * **Restriction vs. Preference Bias:** Parametric constraints vs. Bayesian prior distributions over functions.
  * **Likelihood & Noise Modeling:** Central Limit Theorem justification for i.i.d. Gaussian noise.
  * **Weight-Space Bayesian Regression:** Derivation of posterior mean μ_post and covariance Σ_post by completing the square.
  * **Predictive Distribution:** Integrating out weight parameters to obtain predictive mean and variance (epistemic uncertainty).
  * **Feature Projections & Dimension Bottleneck:** High-dimensional basis expansion φ(x), the N × N matrix inversion bottleneck, and its exact resolution using the **Woodbury Matrix Identity** to invert an n × n data matrix.

---

### 2. 📄 [GPML_Chapter_4_Presentation.pdf](GPML_Chapter_4_Presentation.pdf)
* **Title:** Surrogate Modeling in Statistical Learning: Covariance Functions (Chapter 4)
* **Author:** Prateek Raj (BT23CS021)
* **Key Topics Covered:**
  * **Covariance Functions as Similarity Metrics:** Role of kernel design in encoding functional properties.
  * **Stationarity & Isotropy:** Translation and rotation invariants vs. non-stationary dot product kernels.
  * **Positive Semidefiniteness (PSD):** Proof of non-negative variance and Gram matrix validity.
  * **Spectral Density & Bochner’s Theorem:** Dual spatial-frequency representation of kernels via Wiener-Khintchine.
  * **Path Properties:** Mean Square (MS) continuity and differentiability conditions at the origin.
  * **Upcrossings & Curvature:** Rice's formula and length-scale relationship to path wriggliness.
  * **Covariance Families:** Squared Exponential (RBF), γ-Exponential, Matérn class (ν = 1/2, 3/2, 5/2), and Rational Quadratic (RQ).

---

## 🛠️ Setup and Installation

### 1. Install Dependencies
Ensure you have Python installed, then install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. File Requirements
Since datasets are git-ignored, before running the notebooks locally, ensure you place the data files in their expected locations:
- `Battery_DataSet/` (containing `B0005.mat`) in the root directory.
- `INDIA_AQI_COMPLETE_20251126.csv` in the root directory.
- `ENB2012_data.csv` in the root directory.
- `gt_2014.csv` in the root directory.
- `sarcos_train.csv` and `sarcos_test.csv` in the root directory.
