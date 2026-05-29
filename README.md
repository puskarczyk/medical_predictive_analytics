# Medical Predictive Analytics: NHANES Chronic Conditions Classification

This repository contains two data science pipelines designed to predict chronic health conditions — **Diabetes** and **Depression** — using the **NHANES 2017–2018** datasets.

The project leverages demographic indicators, physiological examinations, lifestyle factors, and clinical laboratory values to benchmark both traditional and modern gradient-boosted machine learning frameworks.


# Technical Overview & Architecture

Both notebooks follow a standardized machine learning workflow implemented in Python:

* **Multi-Source Data Merging**
  Integration of multiple NHANES `.xpt` files using the unique respondent identifier (`SEQN`).

* **Feature Engineering & Data Cleaning**
  Transformation of clinical and survey-specific codes into standardized numerical matrices, including handling ambiguous responses (`7`, `9`, etc.) and missing values.

* **Class Imbalance Handling**
  Application of **SMOTE (Synthetic Minority Over-sampling Technique)** to mitigate skewed target distributions.

* **Model Benchmarking**
  Comparative evaluation of classical machine learning algorithms and modern gradient boosting frameworks.

* **Explainability & Validation**
  Error analysis, threshold optimization, feature importance visualization, and performance diagnostics using ROC/PR metrics and confusion matrices.

---

# Diabetes Predictive Analytics (`diabetes.ipynb`)

## Dataset Construction

The diabetes pipeline combines multiple specialized NHANES diagnostic datasets from the **2017–2018 cycle**:

* `DEMO_J.xpt` — Demographics
* `BMX_J.xpt` — Body Measures
* `BPX_J.xpt` — Blood Pressure
* `GLU_J.xpt` — Plasma Fasting Glucose
* `DIQ_J.xpt` — Diabetes Questionnaire
* `INS_J.xpt` — Insulin
* `TCHOL_J.xpt` — Total Cholesterol
* `TRIGLY_J.xpt` — Triglycerides & LDL

Two feature subsets were constructed for comparative validation:

### Basic Feature Set (`data1`)

Includes:

* Age (`RIDAGEYR`)
* Gender (`RIAGENDR`)
* BMI (`BMXBMI`)
* Systolic Blood Pressure (`BPXSY1`)
* Fasting Glucose (`LBXGLU`)

### Extended Feature Set (`data2`)

Adds:

* Waist Circumference (`BMXWAIST`)
* Insulin Levels (`LBXIN`)
* Total Cholesterol (`LBXTC`)

---

## Model Benchmarking Results

Models were evaluated using stratified train/test splits (`random_state=42`).

| Model              | Accuracy | ROC AUC | Avg Precision | F1 (macro) | Precision (1) | Recall (1) | F1 (1) |
| ------------------ | -------- | ------- | ------------- | ---------- | ------------- | ---------- | ------ |
| LR (Basic)         | 0.8699   | 0.9346  | 0.7997        | 0.7879     | 0.5304        | 0.8592     | 0.6559 |
| LR (Extended)      | 0.8721   | 0.9252  | 0.7919        | 0.7883     | 0.5327        | 0.8507     | 0.6552 |
| LightGBM (+ SMOTE) | 0.9147   | 0.9350  | 0.7953        | 0.8301     | 0.6901        | 0.7313     | 0.7101 |
| XGBoost (+ SMOTE)  | 0.9147   | 0.9362  | 0.7685        | 0.8359     | 0.6753        | 0.7761     | 0.7222 |

> **Observation:**
> Logistic Regression achieved higher clinical recall for screening sensitivity, while gradient-boosted models significantly reduced false positives and improved overall F1-macro performance.

---

# Module 2: Depression Screening Analytics (`depression.ipynb`)

## Target Variable & Feature Engineering

### Depression Target Definition

The target label (`depression`) was computed from the **PHQ-9 questionnaire** responses (`DPQ010`–`DPQ090`).

Responses were aggregated into a cumulative score ranging from **0–27**, using the clinically validated threshold:

* **PHQ-9 score ≥ 10 → Positive Depression Screening**

---

## Features Used

The model incorporates multidimensional predictors grouped into three primary domains:

### Demographics

* Age
* Gender
* Education level (`DMDEDUC2`)
* Marital status
* Family poverty income ratio (`INDFMPIR`)

### Lifestyle Indicators

* BMI
* Sleep duration (`SLD010H`)
* Alcohol consumption frequency
* Tobacco use history
* Physical activity indicators (`PAQ605`, `PAQ650`)
* Food security variables

### Comorbidities

* Diabetes diagnosis history
* Hypertension
* Sleep disorders (`SLQ050`)
* Cardiovascular diseases
* Asthma

---

## Implemented Machine Learning Models

The depression pipeline extends the benchmarking framework with additional classifiers:

* Logistic Regression
* LightGBM Classifier
* XGBoost Classifier
* CatBoost Classifier
* Multi-Layer Perceptron (`MLPClassifier`)

The notebook also includes:

* custom probability threshold tuning,
* precision-recall optimization,
* and clinical screening calibration (e.g. `threshold = 0.45`).


---

# Key Dependencies

| Library                           | Purpose                                    |
| --------------------------------- | ------------------------------------------ |
| `pandas`, `numpy`                 | Data manipulation and numerical processing |
| `scikit-learn`                    | Preprocessing, metrics, ML models          |
| `lightgbm`, `xgboost`, `catboost` | Gradient boosting frameworks               |
| `imbalanced-learn`                | SMOTE oversampling                         |
| `matplotlib`, `seaborn`           | Data visualization and evaluation plots    |

---


# Project Goals

This project aims to demonstrate how publicly available epidemiological datasets can be leveraged for:

* early chronic disease screening,
* predictive healthcare analytics,
* clinical risk stratification,
* and interpretable machine learning research.

The implementation emphasizes reproducibility, model comparison, and practical screening-oriented evaluation metrics commonly used in medical AI systems.
