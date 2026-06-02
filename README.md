# Frequent Value Imputation: Mode-Based Missing Value Recovery

[![Machine Learning](https://img.shields.io/badge/Domain-Data%20Preprocessing-blue)](https://scikit-learn.org/)
[![Preprocessing](https://img.shields.io/badge/Strategy-Frequent%20Value%20Imputation-orange)](https://scikit-learn.org/stable/modules/impute.html)
[![Dataset](https://img.shields.io/badge/Dataset-Housing%20Dataset-green)](./housing_dataset.csv)

---

## 🏗️ Project Overview

Missing values are one of the most common problems encountered in machine learning and data analysis. When categorical features contain missing observations, statistical methods such as mean or median imputation are not suitable because categories do not possess numerical centers.

This project explores **Frequent Value Imputation (Mode Imputation)** using the **Housing Dataset (`housing_dataset.csv`)**. The notebook demonstrates how missing values can be replaced with the most frequently occurring category while preserving the overall dataset size.

The implementation includes:

* Manual imputation using **Pandas**
* Automated imputation using **Scikit-Learn's `SimpleImputer`**
* Distribution comparisons before and after imputation
* Frequency analysis of categorical variables

---

## 🛠️ Advanced Engineering Mechanics

### 1. What is Frequent Value Imputation?

Frequent Value Imputation replaces missing values with the most commonly occurring value (mode) of a feature.

Example:

| Property Type |
| ------------- |
| Apartment     |
| Apartment     |
| House         |
| Missing       |
| Apartment     |

Mode = **Apartment**

After Imputation:

| Property Type |
| ------------- |
| Apartment     |
| Apartment     |
| House         |
| Apartment     |
| Apartment     |

This approach assumes that missing observations are likely to belong to the dominant category.

---

### 2. Statistical Assumptions

Frequent Value Imputation performs best when:

* Missing values are relatively small in proportion.
* The dominant category genuinely represents the population.
* Feature categories are stable and meaningful.
* Missingness is approximately random.

However, excessive missing values may artificially inflate the dominant category and introduce bias.

---

### 3. The Preprocessing Pipeline Architecture

```text
                 ┌─────────────────────────────────────┐
                 │       Raw Housing Dataset           │
                 └─────────────────────────────────────┘
                                   │
                                   ▼

                 ┌─────────────────────────────────────┐
                 │      Missing Value Detection        │
                 └─────────────────────────────────────┘
                                   │
                                   ▼

                 ┌─────────────────────────────────────┐
                 │  Most Frequent Category Discovery   │
                 └─────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                                                     │
        ▼                                                     ▼

    Pandas fillna()                              SimpleImputer
                                                 (most_frequent)

        │                                                     │
        ▼                                                     ▼

 Manual Category Fill                     Stateful ML Pipeline

        │                                                     │
        └──────────────────────┬──────────────────────────────┘
                               ▼

                 ┌─────────────────────────────────────┐
                 │     Complete Feature Matrix         │
                 └─────────────────────────────────────┘
```

---

## 🔬 Implementation Workflow

The notebook follows the following preprocessing pipeline:

### Step 1: Dataset Exploration

* Load the Housing Dataset
* Identify missing values
* Analyze categorical features

### Step 2: Frequency Analysis

* Calculate category frequencies
* Determine the mode of each feature
* Visualize category distributions

### Step 3: Pandas-Based Imputation

```python
mode_value = df['column'].mode()[0]

df['column'].fillna(mode_value, inplace=True)
```

### Step 4: Scikit-Learn Implementation

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
```

### Step 5: Distribution Validation

* Compare feature distributions before imputation
* Compare feature distributions after imputation
* Analyze frequency shifts

---

## 📊 Frequent Value Imputation Selection Matrix

| Imputation Method         | Data Type   | Outlier Sensitivity | Computational Cost | Best Use Case             |
| ------------------------- | ----------- | ------------------- | ------------------ | ------------------------- |
| Mean Imputation           | Numerical   | High                | Low                | Normally Distributed Data |
| Median Imputation         | Numerical   | Low                 | Low                | Skewed Numerical Data     |
| Frequent Value Imputation | Categorical | N/A                 | Very Low           | Categorical Features      |

---

## ⚠️ Limitations

### Frequency Inflation

The most common category becomes even more dominant after imputation.

### Information Loss

Rare categories become relatively less significant.

### Potential Bias

If missing values are not random, mode imputation may introduce systematic errors.

### Reduced Diversity

Large amounts of missing data can oversimplify the feature distribution.

---

## 🚀 Why This Technique Matters

Frequent Value Imputation is widely used because it:

* Preserves dataset size
* Is computationally efficient
* Works well for categorical features
* Integrates easily into machine learning pipelines
* Provides a strong preprocessing baseline

---

## 💻 Tech Stack & Dependencies

### Programming Language

* Python 3.9+

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn (`SimpleImputer`)

### Visualization

* Matplotlib
* Seaborn

### Development Environment

* Jupyter Notebook

---

## 📚 Libraries Used

```python
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 🎯 Key Learning Outcomes

After completing this notebook, you will understand:

* How Frequent Value Imputation works
* When to use mode imputation
* How to implement imputation using Pandas
* How to implement imputation using Scikit-Learn
* The impact of imputation on feature distributions
* Best practices for handling missing categorical data

---

## 🏁 Conclusion

Frequent Value Imputation is one of the simplest and most effective techniques for handling missing categorical data. By replacing missing observations with the most common category, we preserve dataset size and maintain compatibility with machine learning algorithms.

This project provides a practical comparison between manual and automated implementations while highlighting both the advantages and limitations of mode-based imputation.

---

## 🚀 Future Improvements

* Compare Mode vs Mean vs Median Imputation
* Explore KNN Imputation
* Implement Iterative Imputation
* Build Complete Preprocessing Pipelines
* Evaluate Model Performance Before and After Imputation
