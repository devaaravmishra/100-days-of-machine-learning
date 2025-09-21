# 🚀 100 Days of Machine Learning

A comprehensive collection of machine learning concepts, implementations, and hands-on projects covering everything from data preprocessing to advanced algorithms and model evaluation.

## 📋 Table of Contents

-   [Overview](#overview)
-   [Learning Path](#learning-path)
-   [Data Handling & Preprocessing](#data-handling--preprocessing)
-   [Exploratory Data Analysis](#exploratory-data-analysis)
-   [Feature Engineering](#feature-engineering)
-   [Machine Learning Algorithms](#machine-learning-algorithms)
-   [Model Evaluation & Metrics](#model-evaluation--metrics)
-   [Advanced Topics](#advanced-topics)
-   [Getting Started](#getting-started)
-   [Project Structure](#project-structure)
-   [Contributing](#contributing)

## 🎯 Overview

This repository contains a structured learning journey through machine learning concepts, organized into logical modules. Each module includes:

-   **Jupyter Notebooks** with step-by-step implementations
-   **Real-world datasets** for hands-on practice
-   **Visualizations** and interactive demos
-   **Code examples** from scratch and using scikit-learn
-   **Mathematical explanations** with practical applications

## 🛤️ Learning Path

### Phase 1: Data Fundamentals

1. **Data Sources & Formats** → **Data Preprocessing** → **Exploratory Analysis**

### Phase 2: Feature Engineering

1. **Encoding Techniques** → **Scaling & Normalization** → **Feature Selection**

### Phase 3: Machine Learning Core

1. **Linear Models** → **Tree-based Methods** → **Ensemble Techniques**

### Phase 4: Advanced Topics

1. **Clustering** → **Dimensionality Reduction** → **Model Optimization**

---

## 📊 Data Handling & Preprocessing

### Data Sources

-   **`csv-files/`** - Working with CSV files, different separators, and data loading techniques
-   **`json-and-sql/`** - JSON data handling and SQL integration
-   **`api-to-dataframe/`** - Converting API responses to pandas DataFrames
-   **`pandas-dataframe-web-scraping/`** - Web scraping and data extraction

### Data Cleaning & Missing Values

-   **`complete-case-analysis/`** - Complete case analysis for missing data
-   **`imputing-numerical-data/`** - Mean, median, and arbitrary value imputation
-   **`handling-missing-categorical-data/`** - Frequent value and missing category imputation
-   **`missing-indicator/`** - Missing value indicators and random sample imputation
-   **`knn-imputer/`** - K-Nearest Neighbors imputation
-   **`iterative-imputer/`** - Iterative imputation techniques

### Data Types & Mixed Variables

-   **`handling-mixed-variables/`** - Working with mixed data types
-   **`handling-date-and-time/`** - Date and time feature engineering

---

## 🔍 Exploratory Data Analysis

### Statistical Analysis

-   **`descriptive-stats/`** - Understanding your data with descriptive statistics
-   **`univariate-analysis/`** - Single variable analysis and distributions
-   **`bivariate-analysis/`** - Two-variable relationships and correlations
-   **`pandas-profiling/`** - Automated EDA with pandas profiling

### Outlier Detection & Treatment

-   **`outlier-removal-zscore/`** - Z-score based outlier detection
-   **`outlier-removal-iqr/`** - Interquartile Range (IQR) method
-   **`outlier-detection-percentiles/`** - Percentile-based outlier detection

---

## ⚙️ Feature Engineering

### Encoding Techniques

-   **`ordinal-encoding/`** - Ordinal categorical encoding
-   **`one-hot-encoding/`** - One-hot encoding for categorical variables
-   **`binning-and-binarization/`** - Feature binning and binarization

### Scaling & Normalization

-   **`standardization/`** - Z-score standardization
-   **`normalization/`** - Min-max normalization
-   **`power-transformer/`** - Power and Box-Cox transformations

### Advanced Feature Engineering

-   **`feature-construction-and-splitting/`** - Creating and splitting features
-   **`function-transformer/`** - Custom function transformations
-   **`column-transformer/`** - Column-wise transformations
-   **`sklearn-pipelines/`** - End-to-end ML pipelines

---

## 🤖 Machine Learning Algorithms

### Linear Models

-   **`simple-linear-regression/`** - Single variable linear regression
-   **`multiple-linear-regression/`** - Multiple variable linear regression
-   **`polynomial-regression/`** - Polynomial feature regression
-   **`regularized-linear-models/`** - Ridge, Lasso, and Elastic Net
-   **`lasso-regression/`** - Lasso regression implementation
-   **`elasticnet-regression/`** - Elastic Net regression

### Classification

-   **`logistic-regression/`** - Logistic regression fundamentals
-   **`logistic-regression-continued/`** - Advanced logistic regression topics

### Tree-Based Methods

-   **`random-forest/`** - Random Forest implementation and analysis
-   **`adaboost/`** - AdaBoost algorithm and hyperparameter tuning
-   **`gradient-boosting/`** - Gradient Boosting implementation

### Ensemble Methods

-   **`stacking-and-blending/`** - Model stacking and blending techniques

### Clustering

-   **`kmeans/`** - K-Means clustering with interactive demos

---

## 📈 Model Evaluation & Metrics

### Regression Metrics

-   **`regression-metrics/`** - MAE, MSE, RMSE, R², and more

### Classification Metrics

-   **`classification-metrics/`** - Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 🧮 Mathematical Foundations

### Optimization

-   **`gradient-descent/`** - Gradient descent from scratch with animations
-   **`types-of-gradient-descent/`** - Batch, Stochastic, and Mini-batch GD

### Dimensionality Reduction

-   **`pca/`** - Principal Component Analysis

---

## 🚀 Advanced Topics

### Model Optimization

-   **`sklearn-pipelines/`** - Complete ML pipeline implementation
-   **`stacking-and-blending/`** - Advanced ensemble techniques

---

## 🛠️ Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running the Notebooks

1. Clone the repository
2. Navigate to any topic folder
3. Open the Jupyter notebook
4. Run the cells to see the implementation

### Recommended Learning Order

1. Start with **Data Handling & Preprocessing** modules
2. Move to **Exploratory Data Analysis**
3. Learn **Feature Engineering** techniques
4. Implement **Machine Learning Algorithms**
5. Master **Model Evaluation & Metrics**
6. Explore **Advanced Topics**

---

## 📁 Project Structure

```
100-days-of-machine-learning/
├── 📊 Data Handling & Preprocessing/
│   ├── csv-files/                    # CSV data manipulation
│   ├── json-and-sql/                 # JSON and SQL integration
│   ├── api-to-dataframe/             # API data conversion
│   ├── pandas-dataframe-web-scraping/ # Web scraping
│   ├── complete-case-analysis/        # Missing data analysis
│   ├── imputing-numerical-data/       # Numerical imputation
│   ├── handling-missing-categorical-data/ # Categorical imputation
│   ├── missing-indicator/             # Missing value indicators
│   ├── knn-imputer/                   # KNN imputation
│   ├── iterative-imputer/             # Iterative imputation
│   ├── handling-mixed-variables/      # Mixed data types
│   └── handling-date-and-time/        # Date/time features
│
├── 🔍 Exploratory Data Analysis/
│   ├── descriptive-stats/             # Statistical summaries
│   ├── univariate-analysis/           # Single variable analysis
│   ├── bivariate-analysis/            # Two variable analysis
│   ├── pandas-profiling/              # Automated EDA
│   ├── outlier-removal-zscore/        # Z-score outliers
│   ├── outlier-removal-iqr/           # IQR outliers
│   └── outlier-detection-percentiles/ # Percentile outliers
│
├── ⚙️ Feature Engineering/
│   ├── ordinal-encoding/              # Ordinal encoding
│   ├── one-hot-encoding/              # One-hot encoding
│   ├── binning-and-binarization/      # Feature binning
│   ├── standardization/               # Z-score scaling
│   ├── normalization/                 # Min-max scaling
│   ├── power-transformer/             # Power transformations
│   ├── feature-construction-and-splitting/ # Feature creation
│   ├── function-transformer/          # Custom transformations
│   ├── column-transformer/            # Column transformations
│   └── sklearn-pipelines/             # ML pipelines
│
├── 🤖 Machine Learning Algorithms/
│   ├── simple-linear-regression/      # Linear regression
│   ├── multiple-linear-regression/    # Multiple regression
│   ├── polynomial-regression/         # Polynomial regression
│   ├── regularized-linear-models/     # Regularized models
│   ├── lasso-regression/              # Lasso regression
│   ├── elasticnet-regression/         # Elastic Net
│   ├── logistic-regression/           # Logistic regression
│   ├── logistic-regression-continued/ # Advanced logistic regression
│   ├── random-forest/                 # Random Forest
│   ├── adaboost/                      # AdaBoost
│   ├── gradient-boosting/             # Gradient Boosting
│   ├── stacking-and-blending/         # Ensemble methods
│   └── kmeans/                        # K-Means clustering
│
├── 📈 Model Evaluation/
│   ├── regression-metrics/            # Regression evaluation
│   └── classification-metrics/        # Classification evaluation
│
├── 🧮 Mathematical Foundations/
│   ├── gradient-descent/              # Gradient descent
│   ├── types-of-gradient-descent/     # GD variants
│   └── pca/                           # Principal Component Analysis
│
└── 📚 Additional Resources/
    ├── adaboost_demo.ipynb            # AdaBoost demonstration
    └── README.md                      # This file
```

---

## 🎯 Key Features

-   **📚 Comprehensive Coverage**: From basic data handling to advanced ML algorithms
-   **🔬 Hands-on Learning**: Interactive Jupyter notebooks with real datasets
-   **📊 Visual Learning**: Rich visualizations and animations (especially in gradient descent)
-   **🛠️ Practical Implementation**: Both from-scratch and scikit-learn implementations
-   **📈 Real-world Applications**: Industry-standard datasets and use cases
-   **🎨 Interactive Demos**: Streamlit apps and interactive visualizations

---

## 🚀 Quick Start Examples

### Data Loading

```python
import pandas as pd
df = pd.read_csv('your_data.csv')
```

### Basic EDA

```python
import matplotlib.pyplot as plt
df.describe()
df.hist()
plt.show()
```

### Machine Learning Pipeline

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:

-   Add new machine learning topics
-   Improve existing implementations
-   Fix bugs or enhance documentation
-   Add more datasets or examples

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

-   **[CampusX YouTube Playlist](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH)** - 100 Days of Machine Learning course that inspired this repository
-   Scikit-learn community for excellent documentation
-   Pandas team for powerful data manipulation tools
-   Matplotlib and Seaborn for visualization capabilities
-   The machine learning community for continuous learning and sharing

---

**Happy Learning! 🎉**
