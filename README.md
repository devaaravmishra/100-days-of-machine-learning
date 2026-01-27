# 🚀 100 Days of Machine Learning

A comprehensive collection of machine learning concepts, implementations, and hands-on projects covering everything from data preprocessing to advanced algorithms and model evaluation.

## 📋 Table of Contents

-   [Overview](#overview)
-   [Learning Path](#learning-path)
-   [Related Repository](#related-repository)
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

### 🔗 Related Repository

**Looking to continue your learning journey?** Check out the **[100 Days of Deep Learning](https://github.com/devaaravmishra/100-days-of-deep-learning)** repository, which builds on the ML fundamentals covered here to teach neural networks, CNNs, RNNs, transformers, and advanced deep learning concepts.

See the **[LEARNING_ROADMAP.md](LEARNING_ROADMAP.md)** for a complete guide on how these two repositories complement each other!

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

## 📊 01 - Data Handling

### Data Sources

-   **`01-data-handling/csv-files/`** - Working with CSV files, different separators, and data loading techniques
-   **`01-data-handling/json-and-sql/`** - JSON data handling and SQL integration
-   **`01-data-handling/api-to-dataframe/`** - Converting API responses to pandas DataFrames
-   **`01-data-handling/pandas-dataframe-web-scraping/`** - Web scraping and data extraction

## 🔧 02 - Data Preprocessing

### Data Cleaning & Missing Values

-   **`02-data-preprocessing/complete-case-analysis/`** - Complete case analysis for missing data
-   **`02-data-preprocessing/imputing-numerical-data/`** - Mean, median, and arbitrary value imputation
-   **`02-data-preprocessing/handling-missing-categorical-data/`** - Frequent value and missing category imputation
-   **`02-data-preprocessing/missing-indicator/`** - Missing value indicators and random sample imputation
-   **`02-data-preprocessing/knn-imputer/`** - K-Nearest Neighbors imputation
-   **`02-data-preprocessing/iterative-imputer/`** - Iterative imputation techniques

### Data Types & Mixed Variables

-   **`02-data-preprocessing/handling-mixed-variables/`** - Working with mixed data types
-   **`02-data-preprocessing/handling-date-and-time/`** - Date and time feature engineering

---

## 🔍 03 - Exploratory Data Analysis

### Statistical Analysis

-   **`03-exploratory-data-analysis/descriptive-stats/`** - Understanding your data with descriptive statistics
-   **`03-exploratory-data-analysis/univariate-analysis/`** - Single variable analysis and distributions
-   **`03-exploratory-data-analysis/bivariate-analysis/`** - Two-variable relationships and correlations
-   **`03-exploratory-data-analysis/pandas-profiling/`** - Automated EDA with pandas profiling

### Outlier Detection & Treatment

-   **`03-exploratory-data-analysis/outlier-removal-zscore/`** - Z-score based outlier detection
-   **`03-exploratory-data-analysis/outlier-removal-iqr/`** - Interquartile Range (IQR) method
-   **`03-exploratory-data-analysis/outlier-detection-percentiles/`** - Percentile-based outlier detection

---

## ⚙️ 04 - Feature Engineering

### Encoding Techniques

-   **`04-feature-engineering/ordinal-encoding/`** - Ordinal categorical encoding
-   **`04-feature-engineering/one-hot-encoding/`** - One-hot encoding for categorical variables
-   **`04-feature-engineering/binning-and-binarization/`** - Feature binning and binarization

### Scaling & Normalization

-   **`04-feature-engineering/standardization/`** - Z-score standardization
-   **`04-feature-engineering/normalization/`** - Min-max normalization
-   **`04-feature-engineering/power-transformer/`** - Power and Box-Cox transformations

### Advanced Feature Engineering

-   **`04-feature-engineering/feature-construction-and-splitting/`** - Creating and splitting features
-   **`04-feature-engineering/function-transformer/`** - Custom function transformations
-   **`04-feature-engineering/column-transformer/`** - Column-wise transformations
-   **`04-feature-engineering/sklearn-pipelines/`** - End-to-end ML pipelines

---

## 🤖 05 - Machine Learning Algorithms

### Linear Models

-   **`05-machine-learning-algorithms/simple-linear-regression/`** - Single variable linear regression
-   **`05-machine-learning-algorithms/multiple-linear-regression/`** - Multiple variable linear regression
-   **`05-machine-learning-algorithms/polynomial-regression/`** - Polynomial feature regression
-   **`05-machine-learning-algorithms/regularized-linear-models/`** - Ridge, Lasso, and Elastic Net
-   **`05-machine-learning-algorithms/lasso-regression/`** - Lasso regression implementation
-   **`05-machine-learning-algorithms/elasticnet-regression/`** - Elastic Net regression

### Classification

-   **`05-machine-learning-algorithms/logistic-regression/`** - Logistic regression fundamentals
-   **`05-machine-learning-algorithms/logistic-regression-continued/`** - Advanced logistic regression topics
-   **`05-machine-learning-algorithms/knn/`** - K-Nearest Neighbors classification
-   **`05-machine-learning-algorithms/svm/`** - Support Vector Machines

### Tree-Based Methods

-   **`05-machine-learning-algorithms/random-forest/`** - Random Forest implementation and analysis
-   **`05-machine-learning-algorithms/adaboost/`** - AdaBoost algorithm and hyperparameter tuning
-   **`05-machine-learning-algorithms/gradient-boosting/`** - Gradient Boosting implementation
-   **`05-machine-learning-algorithms/bagging/`** - Bagging ensemble methods

### Ensemble Methods

-   **`05-machine-learning-algorithms/voting-classifier/`** - Voting classifiers and regressors
-   **`05-machine-learning-algorithms/stacking-and-blending/`** - Model stacking and blending techniques

### Clustering

-   **`05-machine-learning-algorithms/kmeans/`** - K-Means clustering with interactive demos
-   **`05-machine-learning-algorithms/dbscan/`** - DBSCAN clustering algorithm

### Imbalanced Data

-   **`05-machine-learning-algorithms/imbalanced-data/`** - Handling imbalanced datasets

---

## 📈 06 - Model Evaluation & Metrics

### Regression Metrics

-   **`06-model-evaluation/regression-metrics/`** - MAE, MSE, RMSE, R², and more

### Classification Metrics

-   **`06-model-evaluation/classification-metrics/`** - Accuracy, Precision, Recall, F1-Score, ROC-AUC
-   **`06-model-evaluation/roc-curve/`** - ROC curves and AUC analysis

---

## 🚀 07 - Advanced Topics

### Optimization

-   **`07-advanced-topics/gradient-descent/`** - Gradient descent from scratch with animations
-   **`07-advanced-topics/types-of-gradient-descent/`** - Batch, Stochastic, and Mini-batch GD

### Dimensionality Reduction

-   **`07-advanced-topics/pca/`** - Principal Component Analysis

---

## 🛠️ Getting Started

### Prerequisites

- **Python 3.8+** (recommended: Python 3.10 or higher)
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- Basic knowledge of Python programming
- Familiarity with command line/terminal

### Installation

#### Option 1: Quick Setup (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/devaaravmishra/100-days-of-machine-learning.git
   cd 100-days-of-machine-learning
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   Or use JupyterLab:
   ```bash
   jupyter lab
   ```

#### Option 2: Using Conda

```bash
# Create conda environment
conda create -n ml-learning python=3.10
conda activate ml-learning

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

1. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

2. **Navigate to a topic folder** (e.g., `01-data-handling/csv-files/`)

3. **Open a notebook** (`.ipynb` file)

4. **Run cells** using `Shift + Enter` or click the "Run" button

5. **Follow along** with the explanations and code examples

### Recommended Learning Order

Follow this structured path for optimal learning:

1. **`01-data-handling/`** (4-6 hours)
   - Start with data sources and formats
   - Learn to load data from various sources

2. **`02-data-preprocessing/`** (8-12 hours)
   - Master data cleaning and preprocessing
   - Handle missing values and data types

3. **`03-exploratory-data-analysis/`** (6-8 hours)
   - Explore and understand your data
   - Identify patterns and outliers

4. **`04-feature-engineering/`** (10-14 hours)
   - Transform and engineer features
   - Build ML pipelines

5. **`05-machine-learning-algorithms/`** (20-30 hours)
   - Implement various ML models
   - Learn ensemble methods

6. **`06-model-evaluation/`** (4-6 hours)
   - Evaluate and measure model performance
   - Choose appropriate metrics

7. **`07-advanced-topics/`** (6-8 hours)
   - Dive into advanced optimization
   - Explore dimensionality reduction

**Total Estimated Time: 58-84 hours** (approximately 2-3 months at 1 hour/day)

### Tracking Your Progress

Use the [`PROGRESS_TRACKER.md`](PROGRESS_TRACKER.md) file to track your learning journey:
- Mark completed topics
- Add notes and key learnings
- Set goals and deadlines
- Track time spent

### What's Next?

After completing this repository, continue your learning journey with:
- **[100 Days of Deep Learning](https://github.com/devaaravmishra/100-days-of-deep-learning)** - Neural networks, CNNs, RNNs, transformers, and more
- **[LEARNING_ROADMAP.md](LEARNING_ROADMAP.md)** - Complete guide linking both repositories

---

## 📁 Project Structure

```
100-days-of-machine-learning/
├── 01-data-handling/                  # 📊 Data Sources & Formats
│   ├── csv-files/                     # CSV data manipulation
│   ├── json-and-sql/                  # JSON and SQL integration
│   ├── api-to-dataframe/              # API data conversion
│   └── pandas-dataframe-web-scraping/ # Web scraping
│
├── 02-data-preprocessing/             # 🔧 Data Cleaning & Transformation
│   ├── complete-case-analysis/        # Missing data analysis
│   ├── imputing-numerical-data/       # Numerical imputation
│   ├── handling-missing-categorical-data/ # Categorical imputation
│   ├── missing-indicator/             # Missing value indicators
│   ├── knn-imputer/                   # KNN imputation
│   ├── iterative-imputer/             # Iterative imputation
│   ├── handling-mixed-variables/      # Mixed data types
│   └── handling-date-and-time/        # Date/time features
│
├── 03-exploratory-data-analysis/      # 🔍 Statistical Analysis & EDA
│   ├── descriptive-stats/             # Statistical summaries
│   ├── univariate-analysis/           # Single variable analysis
│   ├── bivariate-analysis/            # Two variable analysis
│   ├── pandas-profiling/              # Automated EDA
│   ├── outlier-removal-zscore/        # Z-score outliers
│   ├── outlier-removal-iqr/           # IQR outliers
│   └── outlier-detection-percentiles/ # Percentile outliers
│
├── 04-feature-engineering/            # ⚙️ Feature Transformation
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
├── 05-machine-learning-algorithms/    # 🤖 ML Models & Algorithms
│   ├── simple-linear-regression/     # Linear regression
│   ├── multiple-linear-regression/   # Multiple regression
│   ├── polynomial-regression/         # Polynomial regression
│   ├── regularized-linear-models/     # Regularized models
│   ├── lasso-regression/              # Lasso regression
│   ├── elasticnet-regression/         # Elastic Net
│   ├── logistic-regression/           # Logistic regression
│   ├── logistic-regression-continued/ # Advanced logistic regression
│   ├── knn/                           # K-Nearest Neighbors
│   ├── svm/                           # Support Vector Machines
│   ├── random-forest/                 # Random Forest
│   ├── bagging/                       # Bagging ensemble
│   ├── adaboost/                      # AdaBoost
│   ├── gradient-boosting/             # Gradient Boosting
│   ├── voting-classifier/             # Voting classifiers
│   ├── stacking-and-blending/         # Stacking & blending
│   ├── kmeans/                        # K-Means clustering
│   ├── dbscan/                        # DBSCAN clustering
│   └── imbalanced-data/               # Handling imbalanced data
│
├── 06-model-evaluation/               # 📈 Evaluation & Metrics
│   ├── regression-metrics/            # Regression evaluation
│   ├── classification-metrics/        # Classification evaluation
│   └── roc-curve/                     # ROC curves & AUC
│
├── 07-advanced-topics/                # 🚀 Advanced Concepts
│   ├── gradient-descent/              # Gradient descent
│   ├── types-of-gradient-descent/     # GD variants
│   └── pca/                           # Principal Component Analysis
│
└── README.md                          # 📚 Documentation
```

---

## 🎯 Key Features

-   **📚 Comprehensive Coverage**: From basic data handling to advanced ML algorithms
-   **🔬 Hands-on Learning**: Interactive Jupyter notebooks with real datasets
-   **📊 Visual Learning**: Rich visualizations and animations (especially in gradient descent)
-   **🛠️ Practical Implementation**: Both from-scratch and scikit-learn implementations
-   **📈 Real-world Applications**: Industry-standard datasets and use cases
-   **🎨 Interactive Demos**: Streamlit apps and interactive visualizations
-   **📖 Structured Learning Path**: Clear progression from basics to advanced topics
-   **✅ Progress Tracking**: Built-in tracker to monitor your learning journey
-   **🔍 Section Guides**: Detailed README files for each section
-   **💡 Best Practices**: Industry-standard code and methodologies

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

We welcome contributions! Whether you're fixing bugs, adding new content, or improving documentation, your help makes this resource better for everyone.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** (follow our [Contributing Guidelines](CONTRIBUTING.md))
4. **Test your changes** (ensure notebooks run without errors)
5. **Commit your changes** (`git commit -m 'Add amazing feature'`)
6. **Push to the branch** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

### What You Can Contribute

- ✅ Add new machine learning topics or algorithms
- ✅ Improve existing implementations and explanations
- ✅ Fix bugs or enhance documentation
- ✅ Add more datasets or examples
- ✅ Improve code quality and best practices
- ✅ Add visualizations and interactive demos

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for detailed guidelines.

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

See the [`LICENSE`](LICENSE) file for details.

---

## 🙏 Acknowledgments

-   **[CampusX YouTube Playlist](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH)** - 100 Days of Machine Learning course that inspired this repository
-   Scikit-learn community for excellent documentation
-   Pandas team for powerful data manipulation tools
-   Matplotlib and Seaborn for visualization capabilities
-   The machine learning community for continuous learning and sharing

---

## 📚 Additional Resources

### Related Repositories
- **[100 Days of Deep Learning](https://github.com/devaaravmishra/100-days-of-deep-learning)** - Continue your AI journey with neural networks and deep learning
- **[LEARNING_ROADMAP.md](LEARNING_ROADMAP.md)** - See how ML and DL repos complement each other

### Learning Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)

### Recommended Books
- "Hands-On Machine Learning" by Aurélien Géron
- "Introduction to Statistical Learning" by James et al.
- "Pattern Recognition and Machine Learning" by Christopher Bishop

### Online Courses
- [CampusX YouTube Playlist](https://www.youtube.com/playlist?list=PLKnIA16_Rmvbr7zKYQuBfsVkjoLcJgxHH) - 100 Days of Machine Learning course
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)
- [Fast.ai Practical Machine Learning](https://course.fast.ai/)

## 🎓 Learning Tips

1. **Practice Regularly**: Consistency is key - even 30 minutes a day helps
2. **Code Along**: Don't just read - type and run the code yourself
3. **Experiment**: Modify code, try different parameters, break things
4. **Take Notes**: Document your learnings in the progress tracker
5. **Build Projects**: Apply what you learn to real-world problems
6. **Join Communities**: Engage with others learning ML

## ⚠️ Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError` when running notebooks
- **Solution**: Make sure you've installed all requirements: `pip install -r requirements.txt`

**Issue**: Jupyter notebook not starting
- **Solution**: Check if Jupyter is installed: `pip install jupyter notebook`

**Issue**: Kernel not found in Jupyter
- **Solution**: Install ipykernel: `pip install ipykernel` and register: `python -m ipykernel install --user`

**Issue**: Import errors with specific packages
- **Solution**: Check the package name and install it: `pip install package-name`

## 📊 Repository Statistics

- **Total Notebooks**: 97+
- **Topics Covered**: 50+
- **Data Files**: 40+
- **Estimated Learning Time**: 58-84 hours

---

**Happy Learning! 🎉**

Remember: The journey of a thousand miles begins with a single step. Start today! 🚀
