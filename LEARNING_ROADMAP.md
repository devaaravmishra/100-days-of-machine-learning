# 🗺️ Complete Learning Roadmap: ML → Deep Learning

This document shows how the **100 Days of Machine Learning** and **100 Days of Deep Learning** repositories complement each other to create a comprehensive AI/ML learning journey.

## 🎯 Overview

- **100 Days of Machine Learning**: Focuses on traditional ML algorithms, data preprocessing, and feature engineering
- **100 Days of Deep Learning**: Builds on ML fundamentals to cover neural networks, CNNs, RNNs, transformers, and advanced deep learning

**Together, they form a complete path from beginner to advanced AI practitioner!**

---

## 📊 Learning Path Integration

### Phase 1: Foundations (Start Here!)

#### Traditional ML Path (This Repository)
1. **`01-data-handling/`** - Learn to work with data from various sources
2. **`02-data-preprocessing/`** - Master data cleaning and preprocessing
3. **`03-exploratory-data-analysis/`** - Understand your data deeply
4. **`04-feature-engineering/`** - Transform features for ML models

#### Deep Learning Path (Deep Learning Repo)
- **Days 1-7**: Python, NumPy, Pandas, Visualization (foundational skills)
- **Days 8-14**: Mathematics for ML (linear algebra, calculus, statistics)
- **Days 15-20**: Data preprocessing and EDA (overlaps with ML repo)

**Recommendation**: Complete ML repo sections 01-04 first, then review DL repo Days 15-20 for reinforcement.

---

### Phase 2: Machine Learning Fundamentals

#### Traditional ML Path (This Repository)
5. **`05-machine-learning-algorithms/`** - Comprehensive ML algorithms:
   - Linear models (regression, classification)
   - Tree-based methods (Random Forest, Gradient Boosting)
   - Ensemble methods (Bagging, Stacking)
   - Clustering (K-Means, DBSCAN)
   - SVM, KNN, and more

6. **`06-model-evaluation/`** - Evaluation metrics and techniques

#### Deep Learning Path (Deep Learning Repo)
- **Days 21-35**: ML Fundamentals
  - Days 21-28: Supervised learning (Linear/Logistic Regression, Trees, SVM, KNN)
  - Days 29-35: Advanced ML (Regularization, Cross-validation, Gradient Descent, PCA)

**Recommendation**: 
- **Option A**: Complete ML repo sections 05-06 first for deep understanding, then review DL repo Days 21-35
- **Option B**: Do both in parallel - ML repo for hands-on practice, DL repo for theory

---

### Phase 3: Advanced ML & Optimization

#### Traditional ML Path (This Repository)
7. **`07-advanced-topics/`** - Advanced concepts:
   - Gradient Descent (from scratch)
   - Types of Gradient Descent
   - Principal Component Analysis (PCA)

#### Deep Learning Path (Deep Learning Repo)
- **Days 32**: Gradient Descent & Optimization
- **Days 34**: Dimensionality Reduction (PCA, t-SNE)

**Recommendation**: Complete ML repo section 07, then move to DL repo for neural networks.

---

### Phase 4: Neural Networks & Deep Learning

#### Deep Learning Path (Deep Learning Repo) - **NEW CONTENT**
- **Days 36-50**: Neural Networks Basics
  - Perceptrons, activation functions
  - Feedforward networks
  - Backpropagation
  - TensorFlow/Keras, PyTorch

- **Days 51-70**: Deep Learning
  - Convolutional Neural Networks (CNNs)
  - Recurrent Neural Networks (RNNs, LSTM, GRU)
  - Transfer Learning

- **Days 71-85**: Advanced Deep Learning
  - Transformers & BERT
  - Generative Models (GANs, Autoencoders)
  - NLP with Deep Learning

- **Days 86-100**: Production & Advanced Topics
  - MLOps & Deployment
  - Model Optimization
  - Ethics & Explainable AI

**Recommendation**: After completing ML repo, dive into DL repo Days 36+ for neural networks.

---

## 🔗 Topic Mapping & Cross-References

### Data Preprocessing
| ML Repo | DL Repo | Notes |
|---------|---------|-------|
| [`02-data-preprocessing/complete-case-analysis/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/02-data-preprocessing/complete-case-analysis) | Day 15 | Both cover missing data |
| [`02-data-preprocessing/imputing-numerical-data/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/02-data-preprocessing/imputing-numerical-data) | Day 15 | ML repo has more detailed imputation methods |
| [`02-data-preprocessing/handling-missing-categorical-data/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/02-data-preprocessing/handling-missing-categorical-data) | Day 15 | ML repo covers categorical imputation |
| [`03-exploratory-data-analysis/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/03-exploratory-data-analysis) | Day 18 | ML repo has comprehensive EDA |

### Feature Engineering
| ML Repo | DL Repo | Notes |
|---------|---------|-------|
| [`04-feature-engineering/standardization/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/04-feature-engineering/standardization) | Day 16 | Both cover scaling |
| [`04-feature-engineering/one-hot-encoding/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/04-feature-engineering/one-hot-encoding) | Day 16 | ML repo has detailed encoding techniques |
| [`04-feature-engineering/sklearn-pipelines/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/04-feature-engineering/sklearn-pipelines) | Day 20 | ML repo focuses on pipelines |

### Machine Learning Algorithms
| ML Repo | DL Repo | Notes |
|---------|---------|-------|
| [`05-machine-learning-algorithms/simple-linear-regression/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/05-machine-learning-algorithms/simple-linear-regression) | Day 22 | ML repo has from-scratch implementations |
| [`05-machine-learning-algorithms/logistic-regression/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/05-machine-learning-algorithms/logistic-regression) | Day 23 | ML repo covers advanced logistic regression |
| [`05-machine-learning-algorithms/random-forest/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/05-machine-learning-algorithms/random-forest) | Day 24 | ML repo has detailed Random Forest analysis |
| [`05-machine-learning-algorithms/svm/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/05-machine-learning-algorithms/svm) | Day 25 | Both cover SVM |
| [`05-machine-learning-algorithms/knn/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/05-machine-learning-algorithms/knn) | Day 26 | Both cover KNN |
| [`05-machine-learning-algorithms/kmeans/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/05-machine-learning-algorithms/kmeans) | Day 33 | Both cover clustering |

### Model Evaluation
| ML Repo | DL Repo | Notes |
|---------|---------|-------|
| [`06-model-evaluation/classification-metrics/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/06-model-evaluation/classification-metrics) | Day 28 | ML repo has comprehensive metrics |
| [`06-model-evaluation/roc-curve/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/06-model-evaluation/roc-curve) | Day 28 | ML repo covers ROC-AUC in detail |

### Advanced Topics
| ML Repo | DL Repo | Notes |
|---------|---------|-------|
| [`07-advanced-topics/gradient-descent/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/07-advanced-topics/gradient-descent) | Day 32 | ML repo has from-scratch implementations |
| [`07-advanced-topics/pca/`](https://github.com/devaaravmishra/100-days-of-machine-learning/tree/main/07-advanced-topics/pca) | Day 34 | Both cover PCA |

---

## 🎓 Recommended Learning Paths

### Path 1: Complete ML First, Then Deep Learning (Recommended for Beginners)

```
Week 1-4:   ML Repo Sections 01-04 (Data handling, preprocessing, EDA, features)
Week 5-10:  ML Repo Sections 05-06 (ML algorithms, evaluation)
Week 11-12: ML Repo Section 07 (Advanced topics)
Week 13+:   DL Repo Days 36-100 (Neural networks and deep learning)
```

**Total Time**: ~4-6 months (2-3 hours/day)

### Path 2: Parallel Learning (For Those with Some Background)

```
Week 1-2:   ML Repo 01-04 + DL Repo Days 1-14 (Foundations)
Week 3-6:   ML Repo 05-06 + DL Repo Days 21-35 (ML fundamentals)
Week 7-8:   ML Repo 07 + DL Repo Day 32, 34 (Advanced ML)
Week 9+:    DL Repo Days 36-100 (Deep learning)
```

**Total Time**: ~3-4 months (2-3 hours/day)

### Path 3: Deep Learning Focus (For Those Who Know Traditional ML)

```
Week 1-2:   Review ML Repo 01-07 (Quick review)
Week 3+:    DL Repo Days 36-100 (Focus on neural networks)
```

**Total Time**: ~2-3 months (2-3 hours/day)

---

## 🔄 How to Navigate Between Repositories

### When Learning Data Preprocessing:
1. Start with **ML Repo** `02-data-preprocessing/` for comprehensive coverage
2. Review **DL Repo** Day 15-16 for reinforcement
3. Practice with both repositories' examples

### When Learning ML Algorithms:
1. Use **ML Repo** `05-machine-learning-algorithms/` for detailed implementations
2. Reference **DL Repo** Days 21-35 for theory and additional context
3. Compare implementations between both repos

### When Learning Neural Networks:
1. Complete **ML Repo** `07-advanced-topics/gradient-descent/` first
2. Move to **DL Repo** Days 36+ for neural networks
3. Apply ML preprocessing knowledge to deep learning projects

---

## 📚 Repository Links

- **100 Days of Machine Learning**: https://github.com/devaaravmishra/100-days-of-machine-learning
- **100 Days of Deep Learning**: https://github.com/devaaravmishra/100-days-of-deep-learning

---

## 💡 Key Differences

| Aspect | ML Repo | DL Repo |
|--------|---------|---------|
| **Focus** | Traditional ML algorithms | Neural networks & deep learning |
| **Depth** | Deep dive into each algorithm | Broader coverage, builds on ML |
| **Implementation** | From scratch + scikit-learn | TensorFlow, PyTorch, Keras |
| **Notebooks** | 97+ detailed notebooks | Day-by-day curriculum |
| **Best For** | Understanding ML fundamentals | Building on ML to learn DL |

---

## 🎯 Learning Goals

### After ML Repo, You Should:
- ✅ Understand traditional ML algorithms deeply
- ✅ Know how to preprocess and engineer features
- ✅ Be able to evaluate models properly
- ✅ Implement algorithms from scratch

### After DL Repo, You Should:
- ✅ Understand neural networks and deep learning
- ✅ Build CNNs, RNNs, and transformers
- ✅ Use TensorFlow/PyTorch effectively
- ✅ Deploy models to production

### After Both Repos, You Should:
- ✅ Have a complete understanding of ML and DL
- ✅ Be able to choose the right approach for any problem
- ✅ Build end-to-end ML/DL pipelines
- ✅ Be ready for real-world AI projects

---

## 🚀 Next Steps

1. **Complete ML Repo** (if you haven't already)
2. **Review this roadmap** to understand the connection
3. **Start DL Repo** from Day 36 (or Day 1 if you want a refresher)
4. **Build projects** combining knowledge from both repos

---

**Happy Learning!** 🎉
