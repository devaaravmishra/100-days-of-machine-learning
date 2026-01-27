# 📈 06 - Model Evaluation

This section covers comprehensive model evaluation metrics and techniques for both regression and classification tasks.

## Topics Covered

### Regression Metrics
- **Regression Metrics** (`regression-metrics/`) - MAE, MSE, RMSE, R², Adjusted R², MAPE

### Classification Metrics
- **Classification Metrics** (`classification-metrics/`) - Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **ROC Curve and AUC** (`roc-curve/`) - Receiver Operating Characteristic curves, Area Under Curve

## Learning Objectives

By the end of this section, you will be able to:
- ✅ Choose appropriate metrics for your problem
- ✅ Interpret regression metrics correctly
- ✅ Understand precision, recall, and F1-score tradeoffs
- ✅ Use ROC curves to evaluate classifiers
- ✅ Avoid common evaluation pitfalls

## Prerequisites

- Completed `05-machine-learning-algorithms/`
- Understanding of train/test split and cross-validation

## Estimated Time

**4-6 hours**

## Key Concepts

### Regression Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference
- **MSE (Mean Squared Error)**: Penalizes large errors more
- **RMSE (Root Mean Squared Error)**: In same units as target
- **R² (R-squared)**: Proportion of variance explained

### Classification Metrics
- **Accuracy**: Overall correctness (can be misleading for imbalanced data)
- **Precision**: Of predicted positives, how many are actually positive
- **Recall**: Of actual positives, how many were predicted correctly
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Model's ability to distinguish between classes

## When to Use Which Metric

- **Regression**: Use RMSE for general cases, MAE if outliers are a concern
- **Classification**: 
  - Balanced data: Accuracy, F1-Score
  - Imbalanced data: Precision, Recall, F1-Score, ROC-AUC
  - Medical/High-stakes: Focus on Recall (sensitivity)

## Next Steps

After completing this section, move to:
- **`07-advanced-topics/`** - Explore advanced optimization and dimensionality reduction

---

**Happy Learning!** 🚀
