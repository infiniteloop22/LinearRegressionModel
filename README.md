# Linear/Multiple Regression and Logistic Regression Models

A Python implementation of linear and logistic regression classification models without using machine learning libraries to implement said models.

*Exception:* `scikit-learn` was only used for train-test data splitting.

## Features
- **Linear Models:** Single-variable linear regression and multiple linear regression using the closed-form **Normal Equation** \(\theta = (X^T X)^{-1} X^T y\).
- **Logistic Regression:** Classification model using the sigmoid function and gradient descent optimization.
- **Error & Optimization:** Implements standard squared error cost for regression and cross-entropy log loss for classification. For classification, the training and test sets were normalized with the z-score formula to prevent data leakage.

## Technologies Used

- **Language:** Python 3.8+
- **Array Modeling:** `numpy` (matrix operations)
- **Data Manipulation:** `pandas` (dataframes and cleaning)
- **Data Visualization:** `matplotlib` (graph plotting)

## Dataset

This project uses the **Diabetes Database** sourced from [Kaggle](https://www.kaggle.com/code/azratuni/diabetes-database-linear-regression/input). 

---

## Technical & Mathematical Summary
- **Linear Regression Cost & Gradients:** Uses squared error $J(w,b) = \frac{1}{2m} \sum (f_{w,b}(x^{(i)}) - y^{(i)})^2$ and gradient descent parameter updates.
- **Multiple Regression (Normal Equation):** Uses multiple linear regression in 1 step by taking the inverse of the matrix.
- **Logistic Regression:** Uses the sigmoid function to output predictions $g(z) = \frac{1}{1 + e^{-z}}$ and uses an iterative approach (gradient descent) since no closed-form solution exists. Uses cross-entropy log loss to evaluate the model's errors.
