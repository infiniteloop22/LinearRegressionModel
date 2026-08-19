# Linear Regression Model

A Python implementation of a single-variable linear regression model without machine learning libraries.

## Features

- **Model:** Single-variable linear regression model: $f_{w,b}(x) = wx + b$.
- **Error Measure:** Standard squared error cost function ($J$).
- **Optimization:** Gradient functions and gradient descent algorithms.
- **Diabetes Study:** Machine learning logic specifically to continuous health data.

## Technologies Used

- **Language:** Python 3.8+
- **Array Modeling:** `numpy` (matrix operations)
- **Data Manipulation:** `pandas` (dataframes and cleaning)
- **Data Visualization:** `matplotlib` (graph plotting)

## Dataset

This project uses the **Diabetes Database** sourced from [Kaggle](https://www.kaggle.com/code/azratuni/diabetes-database-linear-regression/input). 

While many diabetes datasets are formatted for binary classification (logistic regression), this specific set isolates continuous values. This implementation maps **insulin** as the input feature ($x$) to predict **glucose** as the continuous target ($y$).

---

## Technical Notes

### Mathematical Notation
- **$x$**: Input feature variable (e.g., Insulin level).
- **$y$**: Output target variable (e.g., Glucose level).
- **$m$**: Total number of training examples.
- **$(x^{(i)}, y^{(i)})$**: The $i^{th}$ specific training example index.

### 1. Cost Function
Used to measure the accuracy of the model. It calculates the difference between predictions ($\hat{y}$) and actual targets ($y$) using the **Squared Error Cost Function**:

$$J(w,b) = \frac{1}{2m} \sum\limits_{i=1}^{m}(f_{w,b} (x^{(i)}) - y^{(i)})^{2}$$

*The core objective of the algorithm is to discover values for $w$ and $b$ that minimize the output of $J$.*

### 2. Gradient Function
Computes the partial derivatives of the cost function with respect to weights ($w$) and biases ($b$) across each iteration:

$$\frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}$$

$$\frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})$$

### 3. Gradient Descent
Repeats optimization steps automatically until the cost function converges to its lowest possible value using the learning rate ($\alpha$):

$$w = w - \alpha \cdot \frac{\partial J(w,b)}{\partial w}$$

$$b = b - \alpha \cdot \frac{\partial J(w,b)}{\partial b}$$

---