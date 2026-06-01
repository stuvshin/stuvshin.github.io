+++
date = '2026-06-02T04:12:15+09:00'
draft = false
title = 'Test 00'
math = 'true'
+++

# Introduction to Linear Regression

Linear Regression is a foundational machine learning algorithm used to model the relationship between a dependent scalar variable $y$ and one or more explanatory variables $X$.

---

## Mathematical Formulation

The goal of linear regression is to find a linear function that predicts the dependent variable $y$ given the input features $\mathbf{x}$.

The linear model is expressed as:

$$y = \mathbf{w}^T \mathbf{x} + b$$

Where:

* $\mathbf{w}$ is the weight vector (parameters).
* $\mathbf{x}$ is the feature vector.
* $b$ is the bias term (intercept).

### The Cost Function (Mean Squared Error)

To find the optimal parameters $\mathbf{w}$ and $b$, we minimize the Mean Squared Error (MSE), which measures the average squared difference between actual values $y_i$ and predicted values $\hat{y}_i$:

$$J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

Where $m$ is the total number of training examples.

---

## Parameter Optimization: Gradient Descent

We optimize the cost function using Gradient Descent. The parameters are updated iteratively by moving in the opposite direction of the gradient.

The gradient updates for weights and bias are calculated using the partial derivatives of the cost function:

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) \mathbf{x}^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)$$

### Update Rules:

$$\mathbf{w} := \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

Where $\alpha$ is the learning rate.

---

## Python Implementation

Below is a clean NumPy implementation of Linear Regression built from scratch using gradient descent.

```python
import numpy as np

class LinearRegressionGradientDescent:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        # Initialize parameters to zeros
        self.weights = np.zeros(num_features)
        self.bias = 0.0

        # Gradient Descent Loop
        for _ in range(self.iterations):
            # 1. Compute predicted values
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # 2. Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # 3. Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example Execution
if __name__ == "__main__":
    # Generate simple synthetic data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 5, 4, 5])

    # Train model
    model = LinearRegressionGradientDescent(learning_rate=0.01, iterations=500)
    model.fit(X, y)

    # Test predictions
    predictions = model.predict(X)
    print("Trained Weights:", model.weights)
    print("Trained Bias:", model.bias)
    print("Predictions:", predictions)

```