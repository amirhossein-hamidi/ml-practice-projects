import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


pageSpeed = np.random.normal(3, 1, 1000)
purchaseAmount = 100 - (pageSpeed + np.random.normal(0, 0.1, 1000))


X_train, X_test, y_train, y_test = train_test_split(pageSpeed, purchaseAmount, test_size=0.2, random_state=42)


def run_linear_regression(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept, r_value, p_value, std_err


slope, intercept, r_value, p_value, std_err = run_linear_regression(X_train, y_train)
print(f"Training R^2: {r_value**2:.4f}")


def predict(x, slope, intercept):
    return slope * x + intercept


y_pred = predict(X_test, slope, intercept)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")


plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train, y=y_train, label='Training Data')
sns.lineplot(x=np.sort(X_train), y=predict(np.sort(X_train), slope, intercept), color='red', label='Fit Line')
plt.title('Training Data and Fitted Regression Line')
plt.xlabel('Page Speed')
plt.ylabel('Purchase Amount')
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test, y=y_test, label='Test Data')
sns.scatterplot(x=X_test, y=y_pred, color='red', label='Predicted')
plt.title('Test Data vs Predicted Values')
plt.xlabel('Page Speed')
plt.ylabel('Purchase Amount')
plt.legend()
plt.show()


residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot (Test Data)')
plt.xlabel('Predicted Purchase Amount')
plt.ylabel('Residuals (Error)')
plt.show()
