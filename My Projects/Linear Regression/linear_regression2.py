import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
ad_spend = np.random.normal(50, 10, 500)
social_media_ads = np.random.normal(20, 5, 500)
sales = 3*ad_spend + 2*social_media_ads + np.random.normal(0,5,500)

X = np.column_stack((ad_spend, social_media_ads))
y = sales

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

r2_train = model.score(X_train_scaled, y_train)
print(f"Training R^2: {r2_train:.3f}")

y_pred = model.predict(X_test_scaled)

MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)
print(f"Test MAE: {MAE:.3f}")
print(f"Test MSE: {MSE:.3f}")
print(f"Test R^2: {r2_test:.3f}")

plt.figure(figsize=(10,6))
sns.scatterplot(x=X_test[:,0], y=y_test, label='Test Data')
sns.scatterplot(x=X_test[:,0], y=y_pred, color='red', label='Predicted')
plt.title('Sales vs Ad Spend (Test Data)')
plt.xlabel('Ad Spend')
plt.ylabel('Sales')
plt.legend()
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot (Test Data)')
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals')
plt.show()
