import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

# create fake data
np.random.seed(2)
pageSpeed = np.random.normal(3, 1, 1000)
purchaseAmount = np.random.normal(50, 10, 1000) / pageSpeed

# create a data frame of data
data = pd.DataFrame({'pageSpeed': pageSpeed, 'purchaseAmount': purchaseAmount})
print(data.describe())

# show the plot of our data
plt.figure(figsize=(8,5))
sns.scatterplot(x='pageSpeed', y='purchaseAmount', data=data, alpha=0.6)
plt.xlabel("Page Speed (seconds)")
plt.ylabel("Purchase Amount ($)")
plt.title("Scatter plot of Page Speed vs Purchase Amount")
plt.show()

degrees = [1, 2, 3, 4, 5]
r2_scores = []

# turn page speed into a 2D numpy array
x = pageSpeed.reshape(-1,1)
y = purchaseAmount

plt.figure(figsize=(10,6))
sns.scatterplot(x=pageSpeed, y=purchaseAmount, alpha=0.3)
# plt.show()

# create 100 points between max and min amount of page speed.
xp = np.linspace(min(pageSpeed), max(pageSpeed), 100).reshape(-1,1)

# examine each degree for the function we set to predict.
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    x_poly = poly.fit_transform(x)
    model = LinearRegression().fit(x_poly,y)
    y_pred = model.predict(x_poly)
    r2 = r2_score(y, y_pred)
    r2_scores.append(r2)
    xp_poly = poly.transform(xp)
    yp = model.predict(xp_poly)
    plt.plot(xp, yp, label=f'degree {d} | R2={r2:.3f}')

plt.xlabel("Page Speed (seconds)")
plt.ylabel("Purchase Amount ($)")
plt.title("Polynomial Fit Comparison")
plt.legend()
plt.show()

# use Kfold to choose the best model to predict
kf = KFold(n_splits=5, shuffle=True, random_state=2)
cv_scores = []
for d in degrees:
    poly = PolynomialFeatures(degree=d)
    x_poly = poly.fit_transform(x)
    model = LinearRegression()
    scores = cross_val_score(model, x_poly, y, cv=kf, scoring='r2')
    cv_scores.append(np.mean(scores))

best_degree = degrees[np.argmax(cv_scores)]
print(f'Best polynomial degree according to 5-fold CV: {best_degree}')
print("Cross-validated R² scores:", ", ".join(f"{score:.3f}" for score in cv_scores))
