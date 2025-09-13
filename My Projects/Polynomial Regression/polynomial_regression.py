import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# create fake data
np.random.seed(2)
pageSpeed = np.random.normal(3,1,1000)
purchaseAmount = np.random.normal(50,10,1000) / pageSpeed

plt.scatter(pageSpeed,purchaseAmount)
plt.show()

# change page speed and purchase amount into numpy arrays and set a function on it.
x = np.array(pageSpeed)
y = np.array(purchaseAmount)
p4 = np.poly1d(np.polyfit(x,y,4))

# visualizing 
xp = np.linspace(0,7,100)
plt.scatter(x,y)
plt.plot(xp,p4(xp),c='r')
plt.show()

# caclulate r2 to see how the model is perfect.
r2 = r2_score(y,p4(x))
print(r2)