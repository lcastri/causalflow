import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Generate example data for X and Y
np.random.seed(0)
X_data = np.random.normal(loc=0, scale=1, size=1000)
Y_data = np.random.normal(loc=0, scale=1, size=1000)

# Concatenate X and Y data for KDE
data = np.column_stack((X_data, Y_data))

# Define range for plotting the joint density
x_range = np.linspace(-3, 3, 100)
y_range = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_range, y_range)
XY = np.column_stack([X.ravel(), Y.ravel()])

# Compute the joint density using KDE
kde = KernelDensity(bandwidth=0.2)
kde.fit(data)
log_density = kde.score_samples(XY)
density = np.exp(log_density)
density = density.reshape(X.shape)

# Plot the joint density
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, density, cmap='viridis')
plt.colorbar(label='Density')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Joint Density of X and Y')
plt.show()
