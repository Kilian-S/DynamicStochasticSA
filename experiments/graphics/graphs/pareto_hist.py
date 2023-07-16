import numpy as np
import matplotlib.pyplot as plt

# Shape parameter
b = 2.62

# Scale parameter (essentially a shift)
scale = 1000

# Number of samples
size = 10000

# Generate samples from a Pareto distribution
samples = np.random.pareto(b, size) + scale

# Plot histogram of the samples
plt.hist(samples, bins=100)
plt.show()