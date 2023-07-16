import numpy as np
import matplotlib.pyplot as plt

# Parameters for Normal distribution
mu = 100  # mean
sigma = 1  # standard deviation

# Number of samples
size = 10000

# Generate samples from a Normal distribution
samples = np.random.normal(mu, sigma, size)

# Plot histogram of the samples
plt.hist(samples, bins=100)
plt.show()
