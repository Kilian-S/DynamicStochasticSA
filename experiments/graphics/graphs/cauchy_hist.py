import numpy as np
import matplotlib.pyplot as plt

# Parameters for Cauchy distribution
x0 = 100  # location parameter
gamma = 50  # scale parameter

# Number of samples
size = 10000

# Generate samples from a Cauchy distribution
samples = np.random.standard_cauchy(size) * gamma + x0

# Plot histogram of the samples
plt.hist(samples, bins=100, range=(-10, 200))
plt.show()
