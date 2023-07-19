import numpy as np
import matplotlib.pyplot as plt

# Parameters for Normal distribution
mu = 0  # mean
sigma = 1  # standard deviation

# Parameters for Cauchy distribution
x0 = 0  # location parameter
gamma = 5  # scale parameter

# Number of samples
size = 100

# Generate samples from a Normal distribution
normal_samples = np.random.normal(mu, sigma, size)

# Generate samples from a Cauchy distribution
cauchy_samples = np.random.standard_cauchy(size) * gamma + x0

# Create a figure and a set of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Plot histogram of the Normal samples on the first subplot
axs[0, 0].hist(normal_samples, bins=100, range=(-10, 10))
axs[0, 0].set_title('Normal Distribution - Histogram')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('Frequency')

# Plot histogram of the Cauchy samples on the second subplot
axs[1, 0].hist(cauchy_samples, bins=100, range=(-10, 10))
axs[1, 0].set_title('Cauchy Distribution - Histogram')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('Frequency')

# Plot boxplot of the Normal samples on the third subplot
axs[0, 1].boxplot(normal_samples)
axs[0, 1].set_title('Normal Distribution - Boxplot')

# Plot boxplot of the Cauchy samples on the fourth subplot
axs[1, 1].boxplot(cauchy_samples)
axs[1, 1].set_title('Cauchy Distribution - Boxplot')

# Adjust the layout and display the plots
plt.tight_layout()
plt.show()
