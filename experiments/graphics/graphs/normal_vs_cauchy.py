import numpy as np
import matplotlib.pyplot as plt

# Parameters for Normal distribution
mu = 0  # mean
sigma = 1  # standard deviation

# Parameters for Cauchy distribution
x0 = 0  # location parameter
gamma = 1  # scale parameter

# Number of samples
size = 10000

# Generate samples from a Normal distribution
normal_samples = np.random.normal(mu, sigma, size)

# Generate samples from a Cauchy distribution
cauchy_samples = np.random.standard_cauchy(size) * gamma + x0

# Create a figure and a set of subplots for Histograms
fig_hist, axs_hist = plt.subplots(2, 1, figsize=(10, 10))

# Plot histogram of the Normal samples on the first subplot
axs_hist[0].hist(normal_samples, bins=100, range=(-10, 10))
axs_hist[0].set_title('Normal Distribution - Histogram')
axs_hist[0].set_xlabel('x')
axs_hist[0].set_ylabel('Frequency')
axs_hist[0].annotate(f'mean={mu}, sd={sigma}', xy=(0.05, 0.95), xycoords='axes fraction')

# Plot histogram of the Cauchy samples on the second subplot
axs_hist[1].hist(cauchy_samples, bins=100, range=(-10, 10))
axs_hist[1].set_title('Cauchy Distribution - Histogram')
axs_hist[1].set_xlabel('x')
axs_hist[1].set_ylabel('Frequency')
axs_hist[1].annotate(f'x0={x0}, gamma={gamma}', xy=(0.05, 0.95), xycoords='axes fraction')

# Create a figure and a set of subplots for Boxplots
fig_box, axs_box = plt.subplots(1, 2, figsize=(10, 5))

# Plot boxplot of the Normal samples on the first subplot
axs_box[0].boxplot(normal_samples)
axs_box[0].set_title('Normal Distribution - Boxplot')
axs_box[0].annotate(f'mean={mu}, sd={sigma}', xy=(0.05, 0.9), xycoords='axes fraction')

# Plot boxplot of the Cauchy samples on the second subplot
axs_box[1].boxplot(cauchy_samples)
axs_box[1].set_title('Cauchy Distribution - Boxplot')
axs_box[1].annotate(f'x0={x0}, gamma={gamma}', xy=(0.05, 0.9), xycoords='axes fraction')

# Adjust the layout and display the plots
plt.tight_layout()
plt.show()
