# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Define the x-values (iterations)
x = np.linspace(0, 1000, 1000)

# Define the cooling schedules
y1 = 100 / (x + 1)
y2 = 100 - ((100 * x) / 1000)
y3 = 100 / (1 + np.exp(0.1 * (x - 0.5 * 1000)))

# Create a figure with 3 subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot the first cooling schedule
axs[0].plot(x, y1, color='#FF1F5B')
axs[0].set_title('Exponential Decay')
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('Temperature')
axs[0].text(100, 20, r'$f(x) = \frac{100}{x + 1}$', fontsize=15, color='#FF1F5B')

# Plot the second cooling schedule
axs[1].plot(x, y2, color='#00CD6C')
axs[1].set_title('Linear Decay')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('Temperature')
axs[1].text(100, 20, r'$f(x) = 100 - \frac{100x}{1000}$', fontsize=15, color='#00CD6C')

# Plot the third cooling schedule
axs[2].plot(x, y3, color='#009ADE')
axs[2].set_title('Sigmoid Decay')
axs[2].set_xlabel('Iterations')
axs[2].set_ylabel('Temperature')
axs[2].text(15, 20, r'$f(x) = \frac{100}{1 + e^{0.1(x - 0.5 \times 1000)}}$', fontsize=15, color='#009ADE')

# Adjust layout
plt.tight_layout()
plt.show()
