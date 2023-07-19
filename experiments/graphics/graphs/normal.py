import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_normal(mean, std_dev):
    # Parameters for Normal distribution
    mu = mean  # mean
    sigma = std_dev  # standard deviation

    # Create a range of x values
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)

    # Generate the y values from the pdf
    y_pdf = norm.pdf(x, mu, sigma)
    y_cdf = norm.cdf(x, mu, sigma)  # calculate CDF

    # Create the subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the PDF on the first subplot
    axs[0].plot(x, y_pdf, label=f'PDF: Normal (mu={mu}, sigma={sigma})')
    axs[0].set_title('Normal Distribution: PDF')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('Probability')
    axs[0].legend()
    axs[0].grid()

    # Plot the CDF on the second subplot
    axs[1].plot(x, y_cdf, label=f'CDF: Normal (mu={mu}, sigma={sigma})')
    axs[1].set_title('Normal Distribution: CDF')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Cumulative Probability')
    axs[1].legend()
    axs[1].grid()

    # Adjust the layout and display the plots
    plt.tight_layout()
    plt.show()


def normal_cdf(demand, mu, sigma):
    """Calculate the CDF of the Normal distribution at a certain demand value."""
    return norm.cdf(demand, mu, sigma)


def normal_quantile(prob, mu, sigma):
    """Calculate the quantile (inverse CDF) of the Normal distribution at a certain probability."""
    return norm.ppf(prob, mu, sigma)


print(normal_cdf(2, 0, 1))

