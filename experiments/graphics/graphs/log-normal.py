import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

def plot_lognorm(mu, sigma):
    # Parameters for Log-normal distribution
    s = sigma  # shape parameter
    scale = np.exp(mu)  # scale parameter

    # Create a range of x values
    x = np.linspace(0, 25, 1000)

    # Generate the y values from the pdf
    y_pdf = lognorm.pdf(x, s, scale=scale)
    y_cdf = lognorm.cdf(x, s, scale=scale)  # calculate CDF

    # Create the subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the PDF on the first subplot
    axs[0].plot(x, y_pdf, label=f'PDF: Log-normal (mu={mu}, sigma={sigma})')
    axs[0].set_title('Log-normal Distribution: PDF')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('Probability')
    axs[0].legend()
    axs[0].grid()

    # Plot the CDF on the second subplot
    axs[1].plot(x, y_cdf, label=f'CDF: Log-normal (mu={mu}, sigma={sigma})')
    axs[1].set_title('Log-normal Distribution: CDF')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Cumulative Probability')
    axs[1].legend()
    axs[1].grid()

    # Adjust the layout and display the plots
    plt.tight_layout()
    plt.show()

def lognorm_cdf(demand, mu, sigma):
    """Calculate the CDF of the Log-normal distribution at a certain demand value."""
    s = sigma  # shape parameter
    scale = np.exp(mu)  # scale parameter
    return lognorm.cdf(demand, s, scale=scale)

def lognorm_quantile(prob, mu, sigma):
    """Calculate the quantile (inverse CDF) of the Log-normal distribution at a certain probability."""
    s = sigma  # shape parameter
    scale = np.exp(mu)  # scale parameter
    return lognorm.ppf(prob, s, scale=scale)


def plot_lognorm_multiple_sigmas(mu):
    # Parameters for Log-normal distribution
    scale = np.exp(mu)  # scale parameter

    # Create a range of x values
    x = np.linspace(0.01, 25, 10000)

    # Create the plot
    plt.figure(figsize=(10, 5))

    # Iterate over different sigmas
    for sigma in np.arange(0.5, 2, 0.25):
        s = sigma  # shape parameter

        # Generate the y values from the pdf
        y_pdf = lognorm.pdf(x, s, scale=scale)

        # Plot the PDF
        plt.plot(x, y_pdf, label=f'Log-normal PDF (mu={mu}, sigma={sigma:.2f})')

    plt.title('Log-normal Distributions with Varying Sigmas')
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid()
    plt.show()


print(lognorm_cdf(2, 0, 0.5))
plot_lognorm_multiple_sigmas(1)


