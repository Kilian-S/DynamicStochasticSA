import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy


def plot_cauchy(location, gamma):
    # Parameters for Cauchy distribution
    x0 = location  # location parameter
    gamma = gamma

    # Create a range of x values
    x = np.linspace(-5, 5, 1000)

    # Generate the y values from the pdf
    y_pdf = cauchy.pdf(x, x0, gamma)
    y_cdf = cauchy.cdf(x, x0, gamma)  # calculate CDF

    # Create the subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the PDF on the first subplot
    axs[0].plot(x, y_pdf, label=f'PDF: Cauchy (x0={x0}, gamma={gamma})')
    axs[0].set_title('Cauchy Distribution: PDF')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('Probability')
    axs[0].legend()
    axs[0].grid()

    # Plot the CDF on the second subplot
    axs[1].plot(x, y_cdf, label=f'CDF: Cauchy (x0={x0}, gamma={gamma})')
    axs[1].set_title('Cauchy Distribution: CDF')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Cumulative Probability')
    axs[1].legend()
    axs[1].grid()

    # Adjust the layout and display the plots
    plt.tight_layout()
    plt.show()



def cauchy_cdf(demand, x0, gamma):
    """Calculate the CDF of the Cauchy distribution at a certain demand value."""
    return cauchy.cdf(demand, x0, gamma)

def cauchy_quantile(prob, x0, gamma):
    """Calculate the quantile (inverse CDF) of the Cauchy distribution at a certain probability."""
    return cauchy.ppf(prob, x0, gamma)


def plot_cauchy_multiple_gammas(x0):
    # Create a range of x values
    x = np.linspace(-10, 10, 10000)

    # Create the plot
    plt.figure(figsize=(10, 5))

    # Iterate over different gammas
    for gamma in np.arange(0.25, 1.01, 0.25):
        # Generate the y values from the pdf
        y_pdf = cauchy.pdf(x, x0, gamma)

        # Plot the PDF
        plt.plot(x, y_pdf, label=f'Cauchy PDF (x0={x0}, gamma={gamma:.2f})')

    plt.title('Cauchy Distributions with Varying Gammas')
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid()
    plt.show()


location = 0
gamma = 1

plot_cauchy(location, gamma)








