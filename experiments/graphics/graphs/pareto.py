import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pareto


def plot_pareto(b, scale):
    # Create a range of x values
    x = np.linspace(1, 5, 1000)

    # Generate the y values from the pdf
    y_pdf = pareto.pdf(x, b, scale=scale)
    y_cdf = pareto.cdf(x, b, scale=scale)  # calculate CDF

    # Create the subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the PDF on the first subplot
    axs[0].plot(x, y_pdf, label=f'PDF: Pareto (b={b}, scale={scale})')
    axs[0].set_title('Pareto Distribution: PDF')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('Probability')
    axs[0].legend()
    axs[0].grid()

    # Plot the CDF on the second subplot
    axs[1].plot(x, y_cdf, label=f'CDF: Pareto (b={b}, scale={scale})')
    axs[1].set_title('Pareto Distribution: CDF')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('Cumulative Probability')
    axs[1].legend()
    axs[1].grid()

    # Adjust the layout and display the plots
    plt.tight_layout()
    plt.show()


def pareto_cdf(x, b, scale):
    """Calculate the CDF of the Pareto distribution at a certain value."""
    return pareto.cdf(x, b, scale=scale)

def pareto_quantile(prob, b, scale):
    """Calculate the quantile (inverse CDF) of the Pareto distribution at a certain probability."""
    return pareto.ppf(prob, b, scale=scale)


def plot_pareto_multiple_scales(scale):
    # Create a range of x values
    x = np.linspace(-1, 10, 10000)

    # Create the plot
    plt.figure(figsize=(10, 5))

    # Iterate over different scales
    for b in np.arange(1, 5, 1):
        # Generate the y values from the pdf
        y_pdf = pareto.pdf(x, b, scale)

        # Plot the PDF
        plt.plot(x, y_pdf, label=f'Pareto PDF (b={b}, scale={scale})')

    plt.title('Pareto Distributions with Varying Scales')
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid()
    plt.show()


scale = 0

plot_pareto_multiple_scales(scale)
