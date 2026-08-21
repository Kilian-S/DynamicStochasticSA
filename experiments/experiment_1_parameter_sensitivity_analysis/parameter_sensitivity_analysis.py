from pathlib import Path
import pandas as pd
from dynamic_behaviour import dynamic_sa
from inputs.instance import DISTANCES_FILE, NODES_SHEET, VEHICLE_CAPACITY, read_instance_distance_matrix
from inputs.node import create_nodes_static
from simulated_annealing import objective

# Define parameters
TEMPERATURES = [10, 100, 1000]
ITERATION_LEVELS = [10, 100, 1000]
UTILISATION_TARGETS = [1, 0.9, 0.5, 0.25, 0]
TRIALS = 30

# Results are written next to this script rather than into whichever working directory the script happens to be started from
OUTPUT_DIRECTORY = Path(__file__).resolve().parent


def parameter_sensitivity_analysis(cooling_schedule: str = 'linear'):
    """
    Run the SDCVRP on the deterministic instance across every combination of initial temperature, iteration level and utilisation target, and record the distance and runtime of
    each trial.

    Args:
        cooling_schedule (str): The name of the cooling schedule under test. Also determines the name of the output file.

    """
    nodes = create_nodes_static(DISTANCES_FILE, NODES_SHEET)
    distance_matrix = read_instance_distance_matrix()
    output_file = OUTPUT_DIRECTORY / f'results_{cooling_schedule}.xlsx'
    results = []

    # Run the experiments
    for temp in TEMPERATURES:
        for iteration_level in ITERATION_LEVELS:
            for utilisation_target in UTILISATION_TARGETS:
                for i in range(TRIALS):
                    current_tours_value, current_tours, execution_time, *_ = dynamic_sa(nodes, distance_matrix, objective, temp, iteration_level, VEHICLE_CAPACITY,
                                                                                        utilisation_target, cooling_schedule)

                    # Add results to the list of results
                    results.append({
                        "InitialTemp": temp,
                        "Iterations": iteration_level,
                        "UtilTarget": utilisation_target,
                        "Trial": i + 1,
                        "CurrentValue": current_tours_value,
                        "CurrentTours": str(current_tours),  # Convert list to string to store in DataFrame
                        "ExecutionTime": execution_time
                    })

                # Write out after every configuration so that a long run can be inspected while it is still going
                df = pd.DataFrame(results)
                df.to_excel(output_file, index=False)


if __name__ == '__main__':
    parameter_sensitivity_analysis()
