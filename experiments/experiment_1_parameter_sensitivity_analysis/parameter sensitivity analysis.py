import pandas as pd
from dynamic_behaviour import dynamic_sa
from inputs.distances import read_in_distance_matrix
from inputs.node import create_nodes_static
from simulated_annealing import objective

# Create a DataFrame to hold the results
df = pd.DataFrame(columns=["InitialTemp", "IterationLevel", "UtilTarget", "Trial", "CurrentValue", "CurrentTours", "ExecutionTime"])

# Define parameters
temps = [10, 100, 1000]
iterations = [10, 100, 1000]
utilisation_targets = [1, 0.9, 0.5, 0.25, 0]
vehicle_capacity = 2000
nodes = create_nodes_static('../../inputs/distances.xlsx', 'Sheet1')
distance_matrix = read_in_distance_matrix('../../inputs/distances.xlsx', 'Distance matrix (districts)', 'B2', 'AX50')
trials = 30
results = []

# Run the experiments
for temp in temps:
    for iteration_level in iterations:
        for utilisation_target in utilisation_targets:
            for i in range(trials):
                current_tours_value, current_tours, execution_time, *_ = dynamic_sa(nodes, distance_matrix, objective, temp, iteration_level, vehicle_capacity, utilisation_target)

                # Add results to DataFrame
                results.append({
                    "InitialTemp": temp,
                    "Iterations": iteration_level,
                    "UtilTarget": utilisation_target,
                    "Trial": i + 1,
                    "CurrentValue": current_tours_value,
                    "CurrentTours": str(current_tours),  # Convert list to string to store in DataFrame
                    "ExecutionTime": execution_time
                })

            df = pd.DataFrame(results)
            df.to_excel('results_sigmoid.xlsx', index=False)



