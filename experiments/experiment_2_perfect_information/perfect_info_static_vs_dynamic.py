import pandas as pd

from static.static_cvrp import exact_algorithm

df = pd.DataFrame(columns=["Trial", "ObjValue", "Tours", "ExecutionTime"])
trials = 30
results = []

for i in range(trials):
    obj_value, tours, execution_time = exact_algorithm()

    results.append({
        "Trial": i+1,
        "ObjValue": obj_value,
        "Tours": str(tours),
        "ExecutionTime": execution_time
    })

    df = pd.DataFrame(results)
    df.to_excel('results_static.xlsx', index=False)




















