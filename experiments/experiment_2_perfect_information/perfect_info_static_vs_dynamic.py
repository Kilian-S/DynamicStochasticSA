from pathlib import Path
import pandas as pd
from static.static_cvrp import exact_algorithm

TRIALS = 30

# Results are written next to this script rather than into whichever working directory the script happens to be started from
OUTPUT_FILE = Path(__file__).resolve().parent / 'results_static_single_thread.xlsx'


def perfect_information_benchmark(trials: int = TRIALS, output_file=OUTPUT_FILE):
    """
    Solve the deterministic instance exactly, once per trial, to establish the benchmark a dynamic solution is measured against. Every trial faces the same instance; the trials
    exist because the solver is stopped at a time limit and so does not return the same incumbent every time.

    Args:
        trials (int): The number of times the exact algorithm is run.
        output_file: The Excel file the results are written to.

    """
    results = []

    for i in range(trials):
        obj_value, tours, execution_time = exact_algorithm()

        results.append({
            "Trial": i + 1,
            "ObjValue": obj_value,
            "Tours": str(tours),
            "ExecutionTime": execution_time
        })

        # Write out after every trial so that a long run can be inspected while it is still going
        df = pd.DataFrame(results)
        df.to_excel(output_file, index=False)


if __name__ == '__main__':
    perfect_information_benchmark()
