import ast
from pathlib import Path
import pandas as pd

# The exact solution indexes nodes by their node family. The dynamic solution indexes them by child node, so the two must be brought onto the same basis before their tours can
# be compared
INPUT_FILE = Path(__file__).resolve().parents[1] / 'results_static.xlsx'
INPUT_COLUMN = 'Tours'
OUTPUT_FILE = Path(__file__).resolve().parent / 'results_static_feasible.xlsx'


def transform_nested_list(nested_list: list[list[int]]) -> list[list[str]]:
    """
    Transform a list of tours of node family indices into a list of tours of child node IDs. The depot is left as it is because it has no child nodes.

    Args:
        nested_list (list[list[int]]): The tours as the exact solver reports them.

    Returns:
        list[list[str]]: The same tours indexed by child node.

    """
    return [[str(i) + '.1' if i != 0 else str(i) for i in sub_list] for sub_list in nested_list]


def make_static_solution_feasible(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    """
    Read the tours of the exact solution, reindex them by child node, and write them back out. Every node is assumed to require a single visit, so each node family contributes
    its first child node. Instances in which a node family needs more than one visit have to be reindexed by hand.

    Args:
        input_file: The Excel file holding the tours of the exact solution.
        output_file: The Excel file the reindexed tours are written to.

    Returns:
        pd.DataFrame: The tours of the exact solution alongside their reindexed counterparts.

    """
    # Read the Excel file into a DataFrame
    df = pd.read_excel(input_file)

    # Apply the transformation to each row in the DataFrame
    df['tours'] = df[INPUT_COLUMN].apply(lambda x: transform_nested_list(ast.literal_eval(x)))

    df.to_excel(output_file, index=False)

    return df


if __name__ == '__main__':
    print(make_static_solution_feasible())
