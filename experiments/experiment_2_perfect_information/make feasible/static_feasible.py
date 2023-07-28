import pandas as pd
import ast

# Read the Excel file into a DataFrame
df = pd.read_excel('results_static_simple.xlsx')

# This function transforms a list of lists of integers into a list of lists of strings.
def transform_nested_list(nested_list):
    return [[str(i)+'.1' if i != 0 else str(i) for i in sub_list] for sub_list in nested_list]

# Apply the transformation to each row in the DataFrame
df['TransformedList'] = df['tour'].apply(lambda x: transform_nested_list(ast.literal_eval(x)))

# Print the DataFrame to check the result
print(df)

df.to_excel('results_static_feasible.xlsx', index=False)

