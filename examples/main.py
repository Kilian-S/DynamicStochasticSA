import os


abs1 = 'C:\Kilian\TUM\TUM\Bachelor Thesis\Code\simulated annealing\experiments\experiment_3_stochastics\stochastics.py'
abs2 = 'C:\Kilian\TUM\TUM\Bachelor Thesis\Code\simulated annealing\experiments\experiment_2_perfect_information\ssults_static.xlsx'

print(os.path.relpath(abs2, start=abs1))
