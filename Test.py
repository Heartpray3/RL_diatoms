import os

Nblobs = 14
phase_shift = 2.0
Nrods = 15

my_command = f"python3 Generate_all_files_efficiency_optimization.py --Nblobs {Nblobs} --phase_shift {phase_shift} --Nrods {Nrods}"
os.system(my_command)