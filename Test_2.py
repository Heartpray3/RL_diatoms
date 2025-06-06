import os
from My_Generate_all_files_efficiency_optimization7 import main  

if __name__ == '__main__':
    # Paramètres à adapter selon tes besoins
    output_directory = os.path.dirname(__file__) + "/"
    Nblobs = 14
    phase_shift = 2.0
    Nrods = 2
    an = '0.0'
    bn = '1.0'
    nmodes = 1
    dt = 0.0125
    Nstep = 80
    freq = 10

    main(output_directory, Nblobs, phase_shift, Nrods, an, bn, nmodes, dt, Nstep, freq)