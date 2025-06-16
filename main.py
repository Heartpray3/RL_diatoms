from utils import load_config, abs_path
from My_Generate_all_files_efficiency_optimization7 import main

if __name__ == '__main__':
    config = load_config()

    main(
        config.input_directory,
        config.output_directory,
        config.Nblobs,
        config.phase_shift,
        config.Nrods,
        config.an,
        config.bn,
        config.nmodes,
        config.dt,
        config.Nstep,
        config.freq
    )
