from utils import load_config
from train import main

if __name__ == '__main__':
    config = load_config()

    main(
        config.input_file_path,
        config.output_directory,
        config.Nblobs,
        config.Nrods,
        config.dt,
        config.Nstep,
    )
