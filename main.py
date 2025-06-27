from utils import load_config
from train import main

if __name__ == '__main__':
    config = load_config()

    main(
        config.input_file_path,
        config.output_directory,
        config.nb_blobs,
        config.nb_rods,
        config.dt,
        config.nb_step,
        config.nb_episodes,
        config.learning_rate,
        config.discount_factor,
        config.lookahead_steps
    )
