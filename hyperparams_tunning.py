import argparse
import os
from utils import load_config, Config, RewardMethod
from pathlib import Path
import subprocess

if __name__ == '__main__':

    base_config = load_config()
    configs = []
    config_params = [(200, 1000), (40, 5000)]
    for params in config_params:
        for j, blobs in enumerate([2, 5, 10]):
            for method, angle in [(RewardMethod.FORWARD_PROGRESS, 90), (RewardMethod.FORWARD_PROGRESS, 0), (RewardMethod.CIRCULAR_ZONES, 0)]:
                nb_steps, nb_epoch = params
                path_output = os.path.join(base_config.output_directory, f"ppo_3r_{blobs}b_ep_{nb_epoch}_step_{nb_steps}_meth_{method.value}_ang_{angle}")
                new_config = Config(
                    input_file_path=base_config.input_file_path,
                    output_directory=path_output,
                    nb_blobs=blobs,
                    nb_rods=3,
                    dt=base_config.dt,
                    nb_step=nb_steps,
                    nb_episodes=nb_epoch,
                    learning_rate=1,
                    discount_factor=0,
                    lookahead_steps=base_config.lookahead_steps,
                    reward_method=method,
                    reward_angle=angle
                )

                # Génère les arguments en ligne de commande
                args_list = [
                    "python3", "train.py",
                    "--input_file_path", str(new_config.input_file_path),
                    "--output_directory", str(new_config.output_directory),
                    "--nb_blobs", str(new_config.nb_blobs),
                    "--nb_rods", str(new_config.nb_rods),
                    "--dt", str(new_config.dt),
                    "--nb_step", str(new_config.nb_step),
                    "--nb_episodes", str(new_config.nb_episodes),
                    "--learning_rate", str(new_config.learning_rate),
                    "--discount_factor", str(new_config.discount_factor),
                    "--lookahead_steps", str(new_config.lookahead_steps),
                    "--reward_method", new_config.reward_method.name,
                    "--progression_angle", str(new_config.reward_angle)
                ]

                # Préparation du log
                log_file = Path(path_output) / "log.txt"
                log_file.parent.mkdir(parents=True, exist_ok=True)
                err_log = open(log_file, 'w')  # ⚠️ Laisse ouvert, géré par subprocess

                # Lancement du processus détaché
                subprocess.Popen(
                    args_list,
                    stdout=subprocess.DEVNULL,
                    stderr=err_log,
                    start_new_session=True
                )

                print(f"🚀 Launched: {path_output}")

