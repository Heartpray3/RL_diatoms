import argparse
from common.utils import load_config
from q_learning.train import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help="Chemin vers le fichier de configuration YAML")
    args = parser.parse_args()

    config = load_config(path=args.config)
    main(config)
