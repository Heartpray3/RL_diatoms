import argparse
from common.utils import load_config
from pathlib import Path
from ppo.train import main

if __name__ == '__main__':
    base_path = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=str(Path(base_path, 'config.yaml')), help="Chemin vers le fichier de configuration YAML")
    args = parser.parse_args()

    config = load_config(path=args.config)
    main(config)
