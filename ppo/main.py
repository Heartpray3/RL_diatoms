#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 13:00:01 2025

@author: Ely Cheikh Abass

Point d'entrée principal pour l'exécution de l'algorithme PPO.
Lance l'entraînement avec les paramètres spécifiés dans la configuration.
"""

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
