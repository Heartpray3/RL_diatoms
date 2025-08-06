#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 9 13:00:01 2025

@author: Ely Cheikh Abass

Point d'entrée principal pour l'exécution de l'algorithme Q-learning.
Lance l'entraînement avec les paramètres spécifiés dans la configuration.
"""

import argparse
from common.utils import load_config
from q_learning.train import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help="Chemin vers le fichier de configuration YAML")
    args = parser.parse_args()

    config = load_config(path=args.config)
    main(config)
