#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 9 13:00:01 2025

@author: Ely Cheikh Abass

La structure de base est inspire du code de Julien/Stefanie

Code d'apprentissage par renforcement & simulation (Q-learning)
"""

import argparse
import os
from stable_baselines3 import PPO
from sim_env import DiatomEnv
from utils import Config, abs_path, RewardMethod

def main(config):
    # Créer l'env Gym
    env = DiatomEnv(
        input_file_path=config.input_file_path,
        output_dir=config.output_directory,
        n_rods=config.nb_rods,
        n_blobs=config.nb_blobs,
        reward_method=config.reward_method,
        nb_steps_per_ep=config.nb_step,
        a=0.183228708092682,
        dt=config.dt,
        angle=config.reward_angle
    )

    # Instancier PPO
    model = PPO("MlpPolicy", env, verbose=1)

    # Entraîner
    model.learn(total_timesteps=config.nb_step * config.nb_episodes)

    # Sauvegarder
    os.makedirs(config.output_directory, exist_ok=True)
    model.save(os.path.join(config.output_directory, "ppo_diatom"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", required=True)
    parser.add_argument("--output_directory", required=True)
    parser.add_argument("--nb_blobs", type=int, required=True)
    parser.add_argument("--nb_rods", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--nb_step", type=int, default=100)
    parser.add_argument("--reward_method", type=str, default="VELOCITY")
    args = parser.parse_args()

    config = Config(
        input_file_path=abs_path(args.input_file_path),
        output_directory=abs_path(args.output_directory),
        nb_blobs=args.nb_blobs,
        nb_rods=args.nb_rods,
        dt=args.dt,
        nb_step=args.nb_step,
        nb_episodes=0,     # pas utilisé ici
        learning_rate=0,   # pas utilisé
        discount_factor=0, # pas utilisé
        lookahead_steps=0, # pas utilisé
        reward_method=RewardMethod[args.reward_method.upper()],
        reward_angle=None
    )
    main(config)
