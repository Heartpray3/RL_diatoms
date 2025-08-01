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
import random

from mpmath.libmp.libmpf import reciprocal_rnd
from scipy.stats import reciprocal
from stable_baselines3 import PPO

from sim_env import DiatomEnv, ColonyState, Action
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
    random.seed(42)
    model = PPO("MlpPolicy", env, n_epochs=10, n_steps=config.nb_step, verbose=1)

    # Entraîner
    model.learn(total_timesteps=config.nb_step * config.nb_episodes)

    # Sauvegarder
    os.makedirs(config.output_directory, exist_ok=True)
    model.save(os.path.join(config.output_directory, "ppo_diatom"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apprentissage par renforcement.")
    parser.add_argument("--input_file_path", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--nb_blobs", type=int, required=True)
    parser.add_argument("--nb_rods", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--nb_step", type=int, default=100)
    parser.add_argument("--nb_episodes", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--discount_factor", type=float, default=0.95)
    parser.add_argument("--lookahead_steps", type=int, default=1)
    parser.add_argument("--reward_method", type=str, default="VELOCITY")
    parser.add_argument("--progression_angle", type=float, default=None)

    args = parser.parse_args()

    config = Config(
        input_file_path=abs_path(args.input_file_path),
        output_directory=abs_path(args.output_directory),
        nb_blobs=args.nb_blobs,
        nb_rods=args.nb_rods,
        dt=args.dt,
        nb_step=args.nb_step,
        nb_episodes=args.nb_episodes,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        lookahead_steps=args.lookahead_steps,
        reward_method=RewardMethod[args.reward_method.upper()],
        reward_angle=args.progression_angle
    )

    main(config)
