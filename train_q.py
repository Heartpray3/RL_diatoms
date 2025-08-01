#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 9 13:00:01 2025

@author: Ely Cheikh Abass

La structure de base est inspire du code de Julien/Stefanie

Code d'apprentissage par renforcement & simulation (Q-learning)
"""

from __future__ import division, print_function
import argparse
import os
import random
import pickle
from collections import defaultdict
from sim_env import DiatomEnv, RewardMethod
from utils import Config, abs_path


def q_learning(env: DiatomEnv,
               episodes,
               steps_per_episode,
               alpha,
               gamma,
               reward_method: RewardMethod,
               epsilon=0.1,
               lookahead_steps=1,
               reward_angle=None
               ):
    Q = defaultdict(float)

    for ep in range(episodes):
        state = env.reset()
        state = env._obs_to_state(state)
        total_reward = 0

        for step in range(steps_per_episode - 1):  # max steps per episode
            actions = env.get_available_actions(state)

            # ε-greedy policy
            if random.random() < epsilon:
                action = random.choice(list(map(env.encode_action, actions)))
            else:
                q_vals = [Q[(state, a)] for a in actions]
                max_q = max(q_vals)
                best_actions = [a for a in actions if Q[(state, a)] == max_q]
                action = random.choice(list(map(env.encode_action, best_actions)))

            next_state, reward, done, *_ = env.step(action)
            next_state = env._obs_to_state(next_state)

            # Q-learning update
            next_actions = env.get_available_actions(next_state)
            max_next_q = max([Q[(next_state, a)] for a in next_actions], default=0)

            Q[(state, action)] += alpha * (reward + gamma * max_next_q - Q[(state, action)])

            state = next_state
            total_reward += reward
            if done:
                break

        print(f"[Episode {ep+1}] Total reward: {total_reward:.4f}")

    return Q


def main(config: Config):
    a = 0.183228708092682  # blob radius

    # Initialiser l’environnement
    env = DiatomEnv(
        input_file_path=config.input_file_path,
        output_dir=config.output_directory,
        n_rods=config.nb_rods,
        n_blobs=config.nb_blobs,
        dt=config.dt,
        a=a,
        reward_method=config.reward_method,
        angle=config.reward_angle,
        nb_steps_per_ep=config.nb_step
    )

    # Lancer l'apprentissage Q-learning
    reward_angle = config.reward_angle if config.reward_method == RewardMethod.FORWARD_PROGRESS else 0.0
    Q = q_learning(
        env,
        episodes=config.nb_episodes,
        steps_per_episode=config.nb_step,
        alpha=config.learning_rate,
        gamma=config.discount_factor,
        reward_method=config.reward_method,
        reward_angle=reward_angle,
    )

    # Construire le nom de base
    base_filename = os.path.join(
        config.output_directory,
        'q_tables',
        f'q_table_{config.nb_blobs}_blobs_{config.nb_rods}_rods_{config.nb_episodes}_ep_{config.nb_step}_steps_{config.discount_factor}_g_{config.learning_rate}_lr'
    )

    # Créer un nom de fichier unique avec versioning
    version = 0
    final_filename = f"{base_filename}_v{version}.pkl"
    while os.path.exists(final_filename):
        version += 1
        final_filename = f"{base_filename}_v{version}.pkl"

    os.makedirs(os.path.dirname(final_filename), exist_ok=True)

    # Sauvegarder la Q-table
    with open(final_filename, 'wb') as f:
        pickle.dump(dict(Q), f)

    print(f"Q-table sauvegardée dans {final_filename}")

if __name__ == "__main__":

    # test
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