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
from sim_env import DiatomEnv, Action
from validate import validate_policy


def q_learning(env: DiatomEnv, episodes, steps_per_episode, alpha, gamma, epsilon=0.1, lookahead_steps=1):
    Q = defaultdict(float)

    for ep in range(episodes):
        state = env.reset(ep)
        total_reward = 0

        for step in range(steps_per_episode):  # max steps per episode
            actions = env.get_available_actions(state)

            # ε-greedy policy
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                q_vals = [Q[(state, a)] for a in actions]
                max_q = max(q_vals)
                best_actions = [a for a in actions if Q[(state, a)] == max_q]
                action = random.choice(best_actions)

            next_state, inst_vel = env.step(action)

            # reward = norme de la vitesse
            reward = sum(v**2 for v in inst_vel)**0.5

            # Q-learning update
            next_actions = env.get_available_actions(next_state)
            max_next_q = max([Q[(next_state, a)] for a in next_actions], default=0)

            Q[(state, action)] += alpha * (reward + gamma * max_next_q - Q[(state, action)])

            state = next_state
            total_reward += reward

        print(f"[Episode {ep+1}] Total reward: {total_reward:.4f}")

    return Q


def main(input_file_path, output_directory, nb_blobs, nb_rods, dt, nb_step, nb_episodes, learning_rate, discount_factor, steps_ahead):
    a = 0.183228708092682  # blob radius
    # Initialiser l’environnement
    env = DiatomEnv(
        input_file_path=input_file_path,
        output_dir=output_directory,
        n_rods=nb_rods,
        n_blobs=nb_blobs,
        dt=dt,
        a=a
    )

    # Lancer l'apprentissage Q-learning
    # episodes = 1000
    # steps_per_episode = 40
    Q = q_learning(
        env,
        episodes=nb_episodes,
        steps_per_episode=nb_step,
        alpha=learning_rate,
        gamma=discount_factor,

    )

    # Sauvegarder la table Q
    base_filename = os.path.join(
        output_directory,
        'q_tables',
        f'q_table_{nb_blobs}_blobs_{nb_rods}_rods_{nb_episodes}_ep_{nb_step}_steps_{discount_factor}_g_{learning_rate}_lr'
    )

    # Ajout d'une version si nécessaire
    version = 0
    final_filename = f"{base_filename}_v{version}.pkl"
    while os.path.exists(final_filename):
        version += 1
        final_filename = f"{base_filename}_v{version}.pkl"

    # Sauvegarde de la Q-table
    with open(final_filename, 'wb') as f:
        pickle.dump(dict(Q), f)

    print(f"Q-table sauvegardée dans {final_filename}")

if __name__ == "__main__":

    # test
    parser = argparse.ArgumentParser(description="Apprentissage par renforcement.")
    parser.add_argument("--input_file_path", type=str, required=True, help="Chemin du fichier d'entrée (.dat)")
    parser.add_argument("--output_directory", type=str, required=True, help="Répertoire de sortie")
    parser.add_argument("--nb_blobs", type=int, required=True, help="Nombre de blobs")
    parser.add_argument("--nb_rods", type=int, required=True, help="Nombre de rods")
    parser.add_argument("--dt", type=float, default=0.0025, help="Pas de temps")
    parser.add_argument("--nb_step", type=int, default=41, help="Nombre d'étapes par épisode")
    parser.add_argument("--nb_episodes", type=int, default=1000, help="Nombre total d'épisodes")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Taux d'apprentissage (alpha)")
    parser.add_argument("--discount_factor", type=float, default=0.6, help="Facteur de réduction (gamma)")
    parser.add_argument("--steps_ahead", type=int, default=1, help="Nombre d'étapes futures prises en compte (n-step TD)")

    args = parser.parse_args()

    main(
        input_file_path=args.input_file_path,
        output_directory=args.output_directory,
        nb_blobs=args.nb_blobs,
        nb_rods=args.nb_rods,
        dt=args.dt,
        nb_step=args.nb_step,
        nb_episodes=args.nb_episodes,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        steps_ahead=args.steps_ahead
    )