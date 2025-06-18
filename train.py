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
import random
import pickle
from collections import defaultdict
from sim_env import DiatomEnv
from validate import validate_policy


def q_learning(env: DiatomEnv, episodes=1000, steps_per_episode=200, alpha=0.1, gamma=0.95, epsilon=0.1):
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


def main(input_directory, output_directory, Nblobs, Nrods, dt, Nstep):
    a = 0.183228708092682  # blob radius
    # Initialiser l’environnement
    env = DiatomEnv(
        input_file_path=input_directory,
        output_dir=output_directory,
        n_rods=Nrods,
        n_blobs=Nblobs,
        dt=dt,
        a=a
    )

    # Lancer l'apprentissage Q-learning
    Q = q_learning(env, episodes=200, steps_per_episode=100)

    # Sauvegarder la table Q
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(dict(Q), f)
    print("Q-table sauvegardée dans q_table.pkl")

    # validate_policy(env, Q, episodes=3)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Traite des fichiers avec optimisation d'efficacité.")
    parser.add_argument("--input_directory", type=str, required=True, help= "Fichier de d'entree")
    parser.add_argument("--output_directory", type=str, required=True, help= "Fichier de sortie")
    parser.add_argument("--Nblobs", type=int, required=True, help="Nombre de blobs")
    parser.add_argument("--Nrods", type=int, required=True, help="Nombre de rods")
    parser.add_argument("--dt", type=float, default= 0.00125, help = "Timestep")
    parser.add_argument("--Nstep", type=int, default = 160, help = "Nombre d'étapes")
    parser.add_argument("--freq", type=float, default = 10, help = "fréquence du sinus dans le coulissement")
    
    args = parser.parse_args()
    
    main(args.input_directory, args.output_directory, args.Nblobs, args.Nrods, args.dt, args.Nstep)
